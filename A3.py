# YOLO v8 細胞偵測程式 - 圖片標記

import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# ============================================================================
# 配置參數
# ============================================================================

# 資料夾路徑
START_FOLDER = 'start'           # 要標記的圖片資料夾
FINISH_FOLDER = 'finish'         # 處理完成的圖片資料夾
PREPROCESSED_FOLDER = '預處理'   # 預處理後的圖片資料夾

# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# 模型路徑
MODEL_PATH = 'runs/detect/DB_cell_detection15/weights/best.pt'

# 預處理設定
USE_PREPROCESSING = True  # True = 使用預處理，False = 使用原始圖片

# 過濾設定
FILTER_CONFIG = {
    # 第一次信心度過濾（程式層，類別特定）
    # 注意：YOLO 模型層使用最低值，然後在程式層進行類別特定的第一次過濾
    'first_confidence_by_class': {
        'RFID': 0.01,   # RFID 第一次信心度過濾
        'cell': 0.0058,   # cell 第一次信心度過濾
        'point': 0.018  # point 第一次信心度過濾
    },
    # YOLO 模型層使用最低的信心度值（確保所有類別都能通過）
    'yolo_conf_threshold': 0.001,  # 使用所有類別中的最低值
    # NMS（非極大值抑制）參數：過濾重疊的檢測框
    'yolo_iou_threshold': 0.05,  # YOLO 內建 NMS IoU 閾值（只在同類別內過濾）
    'cross_class_iou_threshold': 0.3,  # 跨類別 NMS IoU 閾值（處理不同類別之間的重疊，如 cell 和 point）
    
    # 面積過濾
    'min_area_by_class': {
        'RFID': 1,     # RFID 最小面積
        'cell': 0.01,     # cell 最小面積
        'point': 0.01     # point 最小面積
    },
    'max_area_ratio': 0.99,  # 最大面積比例（相對於圖片大小）
    
    # 第二次信心度過濾（程式層，類別特定）
    'min_confidence_by_class': {
        'RFID': 0.1,  # RFID 最小信心度（第二次過濾）
        'cell': 0.01,  # cell 最小信心度（第二次過濾）
        'point': 0.01  # point 最小信心度（第二次過濾）
    }
}

# 類別顏色設定 (BGR 格式)
CLASS_COLORS = {
    'RFID': (255, 255, 170),  # RFID, 藍色
    'cell': (170, 255, 255),  # 菌落, 黃色
    'point': (255, 170, 255)  # 疙瘩, 紫色
}

# 類別映射
CLASS_MAPPING = {
    0: 'RFID',
    1: 'cell',
    2: 'point'
}

# 預處理參數
PREPROCESS_PARAMS = {
    'bilateral_d': 9,
    'bilateral_sigmaColor': 75,
    'bilateral_sigmaSpace': 75,
    'clahe_clipLimit': 2.0,
    'clahe_tileGridSize': (8, 8),
    'sharpen_weight': 0.6,
    'enhanced_weight': 0.4
}

# ============================================================================
# 預處理函數
# ============================================================================

def preprocess_image(image):
    """
    預處理圖片以增強菌落、標籤等對比度
    使用 去噪 + CLAHE (對比度限制自適應直方圖均衡化) + 銳化 + 混合
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的圖片（BGR 格式）
    """
    result = image.copy()
    
    # 1. 去噪處理（雙邊濾波：去噪但保留邊緣）
    denoised = cv2.bilateralFilter(
        result,
        d=PREPROCESS_PARAMS['bilateral_d'],
        sigmaColor=PREPROCESS_PARAMS['bilateral_sigmaColor'],
        sigmaSpace=PREPROCESS_PARAMS['bilateral_sigmaSpace']
    )
    
    # 2. 轉換到 LAB 色彩空間（分離亮度和顏色）
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. 對 L 通道（亮度）應用 CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=PREPROCESS_PARAMS['clahe_clipLimit'],
        tileGridSize=PREPROCESS_PARAMS['clahe_tileGridSize']
    )
    l_clahe = clahe.apply(l)
    
    # 4. 合併回 LAB 並轉回 BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # 5. 輕微銳化（強化邊緣，但不要太強）
    kernel_sharpen = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 6. 混合原圖和銳化圖（避免過度銳化）
    result = cv2.addWeighted(
        sharpened,
        PREPROCESS_PARAMS['sharpen_weight'],
        enhanced,
        PREPROCESS_PARAMS['enhanced_weight'],
        0
    )
    
    return result

# ============================================================================
# 資料處理函數
# ============================================================================

def get_image_paths(folder):
    """獲取資料夾中所有圖片路徑"""
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_images = glob.glob(os.path.join(folder, ext))
        # 過濾已標記的檔案（以 detected_ 開頭）
        image_paths.extend([
            img for img in all_images 
            if not os.path.basename(img).startswith('detected_')
        ])
    return image_paths

def ensure_folders():
    """確保必要的資料夾存在"""
    os.makedirs(START_FOLDER, exist_ok=True)
    os.makedirs(FINISH_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

# ============================================================================
# 偵測和過濾函數
# ============================================================================

def detect_objects(model, detection_image, conf_threshold, iou_threshold):
    """使用 YOLO 模型偵測物體"""
    results = model(
        detection_image, 
        conf=conf_threshold, 
        iou=iou_threshold,  # NMS IoU 閾值：過濾重疊的檢測框
        verbose=False
    )
    detections = results[0]
    num_boxes = len(detections.boxes) if detections.boxes is not None else 0
    return detections, num_boxes

def print_detection_info(detections, num_boxes):
    """列印偵測結果的診斷資訊"""
    if num_boxes > 0:
        
        # 統計各 class_id 的數量
        class_id_counts = {}
        for box in detections.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            class_id_counts[cls_id] = class_id_counts.get(cls_id, 0) + 1
        
    else:
        print(f"  ⚠️  警告：YOLO 沒有偵測到任何物體！")

def parse_detection_box(box, detections, imgwidth, imgheight):
    """解析單個偵測框的資訊"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    confidence = float(box.conf[0].cpu().numpy())
    class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
    
    # 獲取類別名稱
    model_class_name = detections.names[class_id] if hasattr(detections, 'names') else 'cell'
    class_name = CLASS_MAPPING.get(class_id, model_class_name)
    
    # 計算尺寸和面積
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2
    area = w * h
    
    return {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'w': w, 'h': h, 'cx': cx, 'cy': cy,
        'area': area, 'confidence': confidence,
        'class': class_name, 'class_id': class_id
    }

def validate_and_clip_coordinates(obj, imgwidth, imgheight):
    """驗證並裁剪座標到圖片範圍內"""
    x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
    
    # 裁剪到圖片範圍內
    x1 = max(0, min(x1, imgwidth - 1))
    y1 = max(0, min(y1, imgheight - 1))
    x2 = max(x1 + 1, min(x2, imgwidth))
    y2 = max(y1 + 1, min(y2, imgheight))
    
    # 重新計算尺寸
    w = x2 - x1
    h = y2 - y1
    area = w * h
    cx = x1 + w // 2
    cy = y1 + h // 2
    
    obj.update({
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'w': w, 'h': h, 'cx': cx, 'cy': cy, 'area': area
    })
    
    return obj

def calculate_iou(box1, box2):
    """計算兩個檢測框之間的 IoU（Intersection over Union）"""
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
    
    # 計算交集區域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 計算並集區域
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def apply_cross_class_nms(detected_objects, iou_threshold):
    """應用跨類別 NMS，過濾不同類別之間重疊的檢測框"""
    if len(detected_objects) <= 1:
        return detected_objects
    
    # 按信心度降序排序
    sorted_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while sorted_objects:
        # 取出信心度最高的框
        current = sorted_objects.pop(0)
        keep.append(current)
        
        # 過濾與當前框重疊的其他框（不論類別）
        remaining = []
        for obj in sorted_objects:
            iou = calculate_iou(current, obj)
            if iou <= iou_threshold:
                remaining.append(obj)
        
        sorted_objects = remaining
    
    return keep

def filter_detections(detections, imgwidth, imgheight):
    """過濾偵測結果（包含第一次和第二次信心度過濾）"""
    detected_objects = []
    class_counts = {'RFID': 0, 'cell': 0, 'point': 0}
    max_area = imgwidth * imgheight * FILTER_CONFIG['max_area_ratio']
    
    if detections.boxes is None or len(detections.boxes) == 0:
        return detected_objects, class_counts
    
    for box in detections.boxes:
        obj = parse_detection_box(box, detections, imgwidth, imgheight)
        class_name = obj['class']
        
        # 獲取過濾條件
        min_area = FILTER_CONFIG['min_area_by_class'].get(class_name, 20)
        first_confidence = FILTER_CONFIG['first_confidence_by_class'].get(class_name, 0.01)  # 第一次過濾
        min_confidence = FILTER_CONFIG['min_confidence_by_class'].get(class_name, 0.1)  # 第二次過濾
        
        # 驗證條件
        area_ok = min_area <= obj['area'] <= max_area
        first_conf_ok = obj['confidence'] >= first_confidence  # 第一次信心度過濾
        second_conf_ok = obj['confidence'] >= min_confidence  # 第二次信心度過濾
        conf_ok = first_conf_ok and second_conf_ok  # 兩次過濾都要通過
        coord_ok = (obj['x1'] >= 0 and obj['y1'] >= 0 and 
                   obj['x2'] > obj['x1'] and obj['y2'] > obj['y1'] and
                   obj['x2'] <= imgwidth and obj['y2'] <= imgheight and
                   obj['w'] > 0 and obj['h'] > 0)
        
        # 如果通過過濾，加入結果
        if area_ok and conf_ok and coord_ok:
            obj = validate_and_clip_coordinates(obj, imgwidth, imgheight)
            detected_objects.append(obj)
            if class_name in class_counts:
                class_counts[class_name] += 1
    
    return detected_objects, class_counts

# ============================================================================
# 標記和儲存函數
# ============================================================================

def draw_bounding_boxes(image, detected_objects, imgwidth, imgheight):
    """在圖片上繪製標記框"""
    marked_count = 0
    marked_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    
    for obj in detected_objects:
        if obj["x1"] < obj["x2"] and obj["y1"] < obj["y2"]:
            class_name = obj['class']
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # 繪製矩形框
            cv2.rectangle(image, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), color, 2)
            
            # 添加標籤
            label = f"{class_name} {obj['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            
            # 標籤背景框
            cv2.rectangle(
                image,
                (obj["x1"], obj["y1"] - text_height - 5),
                (obj["x1"] + text_width, obj["y1"]),
                color, -1
            )
            
            # 白色文字
            cv2.putText(
                image, label, (obj["x1"], obj["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            
            marked_count += 1
            if class_name in marked_by_class:
                marked_by_class[class_name] += 1
    
    # 在左下角顯示統計
    text_width = 400
    text_height = 25
    cv2.rectangle(
        image, (10, imgheight - text_height),
        (10 + text_width, imgheight), (0, 0, 0), -1
    )
    
    stats_text = f"RFID:{marked_by_class['RFID']} cell:{marked_by_class['cell']} point:{marked_by_class['point']}"
    cv2.putText(
        image, stats_text, (15, imgheight - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )
    
    return marked_count, marked_by_class

def save_results(image, detected_objects, image_path, imgwidth, imgheight):
    """儲存標記後的圖片和標註檔"""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 儲存標記後的圖片
    output_image_path = os.path.join(FINISH_FOLDER, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)
    
    # 儲存 YOLO 格式的標註檔
    txt_output_path = os.path.join(FINISH_FOLDER, f"{image_name}_detected.txt")
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for obj in detected_objects:
            class_id = obj['class_id']
            norm_center_x = max(0, min(1, obj['cx'] / imgwidth))
            norm_center_y = max(0, min(1, obj['cy'] / imgheight))
            norm_width = max(0, min(1, obj['w'] / imgwidth))
            norm_height = max(0, min(1, obj['h'] / imgheight))
            
            f.write(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"  標註檔已儲存: {txt_output_path}")

# ============================================================================
# 主處理函數
# ============================================================================

def process_image(image_path, model):
    """處理單張圖片並標記細胞"""
    print(f"\n處理: {os.path.basename(image_path)}")
    
    # 讀取圖片
    image_src = cv2.imread(image_path)
    if image_src is None:
        print(f"  錯誤：無法讀取圖片 {image_path}")
        return False
    
    imgheight = image_src.shape[0]
    imgwidth = image_src.shape[1]
    
    # 預處理（如果需要）
    if USE_PREPROCESSING:
        print(f"  正在預處理圖片...")
        image_preprocessed = preprocess_image(image_src)
        
        # 儲存預處理後的圖片
        preprocessed_path = os.path.join(PREPROCESSED_FOLDER, os.path.basename(image_path))
        cv2.imwrite(preprocessed_path, image_preprocessed)
        detection_image = image_preprocessed
    else:
        detection_image = image_src
    
    # YOLO 偵測
    detections, num_boxes = detect_objects(
        model, 
        detection_image, 
        FILTER_CONFIG['yolo_conf_threshold'],
        FILTER_CONFIG['yolo_iou_threshold']  # 加入 IoU 閾值以過濾重疊框
    )
    print_detection_info(detections, num_boxes)
    
    # 過濾偵測結果
    detected_objects, class_counts = filter_detections(detections, imgwidth, imgheight)
    
    # 應用跨類別 NMS，過濾不同類別之間重疊的框（如 cell 和 point）
    detected_objects = apply_cross_class_nms(
        detected_objects, 
        FILTER_CONFIG['cross_class_iou_threshold']
    )
    
    # 重新計算類別計數
    class_counts = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        class_name = obj['class']
        if class_name in class_counts:
            class_counts[class_name] += 1
    
    print(f"  過濾後結果 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']} (原始偵測: {num_boxes} 個)")
    
    # 在圖片上標記
    image = image_src.copy()
    marked_count, marked_by_class = draw_bounding_boxes(image, detected_objects, imgwidth, imgheight)
    
    # 儲存結果
    save_results(image, detected_objects, image_path, imgwidth, imgheight)
    
    return True

# ============================================================================
# 主程式
# ============================================================================

def main():
    """主程式流程"""
    print("=" * 60)
    print("YOLO v8 細胞偵測 - 圖片標記")
    print("=" * 60)
    
    # 確保資料夾存在
    ensure_folders()
    
    # 獲取圖片路徑
    image_paths = get_image_paths(START_FOLDER)
    
    if not image_paths:
        print(f"錯誤：在 {START_FOLDER} 資料夾中找不到圖片檔案！")
        print(f"目前 {START_FOLDER} 資料夾路徑：{os.path.abspath(START_FOLDER)}")
        return
    
    print(f"找到 {len(image_paths)} 張圖片，開始處理...")
    print("=" * 60)
    
    # 載入模型
    print("正在載入 YOLO v8 模型...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"模型載入成功: {MODEL_PATH}")
    except Exception as e:
        print(f"錯誤：無法載入模型 {MODEL_PATH}")
        print(f"錯誤訊息: {e}")
        return
    
    # 處理所有圖片
    success_count = 0
    for image_path in image_paths:
        if process_image(image_path, model):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"處理完成！成功處理 {success_count}/{len(image_paths)} 張圖片")
    print("=" * 60)

if __name__ == "__main__":
    main()
