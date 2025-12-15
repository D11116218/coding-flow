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
MODEL_PATH = r"C:\Users\willes.chen\BB\runs\detect\DB_cell_detection1\weights\best.pt"

# 預處理設定
USE_PREPROCESSING = True  # True = 使用預處理，False = 使用原始圖片
PREPROCESSED_FOLDER_A3 = '預處裡A3'  # A3 預處理輸出路徑

# 過濾設定
FILTER_CONFIG = {
    # 信心度過濾（程式層，類別特定）
    # 注意：YOLO 模型層使用最低值，然後在程式層進行類別特定的過濾
    'first_confidence_by_class': {
        # 提高類別特定的信心度門檻，降低低分框殘留
        'RFID': 0.85,   # 原 0.049 → 0.55（保持高信心）
        'cell': 0.84,   # 原 0.135 → 0.18
        'point': 0.67       # 原 0.0047 → 0.10
    },
    # YOLO 模型層使用最低的信心度值（確保所有類別都能通過）
    'yolo_conf_threshold': 0.05,  # 原 0.001 → 0.05，先在模型層砍掉極低分框
    # NMS（非極大值抑制）參數：過濾重疊的檢測框
    'yolo_iou_threshold': 0.45,   # YOLO 內建 NMS IoU 閾值（只在同類別內過濾）
    'cross_class_iou_threshold': 0.5,  # 跨類別 NMS IoU 閾值（0.1 太低會過度過濾，正常範圍 0.5-0.6）
    
    # 面積過濾
    'min_area_by_class': {
        'RFID': 0.001,   # RFID 最小面積比例
        'cell': 0.001,   # cell 最小面積比例
        'point': 0.001   # point 最小面積比例（）
    },
    'max_area_ratio': 0.99,  # 最大面積比例（相對於圖片大小）

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

# 預處理參數（GIMP 風格，根據實際 GIMP 設定調整）
PREPROCESS_PARAMS = {
    # 顏色轉灰階參數（GIMP 的顏色轉灰階設定）
    # 注意：GIMP 的顏色轉灰階有 Radius/Samples/Iterations 參數，這是特殊算法
    # 我們使用 GIMP 亮度方法作為基礎
    'grayscale_radius': 300,        # GIMP Radius = 300
    'grayscale_samples': 4,         # GIMP Samples = 4
    'grayscale_iterations': 10,     # GIMP Iterations = 10
    'grayscale_enhance_shadows': False,  # GIMP Enhance Shadows = 未勾選
    
    # 銳利化參數（對應 GIMP 的銳利化設定）
    'sharpen_radius': 3.0,          # 銳化半徑（GIMP Radius = 3.000）
    'sharpen_amount': 5.527,        # 銳化強度（GIMP Amount = 5.527）
    'sharpen_threshold': 0.0,       # 銳化閾值（GIMP Threshold = 0.000）
    
    # 降低雜訊參數（對應 GIMP 的降低雜訊設定）
    'denoise_h': 11.0,              # 過濾強度（GIMP Strength = 11）
    'denoise_templateWindowSize': 7, # 模板窗口大小（必須為奇數，建議 5-9）
    'denoise_searchWindowSize': 21,  # 搜索窗口大小（必須為奇數，建議 15-25）
}

# ============================================================================
# 預處理函數
# ============================================================================

def convert_to_grayscale_gimp(image):
    """
    使用 GIMP 亮度方法轉換為灰階
    注意：GIMP 的「顏色轉灰階」有 Radius/Samples/Iterations 參數，這是特殊算法
    我們使用標準亮度方法，並可選添加高斯模糊模擬 Radius 效果
    
    參數：
        image: BGR 格式圖片
    
    返回：
        灰階圖片
    """
    # 1. 使用 GIMP 亮度公式轉換為灰階
    b, g, r = cv2.split(image)
    gray = (0.114 * b.astype(np.float32) + 
            0.587 * g.astype(np.float32) + 
            0.299 * r.astype(np.float32)).astype(np.uint8)
    return gray

def sharpen_image_unsharp_mask(image, radius, amount, threshold=0.0):
    """
    使用 Unsharp Mask 方法銳化圖片（GIMP 風格）
    使用與 GIMP 相同的公式：sharpened = original + (original - blurred) * amount
    
    參數：
        image: 灰階圖片
        radius: 銳化半徑（GIMP 的 Radius）
        amount: 銳化強度（GIMP 的 Amount）
        threshold: 銳化閾值（GIMP 的 Threshold）
    
    返回：
        銳化後的圖片
    """
    # 轉換為 float32 以進行精確計算
    img_float = image.astype(np.float32)
    
    # 計算高斯模糊（GIMP 使用 sigma = radius）
    # 計算 kernel 大小：通常使用 6*sigma+1，確保為奇數
    kernel_size = int(6 * radius + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), radius)
    
    # GIMP 的 Unsharp Mask 公式：sharpened = original + (original - blurred) * amount
    diff = img_float - blurred
    sharpened = img_float + diff * amount
    
    # 應用閾值（如果設定）
    if threshold > 0:
        # 計算差異的絕對值
        diff_abs = np.abs(diff)
        # 只對差異大於閾值的區域進行銳化
        mask = diff_abs > threshold
        sharpened = np.where(mask, sharpened, img_float)
    
    # 限制數值範圍到 [0, 255] 並轉回 uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

def denoise_image(image, strength, template_window_size, search_window_size):
    """
    使用非局部均值去噪降低雜訊（GIMP 風格）
    
    參數：
        image: 灰階圖片
        strength: 過濾強度（GIMP 的 Strength，對應 OpenCV 的 h 參數）
        template_window_size: 模板窗口大小
        search_window_size: 搜索窗口大小
    
    返回：
        去噪後的圖片
    """
    # GIMP 的 Strength 參數範圍通常是 0-10，對應 OpenCV 的 h 參數
    # 直接使用 strength 作為 h 值
    return cv2.fastNlMeansDenoising(
        image,
        h=float(strength),
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )

def preprocess_image(image):
    """
    預處理圖片（與 train_DB 相同）
    依序使用：顏色轉灰階、銳利化、降低雜訊
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的灰階圖片
    """
    # 1. 顏色轉灰階（GIMP 亮度方法）
    gray = convert_to_grayscale_gimp(image)
    
    # 2. 銳利化（Unsharp Mask）
    sharpened = sharpen_image_unsharp_mask(
        gray,
        PREPROCESS_PARAMS['sharpen_radius'],
        PREPROCESS_PARAMS['sharpen_amount'],
        PREPROCESS_PARAMS['sharpen_threshold']
    )
    
    # 3. 降低雜訊（非局部均值去噪）
    denoised = denoise_image(
        sharpened,
        PREPROCESS_PARAMS['denoise_h'],  # GIMP 的 Strength 參數
        PREPROCESS_PARAMS['denoise_templateWindowSize'],
        PREPROCESS_PARAMS['denoise_searchWindowSize']
    )
    
    return denoised

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
    os.makedirs(PREPROCESSED_FOLDER_A3, exist_ok=True)

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
        # 統計各 class_id 的數量與信心度
        class_id_counts = {}
        class_id_confidences = {}
        for box in detections.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            confidence = float(box.conf[0].cpu().numpy())
            class_id_counts[cls_id] = class_id_counts.get(cls_id, 0) + 1
            class_id_confidences.setdefault(cls_id, []).append(confidence)
        
        # 顯示各類別統計
        for cls_id, count in class_id_counts.items():
            class_name = CLASS_MAPPING.get(cls_id, f"class_{cls_id}")
            confs = class_id_confidences.get(cls_id, [])
            if confs:
                avg_conf = sum(confs) / len(confs)
                min_conf = min(confs)
                max_conf = max(confs)
                print(f"  {class_name} (class_id={cls_id}): {count} 個, 信心度範圍: {min_conf:.4f} ~ {max_conf:.4f}, 平均: {avg_conf:.4f}")
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
    """跨類別 NMS，並對類別給優先權：RFID > cell > point"""
    if len(detected_objects) <= 1:
        return detected_objects
    
    # 類別優先權（RFID > cell > point）
    class_priority = {'RFID': 2, 'cell': 1, 'point': 0}
    
    # 按信心度降序排序
    sorted_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while sorted_objects:
        # 取出信心度最高的框
        current = sorted_objects.pop(0)
        # 與已保留的框做比較，如重疊則依類別優先權/信心度決定保留誰
        new_keep = []
        keep_current = True
        for kept in keep:
            iou = calculate_iou(current, kept)
            if iou > iou_threshold:
                p_cur = class_priority.get(current['class'], 0)
                p_keep = class_priority.get(kept['class'], 0)
                
                # 若 current 類別優先或同優先但信心度較高，則替換原框
                if (p_cur > p_keep) or (p_cur == p_keep and current['confidence'] > kept['confidence']):
                    continue  # 丟掉 kept，改保留 current
                else:
                    keep_current = False  # 保留 kept，丟掉 current
                    new_keep.append(kept)
            else:
                new_keep.append(kept)
        
        if keep_current:
            new_keep.append(current)
        
        keep = new_keep
    
    return keep

def filter_detections(detections, imgwidth, imgheight):
    """過濾偵測結果（使用信心度過濾）"""
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
        confidence = FILTER_CONFIG['first_confidence_by_class'].get(class_name, 0.01)  # 信心度過濾
        
        # 驗證條件
        area_ok = min_area <= obj['area'] <= max_area
        conf_ok = obj['confidence'] >= confidence  # 信心度過濾
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
            
            # 黑色文字
            cv2.putText(
                image, label, (obj["x1"], obj["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
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
        
        # 儲存預處理後的圖片到「預處裡A3」資料夾（灰階）
        preprocessed_path_a3 = os.path.join(PREPROCESSED_FOLDER_A3, os.path.basename(image_path))
        cv2.imwrite(preprocessed_path_a3, image_preprocessed)

        # 與 train_DB 一致：預處理產物為灰階，送入模型前轉為 BGR 三通道
        detection_image = cv2.cvtColor(image_preprocessed, cv2.COLOR_GRAY2BGR)
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
    print(f"  程式層過濾後 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']}")
    
    # 應用跨類別 NMS，過濾不同類別之間重疊的框（如 cell 和 point）
    before_cross_nms_count = len(detected_objects)
    detected_objects = apply_cross_class_nms(
        detected_objects, 
        FILTER_CONFIG['cross_class_iou_threshold']
    )
    after_cross_nms_count = len(detected_objects)
    if before_cross_nms_count != after_cross_nms_count:
        print(f"  跨類別 NMS 過濾掉 {before_cross_nms_count - after_cross_nms_count} 個框")
    
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