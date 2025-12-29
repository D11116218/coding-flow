#!/home/dssignal/coding-flow/.venv/bin/python3
# YOLOv12 細胞偵測程式 - 圖片標記

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 配置參數
# ============================================================================

# 資料夾路徑
START_FOLDER = 'start'           # 要標記的圖片資料夾
FINISH_FOLDER = 'finish'         # 處理完成的圖片資料夾
PREPROCESSED_FOLDER = '預處理'   # 預處理後的圖片資料夾

# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# YOLO 模型設定
MODEL_SIZE = 'l'  # 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
YOLO_MODEL_PATH = f"runs/DB_cell_detection12/weights/best.pt"  # YOLO 訓練好的模型路徑
IMGSZ = 640  # 圖片尺寸，必須與訓練時一致（train_DB.py 中的 IMGSZ）

# 預處理設定
USE_PREPROCESSING = True  # True = 使用預處理（與訓練時一致），False = 使用原始圖片
PREPROCESSED_FOLDER_A3 = '預處裡A3'  # A3 預處理輸出路徑

# 過濾設定
FILTER_CONFIG = {
    # 信心度過濾（程式層，類別特定）
    # 注意：YOLO 模型層使用最低值，然後在程式層進行類別特定的過濾
    'first_confidence_by_class': {
        # 降低類別特定的信心度門檻，讓更多檢測結果通過（因模型信心值偏低）
        'RFID': 0.5,   # RFID 保持較高閾值
        'cell': 0.01,  # 大幅降低 cell 信心度閾值，讓更多 cell 通過
        'point': 0.01  # 大幅降低 point 信心度閾值，讓更多 point 通過
    },
    # YOLO 模型層使用最低的信心度值
    'yolo_conf_threshold': 0.001,  # 提高以過濾極低分框，但仍保持較低門檻
    # NMS（非極大值抑制）數：過濾重疊的檢測框
    'yolo_iou_threshold': 0.1,   # 降低 YOLO 內建 NMS IoU 閾值，從 0.15 降到 0.1，保留更多檢測框
    'same_class_nms_threshold': 0.05,  # 進一步降低同類別 NMS IoU 閾值，保留更多重疊的 cell 和 point
    'cross_class_iou_threshold': 0.7,  # 大幅提高跨類別 NMS IoU 閾值，幾乎不進行跨類別過濾（只過濾高度重疊的框）
    'rfid_same_class_nms_threshold': 0.1,  # RFID 專用同類別 NMS 閾值（越高越積極過濾）
    
    # 面積過濾
    'min_area_by_class': {
        'RFID': 0.001,   # RFID  最小面積比例
        'cell': 0.001,   # cell  最小面積比例
        'point': 0.001   # point 最小面積比例
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

# 類別順序映射（從類別名稱到索引）
CLASS_ORDER = {
    'RFID': 0,
    'cell': 1,
    'point': 2
}

# 預處理參數（GIMP 風格，與 train_DB.py 保持一致）
PREPROCESS_PARAMS = {
    # 顏色轉灰階參數
    'grayscale_radius': 300,        # GIMP Radius = 300
    'grayscale_samples': 4,         # GIMP Samples = 4
    'grayscale_iterations': 10,     # GIMP Iterations = 10
    'grayscale_enhance_shadows': False,  # GIMP Enhance Shadows = 未勾選
    
    # 銳利化參數
    'sharpen_radius': 3.0,          # 銳化半徑
    'sharpen_amount': 2.5,          # 銳化強度
    'sharpen_threshold': 0.0,       # 銳化閾值
    
    # 降低雜訊參數
    'denoise_h': 11.0,              # 過濾強度
    'denoise_templateWindowSize': 7, # 模板窗口大小
    'denoise_searchWindowSize': 21,  # 搜索窗口大小
    
    # 對比度增強參數（用於加深黑點）
    'enhance_contrast': True,        # 是否啟用對比度增強
    'contrast_alpha': 1.5,           # 對比度係數（1.0 = 無變化，>1.0 = 增強對比度）
    'contrast_beta': 0,              # 亮度調整（0 = 無變化）
    'gamma_correction': 0.8,         # 伽馬校正值（<1.0 = 增強暗部，讓黑點更深）
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
    
    # 應用閾值 
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

def enhance_contrast(image, alpha, beta):
    """
    增強對比度（線性變換）
    
    參數：
        image: 輸入圖片（灰階）
        alpha: 對比度係數（1.0 = 無變化，>1.0 = 增強對比度）
        beta: 亮度調整（0 = 無變化）
    
    返回：
        增強對比度後的圖片
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_gamma_correction(image, gamma):
    """
    應用伽馬校正（增強暗部，讓黑點更深）
    
    參數：
        image: 輸入圖片（灰階）
        gamma: 伽馬值（<1.0 = 增強暗部，>1.0 = 增強亮部）
    
    返回：
        伽馬校正後的圖片
    """
    # 建立查找表
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    # 應用查找表
    return cv2.LUT(image, table)

def preprocess_image(image):
    """
    預處理圖片（與 train_DB.py 完全一致）
    依序使用：顏色轉灰階、銳利化、降低雜訊、對比度增強、伽馬校正
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的 BGR 三通道圖片（與 train_DB.py 一致，符合 YOLO 要求）
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
    
    # 4. 對比度增強（加深黑點）
    if PREPROCESS_PARAMS.get('enhance_contrast', False):
        enhanced = enhance_contrast(
            denoised,
            PREPROCESS_PARAMS.get('contrast_alpha', 1.5),
            PREPROCESS_PARAMS.get('contrast_beta', 0)
        )
    else:
        enhanced = denoised
    
    # 5. 伽馬校正（進一步增強暗部，讓黑點更深）
    gamma = PREPROCESS_PARAMS.get('gamma_correction', 1.0)
    if gamma != 1.0:
        final = apply_gamma_correction(enhanced, gamma)
    else:
        final = enhanced
    
    # 與 train_DB.py 一致：預處理並輸出三通道以符合 YOLO 要求
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

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

def detect_objects_yolo(model, detection_image, conf_threshold, iou_threshold):
    """使用 YOLO 模型偵測物體"""
    results = model(
        detection_image, 
        conf=conf_threshold, 
        iou=iou_threshold,  # NMS IoU 閾值：過濾重疊的檢測框
        imgsz=IMGSZ,  # 指定圖片尺寸，必須與訓練時一致
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

def is_box_inside(inner_box, outer_box):
    """
    檢查 inner_box 是否完全在 outer_box 內
    
    參數：
        inner_box: 內部框（字典，包含 x1, y1, x2, y2）
        outer_box: 外部框（字典，包含 x1, y1, x2, y2）
    
    返回：
        True 如果 inner_box 完全在 outer_box 內，否則 False
    """
    return (inner_box['x1'] >= outer_box['x1'] and 
            inner_box['y1'] >= outer_box['y1'] and 
            inner_box['x2'] <= outer_box['x2'] and 
            inner_box['y2'] <= outer_box['y2'])

def apply_same_class_nms(detected_objects, iou_threshold):
    """同類別 NMS：過濾同一類別內重疊的框，保留信心度最高的"""
    if len(detected_objects) <= 1:
        return detected_objects
    
    # 按類別分組
    by_class = {}
    for obj in detected_objects:
        class_name = obj['class']
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(obj)
    
    # 對每個類別分別做 NMS
    result = []
    for class_name, objects in by_class.items():
        if len(objects) <= 1:
            result.extend(objects)
            continue
        
        # 按信心度降序排序
        sorted_objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        while sorted_objects:
            current = sorted_objects.pop(0)
            keep_current = True
            
            for kept in keep:
                iou = calculate_iou(current, kept)
                if iou > iou_threshold:
                    # 重疊，保留信心度更高的（current 已經排序過，所以 current 信心度更高）
                    keep_current = False
                    break
            
            if keep_current:
                keep.append(current)
        
        result.extend(keep)
    
    return result

def apply_cross_class_nms(detected_objects, iou_threshold):
    """跨類別 NMS，根據信心度決定保留哪個框"""
    if len(detected_objects) <= 1:
        return detected_objects
    
    # 特殊處理：如果 RFID 和 cell 重疊，保留兩者（由後續的 RFID 框內過濾處理）
    # 按信心度降序排序
    sorted_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while sorted_objects:
        # 取出信心度最高的框
        current = sorted_objects.pop(0)
        # 與已保留的框做比較，如重疊則依信心度決定保留誰
        new_keep = []
        keep_current = True
        for kept in keep:
            iou = calculate_iou(current, kept)
            if iou > iou_threshold:
                # 特殊處理：如果 RFID 和 cell/point 重疊，保留兩者（由後續的 RFID 框內過濾處理）
                if (current['class'] == 'RFID' and kept['class'] in ['cell', 'point']) or \
                   (current['class'] in ['cell', 'point'] and kept['class'] == 'RFID'):
                    # 保留兩者，不互相過濾
                    new_keep.append(kept)
                    continue
                
                # 其他情況：根據信心度決定保留哪個框
                if current['confidence'] > kept['confidence']:
                    continue  # 丟掉 kept，改保留 current（信心度更高）
                else:
                    keep_current = False  # 保留 kept，丟掉 current（kept 信心度更高或相等）
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
    
    # YOLO 格式
    if detections.boxes is None or len(detections.boxes) == 0:
        return detected_objects, class_counts
    
    for box in detections.boxes:
        obj = parse_detection_box(box, detections, imgwidth, imgheight)
        class_name = obj['class']
        
        # 獲取過濾條件
        min_area = imgwidth * imgheight * FILTER_CONFIG['min_area_by_class'].get(class_name, 0.001)
        confidence = FILTER_CONFIG['first_confidence_by_class'].get(class_name, 0.01)
        
        # 驗證條件
        area_ok = min_area <= obj['area'] <= max_area
        conf_ok = obj['confidence'] >= confidence
        coord_ok = (obj['x1'] >= 0 and obj['y1'] >= 0 and 
                   obj['x2'] > obj['x1'] and obj['y2'] > obj['y1'] and
                   obj['x2'] <= imgwidth and obj['y2'] <= imgheight and
                   obj['w'] > 0 and obj['h'] > 0)
        
        # 如果通過過濾，加入結果
        if area_ok and conf_ok and coord_ok:
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
        detection_image = preprocess_image(image_src)
        
        # 儲存預處理後的圖片到「預處裡A3」資料夾（BGR 三通道）
        preprocessed_path_a3 = os.path.join(PREPROCESSED_FOLDER_A3, os.path.basename(image_path))
        cv2.imwrite(preprocessed_path_a3, detection_image)
    else:
        detection_image = image_src
    
    # 使用 YOLO 模型進行偵測
    # 對於 RFID，使用更低的 NMS 閾值以保留更多候選框
    detections, num_boxes = detect_objects_yolo(
        model, 
        detection_image, 
        FILTER_CONFIG['yolo_conf_threshold'],
        FILTER_CONFIG['yolo_iou_threshold']
    )
    print_detection_info(detections, num_boxes)
    
    # 診斷：顯示所有原始偵測結果（在過濾前）
    if detections.boxes is not None and len(detections.boxes) > 0:
        # 統計各類別的原始偵測數量
        raw_counts = {'RFID': 0, 'cell': 0, 'point': 0}
        raw_confidences = {'RFID': [], 'cell': [], 'point': []}
        for box in detections.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            conf = float(box.conf[0].cpu().numpy())
            class_name = CLASS_MAPPING.get(cls_id, f"class_{cls_id}")
            if class_name in raw_counts:
                raw_counts[class_name] += 1
                raw_confidences[class_name].append(conf)
        
        print(f"  🔍 原始 YOLO 偵測結果（過濾前）:")
        for class_name in ['RFID', 'cell', 'point']:
            if raw_counts[class_name] > 0:
                confs = raw_confidences[class_name]
                avg_conf = sum(confs) / len(confs)
                min_conf = min(confs)
                max_conf = max(confs)
                print(f"    {class_name}: {raw_counts[class_name]} 個, 信心度範圍: {min_conf:.4f} ~ {max_conf:.4f}, 平均: {avg_conf:.4f}")
    
    # 過濾偵測結果
    detected_objects, class_counts = filter_detections(detections, imgwidth, imgheight)
    print(f"  程式層信心度過濾後 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']}")
    
    # 診斷：顯示過濾後的 cell 和 point 詳細資訊
    cell_after_filter = [obj for obj in detected_objects if obj['class'] == 'cell']
    point_after_filter = [obj for obj in detected_objects if obj['class'] == 'point']
    if len(cell_after_filter) > 0:
        print(f"  🔍 過濾後的 cell（共 {len(cell_after_filter)} 個）:")
        for i, cell in enumerate(cell_after_filter[:5]):  # 只顯示前 5 個
            print(f"    [{i+1}] conf={cell['confidence']:.4f}, pos=({cell['x1']},{cell['y1']}), size={cell['w']}x{cell['h']}")
    if len(point_after_filter) > 0:
        print(f"  🔍 過濾後的 point（共 {len(point_after_filter)} 個）:")
        for i, point in enumerate(point_after_filter[:5]):  # 只顯示前 5 個
            print(f"    [{i+1}] conf={point['confidence']:.4f}, pos=({point['x1']},{point['y1']}), size={point['w']}x{point['h']}")
    
    # 診斷：顯示被過濾掉的 cell 和 point（信心度不足）
    if detections.boxes is not None and len(detections.boxes) > 0:
        filtered_cells = []
        filtered_points = []
        for box in detections.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            conf = float(box.conf[0].cpu().numpy())
            class_name = CLASS_MAPPING.get(cls_id, f"class_{cls_id}")
            min_conf = FILTER_CONFIG['first_confidence_by_class'].get(class_name, 0.01)
            
            if class_name == 'cell' and conf < min_conf:
                filtered_cells.append(conf)
            elif class_name == 'point' and conf < min_conf:
                filtered_points.append(conf)
        
        if len(filtered_cells) > 0:
            print(f"  ⚠️  有 {len(filtered_cells)} 個 cell 因信心度 < {FILTER_CONFIG['first_confidence_by_class']['cell']} 被過濾")
            if len(filtered_cells) <= 5:
                print(f"    被過濾的 cell 信心度: {[f'{c:.4f}' for c in filtered_cells]}")
        if len(filtered_points) > 0:
            print(f"  ⚠️  有 {len(filtered_points)} 個 point 因信心度 < {FILTER_CONFIG['first_confidence_by_class']['point']} 被過濾")
            if len(filtered_points) <= 5:
                print(f"    被過濾的 point 信心度: {[f'{c:.4f}' for c in filtered_points]}")
    
    # 診斷：顯示過濾後的 RFID 候選框
    rfid_after_filter = [obj for obj in detected_objects if obj['class'] == 'RFID']
    if len(rfid_after_filter) > 0:
        print(f"  🔍 過濾後的 RFID 候選框（共 {len(rfid_after_filter)} 個）:")
        for i, rfid in enumerate(rfid_after_filter):
            print(f"    [{i+1}] conf={rfid['confidence']:.3f}, pos=({rfid['x1']},{rfid['y1']}), size={rfid['w']}x{rfid['h']}")
    
    # 診斷：檢查 cell 框之間的重疊情況
    cell_objects = [obj for obj in detected_objects if obj['class'] == 'cell']
    if len(cell_objects) > 1:
        print(f"  🔍 檢查 {len(cell_objects)} 個 cell 框的重疊情況：")
        for i, obj1 in enumerate(cell_objects):
            for j, obj2 in enumerate(cell_objects[i+1:], start=i+1):
                iou = calculate_iou(obj1, obj2)
                if iou > 0.3:  # 顯示 IoU > 0.3 的重疊
                    print(f"    cell {i+1} (conf={obj1['confidence']:.2f}) 與 cell {j+1} (conf={obj2['confidence']:.2f}) IoU={iou:.3f}")
    
    # 程式層同類別 NMS：針對同類別內的重疊框再做一次過濾
    # 對 RFID 使用更寬鬆的 NMS 閾值，保留更多候選框
    before_same_nms = len(detected_objects)
    before_same_nms_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        if obj['class'] in before_same_nms_by_class:
            before_same_nms_by_class[obj['class']] += 1
    
    # 分開處理 RFID 和其他類別
    rfid_objects_before_nms = [obj for obj in detected_objects if obj['class'] == 'RFID']
    other_objects_before_nms = [obj for obj in detected_objects if obj['class'] != 'RFID']
    
    # RFID 使用更寬鬆的 NMS
    rfid_after_nms = apply_same_class_nms(rfid_objects_before_nms, FILTER_CONFIG['rfid_same_class_nms_threshold'])
    # 其他類別使用標準 NMS
    other_after_nms = apply_same_class_nms(other_objects_before_nms, FILTER_CONFIG['same_class_nms_threshold'])
    
    detected_objects = rfid_after_nms + other_after_nms
    after_same_nms = len(detected_objects)
    after_same_nms_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        if obj['class'] in after_same_nms_by_class:
            after_same_nms_by_class[obj['class']] += 1
    
    if before_same_nms != after_same_nms:
        print(f"  同類別 NMS 過濾掉 {before_same_nms - after_same_nms} 個重疊框")
        for class_name in ['RFID', 'cell', 'point']:
            removed = before_same_nms_by_class[class_name] - after_same_nms_by_class[class_name]
            if removed > 0:
                print(f"    - {class_name}: {before_same_nms_by_class[class_name]} -> {after_same_nms_by_class[class_name]} (移除 {removed} 個)")
    
    # 應用跨類別 NMS，過濾不同類別之間重疊的框（如 cell 和 point）
    before_cross_nms_count = len(detected_objects)
    before_cross_nms_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        if obj['class'] in before_cross_nms_by_class:
            before_cross_nms_by_class[obj['class']] += 1
    
    detected_objects = apply_cross_class_nms(
        detected_objects, 
        FILTER_CONFIG['cross_class_iou_threshold']
    )
    after_cross_nms_count = len(detected_objects)
    after_cross_nms_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        if obj['class'] in after_cross_nms_by_class:
            after_cross_nms_by_class[obj['class']] += 1
    
    if before_cross_nms_count != after_cross_nms_count:
        print(f"  跨類別 NMS 過濾掉 {before_cross_nms_count - after_cross_nms_count} 個框 (IoU threshold: {FILTER_CONFIG['cross_class_iou_threshold']})")
        for class_name in ['RFID', 'cell', 'point']:
            removed = before_cross_nms_by_class[class_name] - after_cross_nms_by_class[class_name]
            if removed > 0:
                print(f"    - {class_name}: {before_cross_nms_by_class[class_name]} -> {after_cross_nms_by_class[class_name]} (移除 {removed} 個)")
    else:
        print(f"  跨類別 NMS 未過濾任何框 (IoU threshold: {FILTER_CONFIG['cross_class_iou_threshold']})")
    
    # 針對 RFID 選擇最佳的一個（考慮信心度和位置）
    rfid_objects = [obj for obj in detected_objects if obj['class'] == 'RFID']
    other_objects = [obj for obj in detected_objects if obj['class'] != 'RFID']
    
    if len(rfid_objects) > 1:
        # RFID 選擇策略：綜合考慮信心度、位置和大小
        # 優先選擇：信心度高 + 位置合理（靠近邊緣或中心區域）+ 長寬比合理
        def rfid_score(obj):
            """計算 RFID 的綜合分數"""
            conf_score = obj['confidence']  # 信心度權重
            
            # 位置分數：RFID 通常在圖片邊緣或特定位置
            # 計算距離圖片邊緣的距離（越小越好）
            edge_dist = min(
                obj['x1'], obj['y1'], 
                imgwidth - obj['x2'], 
                imgheight - obj['y2']
            )
            # 標準化到 0-1（假設圖片邊緣 10% 區域是 RFID 常見位置）
            edge_score = max(0, 1 - edge_dist / (min(imgwidth, imgheight) * 0.1))
            
            # 長寬比分數：RFID 通常是矩形，長寬比約 2:1 到 4:1
            aspect_ratio = max(obj['w'], obj['h']) / max(min(obj['w'], obj['h']), 1)
            aspect_score = 1.0 if 1.5 <= aspect_ratio <= 5.0 else max(0, 1 - abs(aspect_ratio - 3) / 3)
            
            # 綜合分數：信心度 70%，位置 20%，長寬比 10%
            return conf_score * 0.7 + edge_score * 0.2 + aspect_score * 0.1
        
        # 按綜合分數排序
        rfid_objects.sort(key=rfid_score, reverse=True)
        kept_rfid = rfid_objects[0]
        removed_count = len(rfid_objects) - 1
        
        # 顯示所有候選 RFID 的資訊
        print(f"  RFID 候選框 ({len(rfid_objects)} 個):")
        for i, rfid in enumerate(rfid_objects[:3]):  # 只顯示前 3 個
            score = rfid_score(rfid)
            print(f"    [{i+1}] conf={rfid['confidence']:.3f}, pos=({rfid['x1']},{rfid['y1']}), size={rfid['w']}x{rfid['h']}, score={score:.3f}")
        
        print(f"  RFID 選擇：保留分數最高的 (conf={kept_rfid['confidence']:.4f}, pos=({kept_rfid['x1']},{kept_rfid['y1']}))，移除 {removed_count} 個")
        detected_objects = [kept_rfid] + other_objects
    elif len(rfid_objects) == 1:
        # 只有一個 RFID，檢查是否需要位置修正
        rfid = rfid_objects[0]
        center_x, center_y = imgwidth // 2, imgheight // 2
        rfid_center_x, rfid_center_y = rfid['cx'], rfid['cy']
        dist_from_center = ((rfid_center_x - center_x)**2 + (rfid_center_y - center_y)**2)**0.5
        max_dist = ((imgwidth/2)**2 + (imgheight/2)**2)**0.5
        
        # 檢查原始 YOLO 偵測結果中是否有其他 RFID 候選框被過濾掉了
        # 如果當前 RFID 位置不合理，嘗試從原始結果中尋找更好的候選框
        if dist_from_center / max_dist < 0.3:  # 如果 RFID 太靠近中心
            print(f"  ⚠️  RFID 位置警告：RFID 位於圖片中心附近 (距離中心 {dist_from_center/max_dist*100:.1f}%)")
            print(f"  🔍 嘗試從原始 YOLO 結果中尋找其他 RFID 候選框...")
            
            # 從原始 detections 中尋找所有 RFID（即使信心度較低）
            alternative_rfids = []
            if detections.boxes is not None:
                for box in detections.boxes:
                    cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    if cls_id == 0:  # RFID
                        conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 計算位置分數（距離邊緣越近越好）
                        edge_dist = min(x1, y1, imgwidth - x2, imgheight - y2)
                        edge_score = max(0, 1 - edge_dist / (min(imgwidth, imgheight) * 0.1))
                        
                        # 如果這個候選框位置更好（更靠近邊緣），且信心度 > 0.3
                        if edge_score > 0.5 and conf > 0.3:
                            alt_rfid = parse_detection_box(box, detections, imgwidth, imgheight)
                            alternative_rfids.append((alt_rfid, conf, edge_score))
            
            if len(alternative_rfids) > 0:
                # 按位置分數排序，選擇位置最好的
                alternative_rfids.sort(key=lambda x: x[2], reverse=True)
                best_alt = alternative_rfids[0][0]
                print(f"  ✓ 找到更好的 RFID 候選框：conf={alternative_rfids[0][1]:.3f}, pos=({best_alt['x1']},{best_alt['y1']}), edge_score={alternative_rfids[0][2]:.3f}")
                rfid_objects = [best_alt]
            else:
                print(f"  ℹ️  未找到更好的 RFID 候選框，使用當前偵測結果")
        
        detected_objects = rfid_objects + other_objects
    else:
        detected_objects = rfid_objects + other_objects
    
    # 過濾：移除 RFID 框內的 cell 和 point
    # 重新從 detected_objects 中提取 RFID（確保使用最終選擇的 RFID）
    rfid_objects = [obj for obj in detected_objects if obj['class'] == 'RFID']
    if len(rfid_objects) > 0:
        rfid_box = rfid_objects[0]  # 使用保留的 RFID 框
        before_rfid_filter = len(detected_objects)
        
        # 過濾條件：移除完全在 RFID 框內的，或中心點在 RFID 框內的 cell/point
        def should_remove(obj):
            """判斷是否應該移除這個物件"""
            if obj['class'] == 'RFID':
                return False  # 不移除 RFID 本身
            
            # 檢查 1：是否完全在 RFID 框內
            if is_box_inside(obj, rfid_box):
                return True
            
            # 檢查 2：中心點是否在 RFID 框內（處理部分重疊的情況）
            cx_in = rfid_box['x1'] <= obj['cx'] <= rfid_box['x2']
            cy_in = rfid_box['y1'] <= obj['cy'] <= rfid_box['y2']
            if cx_in and cy_in:
                return True
            
            # 檢查 3：計算 IoU，如果 IoU > 0.5 則認為重疊太多，應該移除（降低閾值，保留更多 cell 和 point）
            iou = calculate_iou(obj, rfid_box)
            if iou > 0.5:
                return True
            
            return False
        
        before_rfid_filter_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
        for obj in detected_objects:
            if obj['class'] in before_rfid_filter_by_class:
                before_rfid_filter_by_class[obj['class']] += 1
        
        detected_objects = [
            obj for obj in detected_objects 
            if not should_remove(obj)
        ]
        after_rfid_filter = len(detected_objects)
        after_rfid_filter_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
        for obj in detected_objects:
            if obj['class'] in after_rfid_filter_by_class:
                after_rfid_filter_by_class[obj['class']] += 1
        
        removed_count = before_rfid_filter - after_rfid_filter
        if removed_count > 0:
            print(f"  RFID 框內過濾：移除 {removed_count} 個在 RFID 框內的 cell/point 標記")
            for class_name in ['cell', 'point']:
                removed = before_rfid_filter_by_class[class_name] - after_rfid_filter_by_class[class_name]
                if removed > 0:
                    print(f"    - {class_name}: {before_rfid_filter_by_class[class_name]} -> {after_rfid_filter_by_class[class_name]} (移除 {removed} 個)")
    
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
    model_size_name = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'extra large'}.get(MODEL_SIZE, 'extra large')
    print(f"YOLOv12{MODEL_SIZE.upper()} ({model_size_name}) 細胞偵測 - 圖片標記")
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
    
    # 載入 YOLO 模型
    model_size_name = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'extra large'}.get(MODEL_SIZE, 'extra large')
    print(f"正在載入 YOLOv12{MODEL_SIZE.upper()} ({model_size_name}) 模型...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"模型載入成功: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"錯誤：無法載入模型 {YOLO_MODEL_PATH}")
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