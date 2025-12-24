#!/usr/bin/env python3
# YOLOv12n 細胞偵測模型訓練程式 - DB 訓練集

import os
import shutil
import cv2
import numpy as np
import glob
from ultralytics import YOLO

# ============================================================================
# 配置類別
# ============================================================================

class DatasetConfig:
    """資料集路徑配置"""
    DATASET_PATH = 'DB'
    TRAIN_IMAGES_DIR = 'DB/images/train'
    TRAIN_LABELS_DIR = 'DB/labels/train'
    VAL_IMAGES_DIR = 'DB/images/val'
    VAL_LABELS_DIR = 'DB/labels/val'
    DATASET_YAML = 'DB_dataset.yaml'
    
    # 預處理資料夾路徑
    PREPROCESSED_FOLDER = 'DB預處理'
    
    @classmethod
    def get_preprocessed_train_dir(cls):
        return os.path.join(cls.PREPROCESSED_FOLDER, 'images', 'train')
    
    @classmethod
    def get_preprocessed_val_dir(cls):
        return os.path.join(cls.PREPROCESSED_FOLDER, 'images', 'val')
    
    @classmethod
    def get_preprocessed_labels_train_dir(cls):
        return os.path.join(cls.PREPROCESSED_FOLDER, 'labels', 'train')
    
    @classmethod
    def get_preprocessed_labels_val_dir(cls):
        return os.path.join(cls.PREPROCESSED_FOLDER, 'labels', 'val')
    
    PREPROCESSED_YAML = 'DB預處理_dataset.yaml'


class TrainConfig:
    """訓練配置"""
    BASE_MODEL = 'yolov12n.pt'
    EPOCHS = 500
    IMGSZ = 640
    BATCH = 16
    NAME = 'DB_cell_detection1'
    PROJECT = 'runs'  # 模型儲存路徑（會儲存在 runs/detect/ 下）
    PATIENCE = 20
    
    @classmethod
    def get_augmentation_params(cls):
        """
        獲取資料擴增參數    
        注意：這些擴增技術可以增加資料多樣性，提高模型的泛化能力，但過度擴增可能導致訓練不穩定
        
        停用某個擴增項目：將該參數的值設為 0 即可（例如：'flipud': 0）
        """
        return {
            'degrees': 3.0,        # 旋轉角度：±3°（設為 0 停用）
            'translate': 0.1,      # 平移：最多 10% 的圖片尺寸（設為 0 停用）
            'scale': 0.15,         # 縮放：85% ~ 115%（設為 0 停用）
            # 'flipud': 0.5,         # 上下翻轉機率：50%（設為 0 停用）
            # 'fliplr': 0.5,         # 左右翻轉機率：50%（設為 0 停用）
            'mosaic': 0.3,         # Mosaic 擴增機率：30%（設為 0 停用）
            'mixup': 0.1,          # Mixup 擴增機率：10%（設為 0 停用）
        }


class PreprocessConfig:A3.py

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

# 過濾設定
FILTER_CONFIG = {
    # 信心度過濾（程式層，類別特定）
    # 注意：YOLO 模型層使用最低值，然後在程式層進行類別特定的過濾
    'first_confidence_by_class': {
        'RFID': 0.49,    # RFID 信心度過濾(0.49 ok)
        'cell': 0.15,   # cell 信心度過濾(0.15 ok)
        'point': 0.01   # point 信心度過濾(確認中)
    },
    # YOLO 模型層使用最低的信心度值（確保所有類別都能通過）
    'yolo_conf_threshold': 0.001,  # 使用所有類別中的最低值
    # NMS（非極大值抑制）參數：過濾重疊的檢測框
    'yolo_iou_threshold': 0.1,   # YOLO 內建 NMS IoU 閾值（只在同類別內過濾）
    'cross_class_iou_threshold': 0.7,  # 跨類別 NMS IoU 閾值
    
    # 面積過濾
    'min_area_by_class': {
        'RFID': 0.008,   # RFID 最小面積比例
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
    'sharpen_amount': 5.7,        # 銳化強度（GIMP Amount = 5.527）
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
    
    # 2. GIMP 的顏色轉灰階有 Radius 參數（300），這可能影響邊緣處理
    # 如果 radius 很大，可能需要特殊處理，但標準亮度方法已經足夠
    # 注意：GIMP 的 Radius/Samples/Iterations 是特殊算法，難以完全複製
    # 我們使用標準亮度方法作為基礎
    
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
    預處理圖片（GIMP 風格）
    依序使用：顏色轉灰階、銳利化、降低雜訊
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的圖片（BGR 格式，三個通道相同，以符合 YOLO 模型的 3 通道輸入要求）
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
    
    # 將灰階轉回 BGR（三個通道相同，以符合 YOLO 模型的 3 通道輸入要求）
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
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
 
train_DB.py

# YOLO v8 細胞偵測模型訓練程式 - DB 訓練集

from ultralytics import YOLO
import os
import shutil
import cv2
import numpy as np
import glob

# ============================================================================
# 配置參數
# ============================================================================

# 資料集路徑
DATASET_PATH = 'DB'
TRAIN_IMAGES_DIR = 'DB/images/train'
TRAIN_LABELS_DIR = 'DB/labels/train'
VAL_IMAGES_DIR = 'DB/images/val'
VAL_LABELS_DIR = 'DB/labels/val'
DATASET_YAML = 'DB_dataset.yaml'

# 預處理資料夾路徑
PREPROCESSED_FOLDER = 'DB預處理'
PREPROCESSED_TRAIN_DIR = os.path.join(PREPROCESSED_FOLDER, 'images', 'train')
PREPROCESSED_VAL_DIR = os.path.join(PREPROCESSED_FOLDER, 'images', 'val')
PREPROCESSED_LABELS_TRAIN_DIR = os.path.join(PREPROCESSED_FOLDER, 'labels', 'train')
PREPROCESSED_LABELS_VAL_DIR = os.path.join(PREPROCESSED_FOLDER, 'labels', 'val')
PREPROCESSED_YAML = 'DB預處理_dataset.yaml'

# 訓練配置
TRAIN_CONFIG = {
    'base_model': 'yolov8n.pt',
    'data_yaml': DATASET_YAML,
    'epochs': 200,        # 最大訓練輪數（設定為80，確保至少能跑30次）
    'imgsz': 640,
    'batch': 16,
    'name': 'DB_cell_detection1',
    'patience': 20,     # 早停耐心值（從25增加到80，避免過早停止）
}

# 資料擴增配置
# 合併兩種優化配置：
# 1. 更保守的幾何變換（有助於小物體檢測）
# 2. 降低顏色增強強度（避免過度擴增）
AUGMENTATION_CONFIG = {
    # 顏色增強（降低強度，避免過度擴增）
    'hsv_h': 0.01,      # 從 0.015 降低到 0.01
    'hsv_s': 0.6,       # 從 0.7 降低到 0.6
    'hsv_v': 0.3,       # 從 0.4 降低到 0.3
    
    # 幾何變換（更保守，有助於小物體檢測）
    'degrees': 3.0,      # 從 5.0 降低到 3.0（更保守）
    'translate': 0.1,   # 保持不變
    'scale': 0.15,      # 從 0.2 降低到 0.15（更保守）
    'shear': 2.0,       # 保持不變
    'perspective': 0.0005,  # 保持不變
    
    # 翻轉
    'flipud': 0.5,
    'fliplr': 0.5,
    
    # 進階增強
    'mosaic': 0.7,      # 從 0.5 增加到 0.7（更多馬賽克，有助於小物體）
    'mixup': 0.1,       # 從 0.15 降低到 0.1（避免過度擴增）
    'copy_paste': 0.0,  # 保持關閉
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
    'denoise_searchWindowSize': 21  # 搜索窗口大小（必須為奇數，建議 15-25）
}

# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# ============================================================================
# 工具函數
# ============================================================================

def find_image_files(directory):
    """
    查找目錄中的所有圖片檔案
    
    參數：
        directory: 目錄路徑
    
    返回：
        圖片檔案路徑列表
    """
    if not os.path.exists(directory):
        return []
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
    
    return image_files

def find_label_files(directory):
    """
    查找目錄中的所有標籤檔案
    
    參數：
        directory: 目錄路徑
    
    返回：
        標籤檔案路徑列表
    """
    if not os.path.exists(directory):
        return []
    
    return glob.glob(os.path.join(directory, '*.txt'))

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
    
    # 2. GIMP 的顏色轉灰階有 Radius 參數（300），這可能影響邊緣處理
    # 如果 radius 很大，可能需要特殊處理，但標準亮度方法已經足夠
    # 注意：GIMP 的 Radius/Samples/Iterations 是特殊算法，難以完全複製
    # 我們使用標準亮度方法作為基礎
    
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
    預處理圖片（GIMP 風格）
    依序使用：顏色轉灰階、銳利化、降低雜訊
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的圖片（灰階格式）
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

def preprocess_images_in_directory(directory):
    """
    對目錄中的所有圖片進行預處理
    
    參數：
        directory: 圖片目錄路徑
    
    返回：
        處理的圖片數量
    """
    image_files = find_image_files(directory)
    count = 0
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is not None:
            preprocessed = preprocess_image(image)
            cv2.imwrite(img_path, preprocessed)
            count += 1
    
    return count

# ============================================================================
# 檔案操作函數
# ============================================================================

def copy_files(source_dir, dest_dir, file_list):
    """
    複製檔案列表到目標目錄
    
    參數：
        source_dir: 來源目錄
        dest_dir: 目標目錄
        file_list: 檔案路徑列表
    
    返回：
        複製的檔案數量
    """
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file_path, dest_path)
        count += 1
    
    return count

def setup_preprocessed_folder():
    """設置預處理資料夾結構"""
    if os.path.exists(PREPROCESSED_FOLDER):
        shutil.rmtree(PREPROCESSED_FOLDER)
    
    os.makedirs(PREPROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_LABELS_TRAIN_DIR, exist_ok=True)
    
    if os.path.exists(VAL_IMAGES_DIR):
        os.makedirs(PREPROCESSED_VAL_DIR, exist_ok=True)
        os.makedirs(PREPROCESSED_LABELS_VAL_DIR, exist_ok=True)

def copy_images_to_preprocessed_folder():
    """複製訓練集圖片和標籤到 DB預處理 資料夾"""
    print("\n正在複製圖片和標籤到 DB預處理 資料夾...")
    
    setup_preprocessed_folder()
    
    # 複製訓練圖片
    train_images = find_image_files(TRAIN_IMAGES_DIR)
    train_count = copy_files(TRAIN_IMAGES_DIR, PREPROCESSED_TRAIN_DIR, train_images)
    
    # 複製訓練標籤
    train_labels = find_label_files(TRAIN_LABELS_DIR)
    label_count = copy_files(TRAIN_LABELS_DIR, PREPROCESSED_LABELS_TRAIN_DIR, train_labels)
    
    print(f"  已複製 {train_count} 張訓練圖片和 {label_count} 個標籤到 {PREPROCESSED_TRAIN_DIR}")
    
    # 複製驗證圖片和標籤（如果存在）
    val_count = 0
    val_label_count = 0
    
    if os.path.exists(VAL_IMAGES_DIR):
        val_images = find_image_files(VAL_IMAGES_DIR)
        val_count = copy_files(VAL_IMAGES_DIR, PREPROCESSED_VAL_DIR, val_images)
        
        if os.path.exists(VAL_LABELS_DIR):
            val_labels = find_label_files(VAL_LABELS_DIR)
            val_label_count = copy_files(VAL_LABELS_DIR, PREPROCESSED_LABELS_VAL_DIR, val_labels)
        
        if val_count > 0:
            print(f"  已複製 {val_count} 張驗證圖片和 {val_label_count} 個標籤到 {PREPROCESSED_VAL_DIR}")
    
    return train_count + val_count

def preprocess_images_in_folder():
    """對 DB預處理 資料夾中的圖片進行預處理"""
    print("\n正在預處理 DB預處理 資料夾中的圖片...")
    
    # 處理訓練圖片
    train_count = preprocess_images_in_directory(PREPROCESSED_TRAIN_DIR)
    print(f"  已預處理 {train_count} 張訓練圖片")
    
    # 處理驗證圖片（如果存在）
    val_count = 0
    if os.path.exists(PREPROCESSED_VAL_DIR):
        val_count = preprocess_images_in_directory(PREPROCESSED_VAL_DIR)
        if val_count > 0:
            print(f"  已預處理 {val_count} 張驗證圖片")
    
    return train_count + val_count

def create_preprocessed_yaml():
    """創建預處理資料集的 yaml 檔案"""
    print("\n正在創建預處理資料集設定檔...")
    
    yaml_content = f"""# DB預處理 訓練集設定檔

# 資料集路徑（相對於此檔案）
path: ./{PREPROCESSED_FOLDER}
train: images/train
val: images/val

# 類別名稱
names:
  0: RFID
  1: cell
  2: point

# 類別數量
nc: 3
"""
    
    with open(PREPROCESSED_YAML, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  已創建預處理資料集設定檔: {PREPROCESSED_YAML}")

# ============================================================================
# 資料檢查函數
# ============================================================================

def check_dataset():
    """檢查資料集是否存在且有效"""
    if not os.path.exists(DATASET_PATH):
        print("錯誤：找不到 DB 資料夾！")
        print("請確認 DB 資料夾已建立。")
        return False
    
    train_images_exist = os.path.exists(TRAIN_IMAGES_DIR) and len(find_image_files(TRAIN_IMAGES_DIR)) > 0
    train_labels_exist = os.path.exists(TRAIN_LABELS_DIR) and len(find_label_files(TRAIN_LABELS_DIR)) > 0
    
    if not train_images_exist:
        print("錯誤：DB/images/train/ 資料夾為空或不存在！")
        print("請放入訓練圖片。")
        return False
    
    if not train_labels_exist:
        print("錯誤：DB/labels/train/ 資料夾中沒有 TXT 標註檔！")
        return False
    
    return True

def get_statistics():
    """獲取資料集統計資訊"""
    train_images = find_image_files(TRAIN_IMAGES_DIR)
    train_labels = find_label_files(TRAIN_LABELS_DIR)
    val_images = find_image_files(VAL_IMAGES_DIR)
    
    return {
        'train_images': len(train_images),
        'train_labels': len(train_labels),
        'val_images': len(val_images)
    }

def print_statistics(stats):
    """複製資料集統計資訊"""
    print(f"\n訓練資料統計：")
    print(f"  訓練圖片：{stats['train_images']} 張")
    print(f"  標註檔：{stats['train_labels']} 個")
    
    if stats['train_images'] != stats['train_labels']:
        print(f"\n警告：圖片數量 ({stats['train_images']}) 與標註檔數量 ({stats['train_labels']}) 不一致！")
        print("請確認每張圖片都有對應的標註檔。")
    
    if stats['val_images'] > 0:
        print(f"  驗證圖片：{stats['val_images']} 張")
    else:
        print(f"  驗證圖片：無（可選）")

# ============================================================================
# 訓練函數
# ============================================================================

def load_model():
    """載入預訓練模型"""
    print("\n正在載入預訓練模型...")
    try:
        model = YOLO(TRAIN_CONFIG['base_model'])
        print("模型載入成功！")
        return model
    except Exception as e:
        print(f"錯誤：無法載入模型: {e}")
        return None

def print_train_config():
    """列印訓練設定"""
    print("\n預處理設定（GIMP 風格，根據實際 GIMP 設定）：")
    print("  - 步驟 1：顏色轉灰階（GIMP 亮度方法：0.299*R + 0.587*G + 0.114*B）")
    print(f"    GIMP 設定：Radius={PREPROCESS_PARAMS['grayscale_radius']}, Samples={PREPROCESS_PARAMS['grayscale_samples']}, Iterations={PREPROCESS_PARAMS['grayscale_iterations']}")
    print(f"  - 步驟 2：銳利化（Unsharp Mask，Radius={PREPROCESS_PARAMS['sharpen_radius']}, Amount={PREPROCESS_PARAMS['sharpen_amount']}, Threshold={PREPROCESS_PARAMS['sharpen_threshold']}）")
    print(f"  - 步驟 3：降低雜訊（非局部均值去噪，Strength={PREPROCESS_PARAMS['denoise_h']}）")
    
    print("\n資料擴增設定（優化配置）：")
    print("  - 顏色增強：色調±1.0%、飽和度±60%、亮度±30%（降低強度，避免過度擴增）")
    print("  - 翻轉：上下翻轉（50%機率）、左右翻轉（50%機率）")
    print("  - 幾何變換：開啟（旋轉±3度、平移±10%、縮放0.85-1.15倍、剪切±2度、透視變換）")
    print("    → 更保守的設定，有助於小物體（point）檢測")
    print("  - 馬賽克增強：開啟（70%機率，增加以提升小物體檢測能力）")
    print("  - 混合增強：開啟（10%機率，降低以避免過度擴增）")

def train_model(model):
    """訓練模型"""
    print("\n開始訓練...")
    print("-" * 60)
    
    try:
        results = model.train(
            data=TRAIN_CONFIG['data_yaml'],
            epochs=TRAIN_CONFIG['epochs'],
            imgsz=TRAIN_CONFIG['imgsz'],
            batch=TRAIN_CONFIG['batch'],
            name=TRAIN_CONFIG['name'],
            patience=TRAIN_CONFIG['patience'],
            save=True,
            plots=True,
            
            # 資料擴增參數
            hsv_h=AUGMENTATION_CONFIG['hsv_h'],
            hsv_s=AUGMENTATION_CONFIG['hsv_s'],
            hsv_v=AUGMENTATION_CONFIG['hsv_v'],
            degrees=AUGMENTATION_CONFIG['degrees'],
            translate=AUGMENTATION_CONFIG['translate'],
            scale=AUGMENTATION_CONFIG['scale'],
            shear=AUGMENTATION_CONFIG['shear'],
            perspective=AUGMENTATION_CONFIG['perspective'],
            flipud=AUGMENTATION_CONFIG['flipud'],
            fliplr=AUGMENTATION_CONFIG['fliplr'],
            mosaic=AUGMENTATION_CONFIG['mosaic'],
            mixup=AUGMENTATION_CONFIG['mixup'],
            copy_paste=AUGMENTATION_CONFIG['copy_paste'],
        )
        
        print("\n" + "=" * 60)
        print("訓練完成！")
        print("=" * 60)
        print(f"最後一個檢查點: {results.save_dir}/weights/last.pt")
        print(f"最佳模型: {results.save_dir}/weights/best.pt")
        
        return results
        
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        return None

# ============================================================================
# 主程式
# ============================================================================

def main():
    """主程式流程"""
    print("=" * 60)
    print("YOLO v8 細胞偵測模型訓練 - DB 訓練集")
    print("=" * 60)
    
    # 1. 檢查資料集
    if not check_dataset():
        exit(1)
    
    # 2. 統計資料
    stats = get_statistics()
    print_statistics(stats)
    
    # 3. 載入模型
    model = load_model()
    if model is None:
        exit(1)
    
    # 4. 顯示訓練設定
    print_train_config()
    
    # 5. 複製圖片和標籤到預處理資料夾
    copy_images_to_preprocessed_folder()
    
    # 6. 預處理圖片
    preprocess_images_in_folder()
    
    # 7. 創建預處理資料集設定檔
    create_preprocessed_yaml()
    
    # 8. 訓練模型
    original_data_yaml = TRAIN_CONFIG['data_yaml']
    TRAIN_CONFIG['data_yaml'] = PREPROCESSED_YAML
    
    results = train_model(model)
    
    TRAIN_CONFIG['data_yaml'] = original_data_yaml
    
    if results is None:
        exit(1)
    
    print("\n" + "=" * 60)
    print("所有處理完成！")
    print(f"預處理後的圖片已儲存在: {PREPROCESSED_FOLDER}")
    print(f"預處理資料集設定檔: {PREPROCESSED_YAML}")
    print("=" * 60)

if __name__ == "__main__":
    main()


    """預處理參數配置（GIMP 風格）"""
    # 銳利化參數
    SHARPEN_RADIUS = 3.0
    SHARPEN_AMOUNT = 5.527
    SHARPEN_THRESHOLD = 0.0
    
    # 降低雜訊參數
    DENOISE_H = 11.0
    DENOISE_TEMPLATE_WINDOW_SIZE = 7
    DENOISE_SEARCH_WINDOW_SIZE = 21


# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# ============================================================================
# 工具函數
# ============================================================================

def print_train_config():
    """顯示訓練與資料增強設定"""
    print("\n訓練設定:")
    print(f"  base_model: {TrainConfig.BASE_MODEL}")
    print(f"  epochs: {TrainConfig.EPOCHS}")
    print(f"  imgsz: {TrainConfig.IMGSZ}")
    print(f"  batch: {TrainConfig.BATCH}")
    print(f"  name: {TrainConfig.NAME}")
    print(f"  project: {TrainConfig.PROJECT}")
    print(f"  patience: {TrainConfig.PATIENCE}")

    print("\n資料擴增設定:")
    aug_params = TrainConfig.get_augmentation_params()
    # 只顯示啟用的擴增項目（值不為 0 的）
    if aug_params.get('degrees', 0) != 0:
        print(f"  旋轉角度 (degrees): ±{aug_params['degrees']}°")
    if aug_params.get('translate', 0) != 0:
        print(f"  平移比例 (translate): {aug_params['translate']*100:.0f}%")
    if aug_params.get('scale', 0) != 0:
        print(f"  縮放比例 (scale): {1-aug_params['scale']:.0%} ~ {1+aug_params['scale']:.0%}")
    if aug_params.get('flipud', 0) != 0:
        print(f"  上下翻轉機率 (flipud): {aug_params['flipud']*100:.0f}%")
    if aug_params.get('fliplr', 0) != 0:
        print(f"  左右翻轉機率 (fliplr): {aug_params['fliplr']*100:.0f}%")
    if aug_params.get('mosaic', 0) != 0:
        print(f"  Mosaic 擴增機率: {aug_params['mosaic']*100:.0f}%")
    if aug_params.get('mixup', 0) != 0:
        print(f"  Mixup 擴增機率: {aug_params['mixup']*100:.0f}%")

    print("\n預處理輸出路徑:")
    print(f"  資料夾: {DatasetConfig.PREPROCESSED_FOLDER}")
    print(f"  YAML: {DatasetConfig.PREPROCESSED_YAML}")

def find_image_files(directory):
    """
    查找目錄中的所有圖片檔案
    
    參數：
        directory: 目錄路徑
    
    返回：
        圖片檔案路徑列表
    """
    if not os.path.exists(directory):
        return []
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
    
    return image_files

def find_label_files(directory):
    """
    查找目錄中的所有標籤檔案
    
    參數：
        directory: 目錄路徑
    
    返回：
        標籤檔案路徑列表
    """
    if not os.path.exists(directory):
        return []
    
    return glob.glob(os.path.join(directory, '*.txt'))

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
    
    # 2. GIMP 的顏色轉灰階有 Radius 參數（300），這可能影響邊緣處理
    # 如果 radius 很大，可能需要特殊處理，但標準亮度方法已經足夠
    # 注意：GIMP 的 Radius/Samples/Iterations 是特殊算法，難以完全複製
    # 我們使用標準亮度方法作為基礎
    
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
    預處理圖片（GIMP 風格）
    依序使用：顏色轉灰階、銳利化、降低雜訊
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的圖片（灰階格式）
    """
    # 1. 顏色轉灰階（GIMP 亮度方法）
    gray = convert_to_grayscale_gimp(image)
    
    # 2. 銳利化（Unsharp Mask）
    sharpened = sharpen_image_unsharp_mask(
        gray,
        PreprocessConfig.SHARPEN_RADIUS,
        PreprocessConfig.SHARPEN_AMOUNT,
        PreprocessConfig.SHARPEN_THRESHOLD
    )
    
    # 3. 降低雜訊（非局部均值去噪）
    denoised = denoise_image(
        sharpened,
        PreprocessConfig.DENOISE_H,
        PreprocessConfig.DENOISE_TEMPLATE_WINDOW_SIZE,
        PreprocessConfig.DENOISE_SEARCH_WINDOW_SIZE
    )
    
    return denoised

def preprocess_images_in_directory(directory):
    """
    對目錄中的所有圖片進行預處理
    
    參數：
        directory: 圖片目錄路徑
    
    返回：
        處理的圖片數量
    """
    image_files = find_image_files(directory)
    count = 0
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is not None:
            preprocessed = preprocess_image(image)
            cv2.imwrite(img_path, preprocessed)
            count += 1
    
    return count

# ============================================================================
# 檔案操作函數
# ============================================================================

def copy_files(source_dir, dest_dir, file_list):
    """
    複製檔案列表到目標目錄
    
    參數：
        source_dir: 來源目錄
        dest_dir: 目標目錄
        file_list: 檔案路徑列表
    
    返回：
        複製的檔案數量
    """
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file_path, dest_path)
        count += 1
    
    return count

def setup_preprocessed_folder():
    """設置預處理資料夾結構"""
    if os.path.exists(DatasetConfig.PREPROCESSED_FOLDER):
        shutil.rmtree(DatasetConfig.PREPROCESSED_FOLDER)
    
    os.makedirs(DatasetConfig.get_preprocessed_train_dir(), exist_ok=True)
    os.makedirs(DatasetConfig.get_preprocessed_labels_train_dir(), exist_ok=True)
    
    if os.path.exists(DatasetConfig.VAL_IMAGES_DIR):
        os.makedirs(DatasetConfig.get_preprocessed_val_dir(), exist_ok=True)
        os.makedirs(DatasetConfig.get_preprocessed_labels_val_dir(), exist_ok=True)

def copy_images_to_preprocessed_folder():
    """複製已經預處理過的圖片和標籤到 DB預處理 資料夾"""
    setup_preprocessed_folder()
    
    # 複製已經預處理過的訓練圖片
    train_images = find_image_files(DatasetConfig.TRAIN_IMAGES_DIR)
    train_count = copy_files(
        DatasetConfig.TRAIN_IMAGES_DIR,
        DatasetConfig.get_preprocessed_train_dir(),
        train_images
    )
    
    # 複製訓練標籤
    train_labels = find_label_files(DatasetConfig.TRAIN_LABELS_DIR)
    label_count = copy_files(
        DatasetConfig.TRAIN_LABELS_DIR,
        DatasetConfig.get_preprocessed_labels_train_dir(),
        train_labels
    )
    
    preprocessed_train_dir = DatasetConfig.get_preprocessed_train_dir()
    print(f"  已複製 {train_count} 張已預處理的訓練圖片和 {label_count} 個標籤到 {preprocessed_train_dir}")
    
    # 複製已經預處理過的驗證圖片和標籤（如果存在）
    val_count = 0
    val_label_count = 0
    
    if os.path.exists(DatasetConfig.VAL_IMAGES_DIR):
        val_images = find_image_files(DatasetConfig.VAL_IMAGES_DIR)
        val_count = copy_files(
            DatasetConfig.VAL_IMAGES_DIR,
            DatasetConfig.get_preprocessed_val_dir(),
            val_images
        )
        
        if os.path.exists(DatasetConfig.VAL_LABELS_DIR):
            val_labels = find_label_files(DatasetConfig.VAL_LABELS_DIR)
            val_label_count = copy_files(
                DatasetConfig.VAL_LABELS_DIR,
                DatasetConfig.get_preprocessed_labels_val_dir(),
                val_labels
            )
        
        if val_count > 0:
            preprocessed_val_dir = DatasetConfig.get_preprocessed_val_dir()
            print(f"  已複製 {val_count} 張已預處理的驗證圖片和 {val_label_count} 個標籤到 {preprocessed_val_dir}")
    
    return train_count + val_count

def preprocess_images_in_folder():
    """對 DB預處理 資料夾中的圖片進行預處理"""
    print("\n正在預處理 DB預處理 資料夾中的圖片...")
    
    # 處理訓練圖片
    train_count = preprocess_images_in_directory(DatasetConfig.get_preprocessed_train_dir())
    print(f"  已預處理 {train_count} 張訓練圖片")
    
    # 處理驗證圖片（如果存在）
    val_count = 0
    preprocessed_val_dir = DatasetConfig.get_preprocessed_val_dir()
    if os.path.exists(preprocessed_val_dir):
        val_count = preprocess_images_in_directory(preprocessed_val_dir)
        if val_count > 0:
            print(f"  已預處理 {val_count} 張驗證圖片")
    
    return train_count + val_count

def create_preprocessed_yaml():
    """創建預處理資料集的 yaml 檔案"""
    print("\n正在創建預處理資料集設定檔...")
    
    yaml_content = f"""# DB預處理 訓練集設定檔

# 資料集路徑（相對於此檔案）
path: ./{DatasetConfig.PREPROCESSED_FOLDER}
train: images/train
val: images/val

# 類別名稱
names:
  0: RFID
  1: cell
  2: point

# 類別數量
nc: 3
"""
    
    with open(DatasetConfig.PREPROCESSED_YAML, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  已創建預處理資料集設定檔: {DatasetConfig.PREPROCESSED_YAML}")

# ============================================================================
# 資料檢查函數
# ============================================================================

def check_dataset():
    """檢查資料集是否存在且有效"""
    if not os.path.exists(DatasetConfig.DATASET_PATH):
        print("錯誤：找不到 DB 資料夾！")
        print("請確認 DB 資料夾已建立。")
        return False
    
    train_images_exist = (
        os.path.exists(DatasetConfig.TRAIN_IMAGES_DIR) and
        len(find_image_files(DatasetConfig.TRAIN_IMAGES_DIR)) > 0
    )
    train_labels_exist = (
        os.path.exists(DatasetConfig.TRAIN_LABELS_DIR) and
        len(find_label_files(DatasetConfig.TRAIN_LABELS_DIR)) > 0
    )
    
    if not train_images_exist:
        print("錯誤：DB/images/train/ 資料夾為空或不存在！")
        print("請放入訓練圖片。")
        return False
    
    if not train_labels_exist:
        print("錯誤：DB/labels/train/ 資料夾中沒有 TXT 標註檔！")
        return False
    
    return True

def get_statistics():
    """獲取資料集統計資訊"""
    train_images = find_image_files(DatasetConfig.TRAIN_IMAGES_DIR)
    train_labels = find_label_files(DatasetConfig.TRAIN_LABELS_DIR)
    val_images = find_image_files(DatasetConfig.VAL_IMAGES_DIR)
    
    return {
        'train_images': len(train_images),
        'train_labels': len(train_labels),
        'val_images': len(val_images)
    }

def print_statistics(stats):
    """複製資料集統計資訊"""
    print(f"\n訓練資料統計：")
    print(f"  訓練圖片：{stats['train_images']} 張")
    print(f"  標註檔：{stats['train_labels']} 個")
    
    if stats['train_images'] != stats['train_labels']:
        print(f"\n警告：圖片數量 ({stats['train_images']}) 與標註檔數量 ({stats['train_labels']}) 不一致！")
        print("請確認每張圖片都有對應的標註檔。")
    
    if stats['val_images'] > 0:
        print(f"  驗證圖片：{stats['val_images']} 張")
    else:
        print(f"  驗證圖片：無（可選）")

# ============================================================================
# 訓練函數
# ============================================================================

def load_model():
    """載入預訓練模型"""
    print("\n正在載入預訓練模型...")
    try:
        model = YOLO(TrainConfig.BASE_MODEL)
        print("模型載入成功！")
        return model
    except Exception as e:
        print(f"錯誤：無法載入模型: {e}")
        return None


def train_model(model, data_yaml):
    """
    訓練模型
    
    參數：
        model: YOLO 模型實例
        data_yaml: 資料集 YAML 檔案路徑
    
    返回：
        訓練結果物件，失敗時返回 None
    """
    print("\n開始訓練...")
    print("-" * 60)
    
    try:
        augmentation_params = TrainConfig.get_augmentation_params()
        # 過濾掉值為 0 的參數（停用的擴增項目）
        augmentation_params = {k: v for k, v in augmentation_params.items() if v != 0}
        
        results = model.train(
            data=data_yaml,
            epochs=TrainConfig.EPOCHS,
            imgsz=TrainConfig.IMGSZ,
            batch=TrainConfig.BATCH,
            name=TrainConfig.NAME,
            project=TrainConfig.PROJECT,
            patience=TrainConfig.PATIENCE,
            save=True,
            plots=True,
            **augmentation_params
        )
        
        print("\n" + "=" * 60)
        print("訓練完成！")
        print("=" * 60)
        print(f"最後一個檢查點: {results.save_dir}/weights/last.pt")
        print(f"最佳模型: {results.save_dir}/weights/best.pt")
        
        return results
        
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        return None

# ============================================================================
# 主程式
# ============================================================================

def main():
    """主程式流程"""
    print("=" * 60)
    print("YOLOv12n 細胞偵測模型訓練 - DB 訓練集")
    print("=" * 60)
    
    # 1. 檢查資料集
    if not check_dataset():
        exit(1)
    
    # 2. 統計資料
    stats = get_statistics()
    print_statistics(stats)
    
    # 3. 載入模型
    model = load_model()
    if model is None:
        exit(1)
    
    # 4. 顯示訓練設定
    print_train_config()
    
    # 5. 複製已經預處理過的圖片和標籤到 DB預處理 資料夾
    copy_images_to_preprocessed_folder()
    
    # 6. 創建預處理資料集設定檔
    create_preprocessed_yaml()
    
    # 7. 訓練模型
    results = train_model(model, DatasetConfig.PREPROCESSED_YAML)
    
    if results is None:
        exit(1)
    
    print("\n" + "=" * 60)
    print("所有處理完成！")
    print(f"預處理後的圖片已儲存在: {DatasetConfig.PREPROCESSED_FOLDER}")
    print(f"預處理資料集設定檔: {DatasetConfig.PREPROCESSED_YAML}")
    print("=" * 60)

if __name__ == "__main__":
    main()
