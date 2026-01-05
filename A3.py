
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
PREPROCESSED_FOLDER_A3 = '預處理A3'  # A3 預處理輸出路徑

# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# YOLO 模型設定
MODEL_SIZE = 'l'  # 'n'、's'、'm' 、'l'、'x'
YOLO_MODEL_PATH = f"runs/DB_cell_detection12/weights/best.pt"
IMGSZ = 640  # 圖片尺寸，與訓練時一致

# 預處理設定
USE_PREPROCESSING = True  # True = 使用預處理（與訓練時一致），False = 使用原始圖片

# 過濾設定
FILTER_CONFIG = {
    # 1. 類別信心度過濾（程式層，類別特定）
    'class_confidence_threshold': {
        'RFID': 0.6,   # RFID 類別信心度閾值
        'cell': 0.2,   # cell 類別信心度閾值
        'point': 0.18   # point 類別信心度閾值
    },
    
    # 2. YOLO信心度閾值（模型層，最低值）
    'yolo_conf_threshold': 0.001,  # YOLO 模型層使用最低的信心度值
    
    # 3. 重疊框過濾（參數越低越嚴格)
    'yolo_iou_threshold': 0.03,  # YOLO 內建 NMS IoU 閾值，過濾重疊的檢測框
    
    # 4. 同類別NMS過濾（參數越低越嚴格)
    'same_class_nms_threshold': 0.05,  # 同類別 NMS IoU 閾值
    'rfid_same_class_nms_threshold': 0.05,  # RFID 專用同類別 NMS 閾值
    
    # 5. 跨類別NMS過濾（參數越低越嚴格)
    'cross_class_iou_threshold': 0.02,  # 跨類別 NMS IoU 閾值
    
    # 6. 面積過濾
    'min_area_by_class': {
        'RFID': 0.0001,   # RFID 最小面積比例（相對於圖片大小，降低：從 0.001 降到 0.0001）
        'cell': 0.00001,   # cell 最小面積比例（降低：從 0.001 降到 0.0001）
        'point': 0.0001   # point 最小面積比例（降低：從 0.001 降到 0.0001）
    },
    'max_area_ratio': 0.99,  # 最大面積比例（相對於圖片大小）
    
    # 7. RFID框擴展設定（確保邊邊角角都被框到）
    'rfid_expand': {
        'enabled': True,           # 是否啟用RFID框擴展
        'expand_ratio': 0.06,       # 擴展比例（6%，即每邊擴展3%，稍微增加以確保邊角）
        'min_expand_pixels': 4,     # 最小擴展像素數
        'max_expand_pixels': 18,    # 最大擴展像素數
    },
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

# 預處理參數（與 train_DB.py 保持一致）
PREPROCESS_PARAMS = {
    # 顏色轉灰階參數
    'grayscale_radius': 300,        # GIMP Radius = 300
    'grayscale_samples': 4,         # GIMP Samples = 4
    'grayscale_iterations': 10,     # GIMP Iterations = 10
    'grayscale_enhance_shadows': False,  # GIMP Enhance Shadows = 未勾選
    
    # 銳利化參數
    'sharpen_radius': 3.0,          # 銳化半徑
    'sharpen_amount': 1.5,           # 銳化強度
    'sharpen_threshold': 0.0,       # 銳化閾值
    
    # 降低雜訊參數
    'denoise_h': 11.0,              # 過濾強度
    'denoise_templateWindowSize': 7, # 模板窗口大小
    'denoise_searchWindowSize': 21,  # 搜索窗口大小
    
    # 對比度增強參數（用於加深黑點）
    'enhance_contrast': True,        # 是否啟用對比度增強
    'contrast_alpha': 1.2,           # 對比度係數（1.0 = 無變化，>1.0 = 增強對比度，<1.0 = 降低對比度）
    'contrast_beta': 0,              # 亮度調整（0 = 無變化）
}


# 預處理函數
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
    # 使用 GIMP 亮度公式轉換為灰階
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

def preprocess_image(image):
    """
    預處理圖片（與 train_DB.py 完全一致）
    依序使用：顏色轉灰階、銳利化、降低雜訊、對比度增強
    
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
        PREPROCESS_PARAMS['denoise_h'],
        PREPROCESS_PARAMS['denoise_templateWindowSize'],
        PREPROCESS_PARAMS['denoise_searchWindowSize']
    )
    
    # 4. 對比度增強（加深黑點）
    if PREPROCESS_PARAMS.get('enhance_contrast', False):
        final = enhance_contrast(
            denoised,
            PREPROCESS_PARAMS['contrast_alpha'],  # 使用參數定義的值（1.2）
            PREPROCESS_PARAMS.get('contrast_beta', 0)
        )
    else:
        final = denoised
    
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
    """
    使用 YOLO 模型偵測物體
    
    參數：
        model: YOLO 模型
        detection_image: 輸入圖片
        conf_threshold: YOLO信心度閾值（模型層）
        iou_threshold: YOLO內建NMS IoU閾值（重疊框過濾）
    
    返回：
        detections: 偵測結果
        num_boxes: 偵測框數量
    """
    results = model(
        detection_image, 
        conf=conf_threshold,  # YOLO信心度閾值
        iou=iou_threshold,    # YOLO內建NMS IoU閾值（重疊框過濾）
        imgsz=IMGSZ,
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
        all_confidences = []  # 所有信心度
        
        for box in detections.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            confidence = float(box.conf[0].cpu().numpy())
            class_id_counts[cls_id] = class_id_counts.get(cls_id, 0) + 1
            class_id_confidences.setdefault(cls_id, []).append(confidence)
            all_confidences.append(confidence)
        
        # 顯示整體信心度統計
        if all_confidences:
            print(f"  📊 整體信心度統計:")
            print(f"    總偵測數: {num_boxes} 個")
            print(f"    信心度範圍: {min(all_confidences):.4f} ~ {max(all_confidences):.4f}")
            print(f"    平均信心度: {sum(all_confidences) / len(all_confidences):.4f}")
            print(f"    中位數信心度: {sorted(all_confidences)[len(all_confidences)//2]:.4f}")
            
            # 信心度分佈
            low_conf = [c for c in all_confidences if c < 0.1]
            mid_conf = [c for c in all_confidences if 0.1 <= c < 0.5]
            high_conf = [c for c in all_confidences if c >= 0.5]
            print(f"    信心度分佈: 低(<0.1): {len(low_conf)} 個, 中(0.1-0.5): {len(mid_conf)} 個, 高(>=0.5): {len(high_conf)} 個")
        
        # 顯示各類別統計
        print(f"  📋 各類別詳細統計:")
        for cls_id, count in class_id_counts.items():
            class_name = CLASS_MAPPING.get(cls_id, f"class_{cls_id}")
            confs = class_id_confidences.get(cls_id, [])
            if confs:
                avg_conf = sum(confs) / len(confs)
                min_conf = min(confs)
                max_conf = max(confs)
                threshold = FILTER_CONFIG['class_confidence_threshold'].get(class_name, 0.01)
                below_threshold = len([c for c in confs if c < threshold])
                
                print(f"    {class_name} (class_id={cls_id}):")
                print(f"      數量: {count} 個")
                print(f"      信心度範圍: {min_conf:.4f} ~ {max_conf:.4f}")
                print(f"      平均信心度: {avg_conf:.4f}")
                print(f"      設定閾值: {threshold:.4f}")
                print(f"      低於閾值: {below_threshold} 個 (會被過濾)")
                
                # 顯示前5個最低信心度
                sorted_confs = sorted(confs)
                if len(sorted_confs) <= 5:
                    conf_strs = [f"{c:.4f}" for c in sorted_confs]
                    print(f"      所有信心度: {conf_strs}")
                else:
                    conf_strs = [f"{c:.4f}" for c in sorted_confs[:5]]
                    print(f"      最低5個信心度: {conf_strs}")
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

def expand_rfid_box(rfid_obj, imgwidth, imgheight):
    """
    擴展RFID框，確保邊邊角角都被框到
    
    參數：
        rfid_obj: RFID物件（字典，包含 x1, y1, x2, y2, w, h, area, cx, cy）
        imgwidth: 圖片寬度
        imgheight: 圖片高度
    
    返回：
        擴展後的RFID物件（更新座標、尺寸、面積、中心點）
    """
    if not FILTER_CONFIG['rfid_expand']['enabled']:
        return rfid_obj
    
    expand_config = FILTER_CONFIG['rfid_expand']
    expand_ratio = expand_config['expand_ratio']
    min_expand = expand_config['min_expand_pixels']
    max_expand = expand_config['max_expand_pixels']
    
    # 計算當前框的寬度和高度
    w = rfid_obj['w']
    h = rfid_obj['h']
    
    # 計算擴展量（按比例，但限制在最小和最大像素數之間）
    expand_w = max(min_expand, min(max_expand, int(w * expand_ratio / 2)))
    expand_h = max(min_expand, min(max_expand, int(h * expand_ratio / 2)))
    
    # 擴展座標（每邊擴展）
    new_x1 = max(0, rfid_obj['x1'] - expand_w)
    new_y1 = max(0, rfid_obj['y1'] - expand_h)
    new_x2 = min(imgwidth, rfid_obj['x2'] + expand_w)
    new_y2 = min(imgheight, rfid_obj['y2'] + expand_h)
    
    # 更新物件資訊
    rfid_obj['x1'] = new_x1
    rfid_obj['y1'] = new_y1
    rfid_obj['x2'] = new_x2
    rfid_obj['y2'] = new_y2
    rfid_obj['w'] = new_x2 - new_x1
    rfid_obj['h'] = new_y2 - new_y1
    rfid_obj['area'] = rfid_obj['w'] * rfid_obj['h']
    rfid_obj['cx'] = new_x1 + rfid_obj['w'] // 2
    rfid_obj['cy'] = new_y1 + rfid_obj['h'] // 2
    
    return rfid_obj

def apply_same_class_nms(detected_objects, iou_threshold):
    """
    同類別NMS：過濾同類別內重疊的框，保留信心度最高的
    
    參數：
        detected_objects: 偵測物件列表
        iou_threshold: IoU 閾值
    
    返回：
        過濾後的物件列表
    """
    if not detected_objects:
        return []
    
    # 按類別分組
    objects_by_class = {}
    for obj in detected_objects:
        class_name = obj['class']
        if class_name not in objects_by_class:
            objects_by_class[class_name] = []
        objects_by_class[class_name].append(obj)
    
    result = []
    # 對每個類別分別進行 NMS
    for class_name, objects in objects_by_class.items():
        if len(objects) == 1:
            result.append(objects[0])
            continue
        
        # 按信心度降序排序
        objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while objects:
            current = objects.pop(0)
            keep.append(current)
            
            # 移除與 current 重疊度高的框
            new_objects = []
            for obj in objects:
                iou = calculate_iou(current, obj)
                if iou < iou_threshold:
                    new_objects.append(obj)
            objects = new_objects
        
        result.extend(keep)
    
    return result

def apply_cross_class_nms(detected_objects, iou_threshold):
    """
    跨類別NMS：過濾不同類別之間重疊的框，根據信心度決定保留哪個
    特殊處理：RFID 優先保留（因為每張圖必須有一個 RFID）
    
    參數：
        detected_objects: 偵測物件列表
        iou_threshold: IoU 閾值
    
    返回：
        過濾後的物件列表
    """
    if not detected_objects:
        return []
    
    # 分離 RFID 和其他物件
    rfid_objects = [obj for obj in detected_objects if obj['class'] == 'RFID']
    other_objects = [obj for obj in detected_objects if obj['class'] != 'RFID']
    
    # 先對其他物件進行跨類別 NMS（不包括 RFID）
    if not other_objects:
        return rfid_objects
    
    # 按信心度降序排序（不包括 RFID）
    sorted_objects = sorted(other_objects, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for current in sorted_objects:
        # 檢查 current 是否與 keep 中的框重疊（不同類別）
        should_keep = True
        new_keep = []
        
        for kept in keep:
            if kept['class'] == current['class']:
                # 同類別不在此處理（已在同類別NMS中處理）
                new_keep.append(kept)
            else:
                # 不同類別，檢查 IoU
                iou = calculate_iou(current, kept)
                if iou >= iou_threshold:
                    # 重疊度高，保留信心度更高的
                    # 因為已按降序排序，current 的信心度 >= kept 的信心度
                    # 所以保留 current，移除 kept（不加入 new_keep）
                    should_keep = True
                else:
                    # 重疊度低，兩個都保留
                    new_keep.append(kept)
        
        if should_keep:
            keep = new_keep + [current]
        else:
            keep = new_keep
    
    # 最後處理 RFID：如果 RFID 與其他物件重疊，優先保留 RFID
    final_keep = keep.copy()
    for rfid_obj in rfid_objects:
        should_add_rfid = True
        new_final_keep = []
        
        for kept_obj in final_keep:
            iou = calculate_iou(rfid_obj, kept_obj)
            if iou >= iou_threshold:
                # RFID 與其他物件重疊，優先保留 RFID，移除其他物件
                # 不將 kept_obj 加入 new_final_keep
                pass
            else:
                # 不重疊，保留其他物件
                new_final_keep.append(kept_obj)
        
        if should_add_rfid:
            final_keep = new_final_keep + [rfid_obj]
        else:
            final_keep = new_final_keep
    
    return final_keep

def filter_detections(detections, imgwidth, imgheight):
    """
    過濾偵測結果
    應用：類別信心度過濾、面積過濾
    
    參數：
        detections: YOLO 偵測結果
        imgwidth: 圖片寬度
        imgheight: 圖片高度
    
    返回：
        detected_objects: 過濾後的物件列表
        class_counts: 各類別計數
    """
    detected_objects = []
    class_counts = {'RFID': 0, 'cell': 0, 'point': 0}
    max_area = imgwidth * imgheight * FILTER_CONFIG['max_area_ratio']
    
    # YOLO 格式
    if detections.boxes is None or len(detections.boxes) == 0:
        return detected_objects, class_counts
    
    filtered_by_confidence = 0
    filtered_by_area = 0
    filtered_by_coords = 0
    filtered_confidence_details = []  # 記錄被信心度過濾的詳細資訊
    
    for box in detections.boxes:
        obj = parse_detection_box(box, detections, imgwidth, imgheight)
        class_name = obj['class']
        
        # 1. 類別信心度過濾
        class_conf_threshold = FILTER_CONFIG['class_confidence_threshold'].get(class_name, 0.01)
        if obj['confidence'] < class_conf_threshold:
            filtered_by_confidence += 1
            filtered_confidence_details.append({
                'class': class_name,
                'confidence': obj['confidence'],
                'threshold': class_conf_threshold,
                'diff': class_conf_threshold - obj['confidence']
            })
            continue
        
        # 2. 面積過濾
        min_area = imgwidth * imgheight * FILTER_CONFIG['min_area_by_class'].get(class_name, 0.001)
        if not (min_area <= obj['area'] <= max_area):
            filtered_by_area += 1
            continue
        
        # 3. 座標驗證（允許稍微超出範圍，因為 YOLO resize 可能導致邊界問題）
        # 將座標限制在圖片範圍內
        obj['x1'] = max(0, min(obj['x1'], imgwidth - 1))
        obj['y1'] = max(0, min(obj['y1'], imgheight - 1))
        obj['x2'] = max(obj['x1'] + 1, min(obj['x2'], imgwidth))
        obj['y2'] = max(obj['y1'] + 1, min(obj['y2'], imgheight))
        obj['w'] = obj['x2'] - obj['x1']
        obj['h'] = obj['y2'] - obj['y1']
        obj['area'] = obj['w'] * obj['h']  # 重新計算面積
        obj['cx'] = obj['x1'] + obj['w'] // 2  # 重新計算中心點
        obj['cy'] = obj['y1'] + obj['h'] // 2
        
        # 驗證基本有效性
        if not (obj['x1'] >= 0 and obj['y1'] >= 0 and 
                obj['x2'] > obj['x1'] and obj['y2'] > obj['y1'] and
                obj['w'] > 0 and obj['h'] > 0):
            filtered_by_coords += 1
            continue
        
        # 通過所有過濾條件
        detected_objects.append(obj)
        if class_name in class_counts:
            class_counts[class_name] += 1
    
    # 輸出過濾統計（僅在有多個檢測框時顯示）
    if len(detections.boxes) > 0:
        if filtered_by_confidence > 0:
            print(f"  ⚠️  信心度過濾: {filtered_by_confidence} 個")
            # 顯示被過濾的信心度詳細資訊
            if len(filtered_confidence_details) > 0:
                print(f"     被過濾的信心度詳細資訊:")
                # 按類別分組
                by_class = {}
                for detail in filtered_confidence_details:
                    class_name = detail['class']
                    if class_name not in by_class:
                        by_class[class_name] = []
                    by_class[class_name].append(detail)
                
                for class_name, details in by_class.items():
                    confs = [d['confidence'] for d in details]
                    threshold = details[0]['threshold']
                    print(f"       {class_name}: {len(details)} 個 (閾值: {threshold:.4f})")
                    print(f"         信心度範圍: {min(confs):.4f} ~ {max(confs):.4f}")
                    print(f"         平均信心度: {sum(confs)/len(confs):.4f}")
                    print(f"         與閾值差距: {threshold - max(confs):.4f} ~ {threshold - min(confs):.4f}")
                    # 顯示最低的幾個
                    sorted_details = sorted(details, key=lambda x: x['confidence'])
                    if len(sorted_details) <= 3:
                        conf_strs = [f"{d['confidence']:.4f}" for d in sorted_details]
                        print(f"         所有被過濾的信心度: {conf_strs}")
                    else:
                        conf_strs = [f"{d['confidence']:.4f}" for d in sorted_details[:3]]
                        print(f"         最低3個信心度: {conf_strs}")
        if filtered_by_area > 0:
            print(f"  過濾統計 - 面積過濾: {filtered_by_area} 個")
        if filtered_by_coords > 0:
            print(f"  過濾統計 - 座標驗證: {filtered_by_coords} 個")
    
    return detected_objects, class_counts

# ============================================================================
# 標記和儲存函數
# ============================================================================

def draw_bounding_boxes(image, detected_objects, imgwidth, imgheight):
    """在圖片上繪製標記框"""
    marked_count = 0
    marked_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    invalid_coords_count = 0  # 記錄座標無效的物件數量
    
    for obj in detected_objects:
        # 完整的座標驗證
        x1, y1, x2, y2 = obj.get("x1", 0), obj.get("y1", 0), obj.get("x2", 0), obj.get("y2", 0)
        
        # 驗證座標有效性
        if not (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and 
                x1 < imgwidth and y1 < imgheight and 
                x2 <= imgwidth and y2 <= imgheight):
            invalid_coords_count += 1
            print(f"  ⚠️  警告：物件 {obj.get('class', 'unknown')} 座標無效，跳過繪製")
            print(f"     座標: x1={x1}, y1={y1}, x2={x2}, y2={y2}, 圖片尺寸: {imgwidth}x{imgheight}")
            continue
        
        class_name = obj['class']
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # 繪製矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 添加標籤
        label = f"{class_name} {obj['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        
        # 標籤背景框（確保不會超出圖片上邊界）
        if y1 >= text_height + 5:
            # 標籤在框上方（正常情況）
            label_y1 = y1 - text_height - 5
            label_y2 = y1
            text_y = y1 - 5
        else:
            # 框太靠近上邊界，標籤放在框下方
            label_y1 = y2
            label_y2 = min(imgheight, y2 + text_height + 5)
            text_y = min(imgheight - 5, y2 + text_height)
        
        # 確保標籤背景框不會超出圖片範圍
        label_y1 = max(0, label_y1)
        label_y2 = min(imgheight, label_y2)
        text_x = max(0, min(x1, imgwidth - text_width))
        
        cv2.rectangle(
            image,
            (text_x, label_y1),
            (text_x + text_width, label_y2),
            color, -1
        )
        
        # 黑色文字
        cv2.putText(
            image, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )
        
        marked_count += 1
        if class_name in marked_by_class:
            marked_by_class[class_name] += 1
    
    if invalid_coords_count > 0:
        print(f"  ⚠️  總共有 {invalid_coords_count} 個物件因座標無效而跳過繪製")
    
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
        print(f"  正在預處理圖片（轉灰階、銳利化、降低雜訊）...")
        detection_image = preprocess_image(image_src)
        
        # 儲存預處理後的圖片到「預處理A3」資料夾
        preprocessed_path_a3 = os.path.join(PREPROCESSED_FOLDER_A3, os.path.basename(image_path))
        cv2.imwrite(preprocessed_path_a3, detection_image)
    else:
        detection_image = image_src
    
    # 1. 使用 YOLO 模型進行偵測（YOLO信心度閾值、重疊框過濾）
    print(f"  🔍 YOLO 偵測設定:")
    print(f"    模型層信心度閾值: {FILTER_CONFIG['yolo_conf_threshold']:.4f}")
    print(f"    程式層類別信心度閾值:")
    for class_name, threshold in FILTER_CONFIG['class_confidence_threshold'].items():
        print(f"      {class_name}: {threshold:.4f}")
    
    detections, num_boxes = detect_objects_yolo(
        model, 
        detection_image, 
        FILTER_CONFIG['yolo_conf_threshold'],      # YOLO信心度閾值
        FILTER_CONFIG['yolo_iou_threshold']        # YOLO內建NMS（重疊框過濾）
    )
    print_detection_info(detections, num_boxes)
    
    # 診斷：檢查原始檢測結果中是否有 RFID（過濾前）
    if detections.boxes is not None and len(detections.boxes) > 0:
        rfid_in_raw = 0
        rfid_confidences = []
        for box in detections.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            confidence = float(box.conf[0].cpu().numpy())
            if cls_id == 0:  # RFID
                rfid_in_raw += 1
                rfid_confidences.append(confidence)
        
        if rfid_in_raw > 0:
            print(f"  🔍 原始檢測結果中有 {rfid_in_raw} 個 RFID（過濾前）")
            print(f"     RFID 信心度範圍: {min(rfid_confidences):.4f} ~ {max(rfid_confidences):.4f}")
        else:
            print(f"  ⚠️  原始檢測結果中沒有 RFID（模型未輸出 RFID 檢測框）")
    
    # 2. 過濾偵測結果（類別信心度過濾、面積過濾）
    detected_objects, class_counts = filter_detections(detections, imgwidth, imgheight)
    print(f"  類別信心度過濾 + 面積過濾後 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']}")
    
    # 3. 同類別NMS過濾
    before_same_nms = len(detected_objects)
    before_same_nms_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        if obj['class'] in before_same_nms_by_class:
            before_same_nms_by_class[obj['class']] += 1
    
    # 分開處理 RFID 和其他類別
    rfid_objects = [obj for obj in detected_objects if obj['class'] == 'RFID']
    other_objects = [obj for obj in detected_objects if obj['class'] != 'RFID']
    
    # RFID 使用專用 NMS 閾值
    rfid_after_nms = apply_same_class_nms(rfid_objects, FILTER_CONFIG['rfid_same_class_nms_threshold'])
    # 其他類別使用標準 NMS 閾值
    other_after_nms = apply_same_class_nms(other_objects, FILTER_CONFIG['same_class_nms_threshold'])
    
    detected_objects = rfid_after_nms + other_after_nms
    after_same_nms = len(detected_objects)
    after_same_nms_by_class = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        if obj['class'] in after_same_nms_by_class:
            after_same_nms_by_class[obj['class']] += 1
    
    if before_same_nms != after_same_nms:
        print(f"  同類別NMS過濾掉 {before_same_nms - after_same_nms} 個重疊框")
        for class_name in ['RFID', 'cell', 'point']:
            removed = before_same_nms_by_class[class_name] - after_same_nms_by_class[class_name]
            if removed > 0:
                print(f"    - {class_name}: {before_same_nms_by_class[class_name]} -> {after_same_nms_by_class[class_name]} (移除 {removed} 個)")
    
    # 4. 跨類別NMS過濾
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
        print(f"  跨類別NMS過濾掉 {before_cross_nms_count - after_cross_nms_count} 個框 (IoU threshold: {FILTER_CONFIG['cross_class_iou_threshold']})")
        for class_name in ['RFID', 'cell', 'point']:
            removed = before_cross_nms_by_class[class_name] - after_cross_nms_by_class[class_name]
            if removed > 0:
                print(f"    - {class_name}: {before_cross_nms_by_class[class_name]} -> {after_cross_nms_by_class[class_name]} (移除 {removed} 個)")
    
    # 5. RFID過濾：每張圖只保留一個RFID標記（信心度最高的）
    rfid_objects = [obj for obj in detected_objects if obj['class'] == 'RFID']
    other_objects = [obj for obj in detected_objects if obj['class'] != 'RFID']
    
    if len(rfid_objects) > 1:
        # 按信心度降序排序，只保留第一個（信心度最高的）
        rfid_objects.sort(key=lambda x: x['confidence'], reverse=True)
        removed_rfid_count = len(rfid_objects) - 1
        print(f"  RFID過濾：保留信心度最高的RFID標記，移除 {removed_rfid_count} 個其他RFID標記")
        print(f"    保留的RFID信心度: {rfid_objects[0]['confidence']:.4f}")
        if removed_rfid_count > 0:
            removed_confidences = [obj['confidence'] for obj in rfid_objects[1:]]
            print(f"    移除的RFID信心度: {[f'{c:.4f}' for c in removed_confidences]}")
        rfid_objects = [rfid_objects[0]]  # 只保留第一個
    elif len(rfid_objects) == 1:
        print(f"  RFID過濾：已有一個RFID標記（信心度: {rfid_objects[0]['confidence']:.4f}）")
    else:
        print(f"  RFID過濾：未偵測到RFID標記")
    
    # 5.5. RFID框擴展（確保邊邊角角都被框到）
    if len(rfid_objects) > 0 and FILTER_CONFIG['rfid_expand']['enabled']:
        original_box = rfid_objects[0].copy()
        rfid_objects[0] = expand_rfid_box(rfid_objects[0], imgwidth, imgheight)
        expand_info = FILTER_CONFIG['rfid_expand']
        print(f"  RFID框擴展：")
        print(f"    原始尺寸: {original_box['w']}x{original_box['h']}")
        print(f"    擴展後尺寸: {rfid_objects[0]['w']}x{rfid_objects[0]['h']}")
        print(f"    擴展比例: {expand_info['expand_ratio']*100:.1f}% (每邊 {expand_info['expand_ratio']*50:.1f}%)")
        print(f"    擴展像素: 寬度 +{rfid_objects[0]['w'] - original_box['w']}, 高度 +{rfid_objects[0]['h'] - original_box['h']}")
    
    # 6. 刪除RFID框內信心度低於0.05的cell和point
    if len(rfid_objects) > 0:
        rfid_box = rfid_objects[0]  # 使用唯一的RFID框（已擴展）
        cell_objects = [obj for obj in other_objects if obj['class'] == 'cell']
        point_objects = [obj for obj in other_objects if obj['class'] == 'point']
        
        filtered_cells = []
        removed_cells = []
        for cell_obj in cell_objects:
            # 檢查cell是否在RFID框內
            if is_box_inside(cell_obj, rfid_box):
                # 在RFID框內，檢查信心度
                if cell_obj['confidence'] < 0.6:
                    removed_cells.append(cell_obj)
                else:
                    filtered_cells.append(cell_obj)
            else:
                # 不在RFID框內，保留
                filtered_cells.append(cell_obj)
        
        if len(removed_cells) > 0:
            print(f"  RFID框內cell過濾：移除 {len(removed_cells)} 個信心度低於0.05的cell")
            removed_confidences = [obj['confidence'] for obj in removed_cells]
            print(f"    移除的cell信心度: {[f'{c:.4f}' for c in removed_confidences]}")
        else:
            print(f"  RFID框內cell過濾：無需移除的cell")
        
        # 過濾RFID框內的point：只保留信心度大於0.05的point
        filtered_points = []
        removed_points = []
        for point_obj in point_objects:
            # 檢查point是否在RFID框內
            if is_box_inside(point_obj, rfid_box):
                # 在RFID框內，檢查信心度
                if point_obj['confidence'] <= 0.22:
                    removed_points.append(point_obj)
                else:
                    filtered_points.append(point_obj)
            else:
                # 不在RFID框內，保留
                filtered_points.append(point_obj)
        
        if len(removed_points) > 0:
            print(f"  RFID框內point過濾：移除 {len(removed_points)} 個信心度低於等於0.05的point")
            removed_confidences = [obj['confidence'] for obj in removed_points]
            print(f"    移除的point信心度: {[f'{c:.4f}' for c in removed_confidences]}")
        else:
            print(f"  RFID框內point過濾：無需移除的point")
        
        # 重新組合物件列表
        detected_objects = rfid_objects + filtered_cells + filtered_points
    else:
        # 沒有RFID框，保留所有其他物件
        detected_objects = other_objects
    
    # 重新計算類別計數
    class_counts = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        class_name = obj['class']
        if class_name in class_counts:
            class_counts[class_name] += 1
    
    print(f"  最終結果 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']} (原始偵測: {num_boxes} 個)")
    
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
    print("過濾功能：")
    print("  1. 類別信心度過濾")
    print("  2. YOLO信心度閾值")
    print("  3. 重疊框過濾（YOLO內建NMS）")
    print("  4. 同類別NMS過濾")
    print("  5. 跨類別NMS過濾")
    print("  6. 面積過濾")
    print("預處理：轉灰階、銳利化、降低雜訊（與 train_DB.py 一致）")
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

