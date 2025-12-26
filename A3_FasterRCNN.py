#!/usr/bin/env python3
"""
Faster R-CNN ResNet-50 FPN V2 細胞偵測程式 - 圖片標記
"""

import os
import glob
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.transforms import functional as F

# ============================================================================
# 配置參數
# ============================================================================

# 資料夾路徑
START_FOLDER = 'start'           # 要標記的圖片資料夾
FINISH_FOLDER = 'finish'         # 處理完成的圖片資料夾
PREPROCESSED_FOLDER_A3 = '預處裡A3_FasterRCNN'  # A3 預處理輸出路徑

# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

# 模型路徑
MODEL_PATH = "runs/FasterRCNN_DB_cell_detection12/best.pth"
NUM_CLASSES = 4  # 背景(0) + RFID(1) + cell(2) + point(3)

# 預處理設定
USE_PREPROCESSING = True  # True = 使用預處理，False = 使用原始圖片

# 過濾設定
FILTER_CONFIG = {
    'confidence_by_class': {
        'RFID':  0.1,
        'cell':  0.95,
        'point': 0.17
    },
    'nms_threshold': 0.5,  # NMS IoU 閾值
    
    # 面積過濾
    'min_area_by_class': {
        'RFID':  0.001,
        'cell':  0.0001,
        'point': 0.0001
    },
    'max_area_ratio': 0.99,
}

# 類別顏色設定 (BGR 格式)
CLASS_COLORS = {
    'RFID': (255, 255, 170),
    'cell': (170, 255, 255),
    'point': (255, 170, 255)
}

# 類別映射（Faster R-CNN: 0=背景, 1=RFID, 2=cell, 3=point）
CLASS_MAPPING = {
    1: 'RFID',
    2: 'cell',
    3: 'point'
}

# 預處理參數（與 train_FasterRCNN.py 保持一致）
PREPROCESS_PARAMS = {
    'sharpen_radius': 2.5,
    'sharpen_amount': 3.5,
    'sharpen_threshold': 0.0,
    'denoise_h': 8.0,
    'denoise_templateWindowSize': 7,
    'denoise_searchWindowSize': 21,
    'use_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'darken_black_spots': True,
    'gamma_correction': 0.7,
}

# ============================================================================
# 預處理函數（與 train_FasterRCNN.py 一致）
# ============================================================================

def convert_to_grayscale_gimp(image):
    """使用 GIMP 亮度方法轉換為灰階"""
    b, g, r = cv2.split(image)
    gray = (0.114 * b.astype(np.float32) + 
            0.587 * g.astype(np.float32) + 
            0.299 * r.astype(np.float32)).astype(np.uint8)
    return gray

def sharpen_image_unsharp_mask(image, radius, amount, threshold=0.0):
    """使用 Unsharp Mask 方法銳化圖片"""
    img_float = image.astype(np.float32)
    kernel_size = int(6 * radius + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), radius)
    diff = img_float - blurred
    sharpened = img_float + diff * amount
    if threshold > 0:
        diff_abs = np.abs(diff)
        mask = diff_abs > threshold
        sharpened = np.where(mask, sharpened, img_float)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def denoise_image(image, strength, template_window_size, search_window_size):
    """使用非局部均值去噪降低雜訊"""
    return cv2.fastNlMeansDenoising(
        image,
        h=float(strength),
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )

def darken_black_spots(image, gamma):
    """使用伽馬校正加深黑點"""
    img_normalized = image.astype(np.float32) / 255.0
    img_gamma = np.power(img_normalized, 1.0 / gamma)
    img_gamma = (img_gamma * 255.0).astype(np.uint8)
    return img_gamma

def preprocess_image(image):
    """預處理圖片"""
    gray = convert_to_grayscale_gimp(image)
    
    if PREPROCESS_PARAMS.get('use_clahe', False):
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESS_PARAMS.get('clahe_clip_limit', 2.0),
            tileGridSize=PREPROCESS_PARAMS.get('clahe_tile_grid_size', (8, 8))
        )
        gray = clahe.apply(gray)
    
    sharpened = sharpen_image_unsharp_mask(
        gray,
        PREPROCESS_PARAMS['sharpen_radius'],
        PREPROCESS_PARAMS['sharpen_amount'],
        PREPROCESS_PARAMS['sharpen_threshold']
    )
    
    denoised = denoise_image(
        sharpened,
        PREPROCESS_PARAMS['denoise_h'],
        PREPROCESS_PARAMS['denoise_templateWindowSize'],
        PREPROCESS_PARAMS['denoise_searchWindowSize']
    )
    
    if PREPROCESS_PARAMS.get('darken_black_spots', False):
        denoised = darken_black_spots(
            denoised,
            PREPROCESS_PARAMS.get('gamma_correction', 0.7)
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
# 過濾和處理函數
# ============================================================================

def is_box_inside(box_inner, box_outer):
    """檢查內層框是否完全在外層框內部"""
    return (box_inner['x1'] >= box_outer['x1'] and 
            box_inner['y1'] >= box_outer['y1'] and 
            box_inner['x2'] <= box_outer['x2'] and 
            box_inner['y2'] <= box_outer['y2'])

def filter_detections(detections, imgwidth, imgheight):
    """過濾偵測結果"""
    detected_objects = []
    class_counts = {'RFID': 0, 'cell': 0, 'point': 0}
    total_area = imgwidth * imgheight
    max_area = total_area * FILTER_CONFIG['max_area_ratio']
    
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if label not in CLASS_MAPPING:
            continue
        
        class_name = CLASS_MAPPING[label]
        confidence_threshold = FILTER_CONFIG['confidence_by_class'].get(class_name, 0.1)
        
        if score < confidence_threshold:
            continue
        
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        min_area_ratio = FILTER_CONFIG['min_area_by_class'].get(class_name, 0.001)
        min_area = total_area * min_area_ratio
        
        if min_area <= area <= max_area and w > 0 and h > 0:
            obj = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'w': w, 'h': h,
                'cx': x1 + w // 2, 'cy': y1 + h // 2,
                'area': area, 'confidence': float(score),
                'class': class_name, 'class_id': int(label) - 1  # 轉回 YOLO 格式 (0,1,2)
            }
            detected_objects.append(obj)
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
            
            cv2.rectangle(image, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), color, 2)
            
            label = f"{class_name} {obj['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            
            cv2.rectangle(
                image,
                (obj["x1"], obj["y1"] - text_height - 5),
                (obj["x1"] + text_width, obj["y1"]),
                color, -1
            )
            
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
    
    output_image_path = os.path.join(FINISH_FOLDER, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)
    
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

def process_image(image_path, model, device):
    """處理單張圖片並標記細胞"""
    print(f"\n處理: {os.path.basename(image_path)}")
    
    image_src = cv2.imread(image_path)
    if image_src is None:
        print(f"  錯誤：無法讀取圖片 {image_path}")
        return False
    
    imgheight = image_src.shape[0]
    imgwidth = image_src.shape[1]
    
    # 預處理
    if USE_PREPROCESSING:
        print(f"  正在預處理圖片...")
        image_preprocessed = preprocess_image(image_src)
        preprocessed_path_a3 = os.path.join(PREPROCESSED_FOLDER_A3, os.path.basename(image_path))
        cv2.imwrite(preprocessed_path_a3, image_preprocessed)
        image_for_model = cv2.cvtColor(image_preprocessed, cv2.COLOR_GRAY2BGR)
    else:
        image_for_model = image_src
    
    # 轉換為 RGB 和 tensor
    image_rgb = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        detections = model([image_tensor])[0]
    
    # 過濾偵測結果
    detected_objects, class_counts = filter_detections(detections, imgwidth, imgheight)
    print(f"  偵測結果 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']}")
    
    # RFID 過濾：只保留一個，並過濾內部的 cell 和 point
    rfid_objects = [obj for obj in detected_objects if obj['class'] == 'RFID']
    other_objects = [obj for obj in detected_objects if obj['class'] != 'RFID']
    
    if len(rfid_objects) > 1:
        rfid_objects.sort(key=lambda x: x['confidence'], reverse=True)
        kept_rfid = rfid_objects[0]
        removed_count = len(rfid_objects) - 1
        print(f"  RFID 過濾：從 {len(rfid_objects)} 個中只保留信心值最高的一個 (conf={kept_rfid['confidence']:.4f})，移除 {removed_count} 個")
        rfid_objects = [kept_rfid]
    
    if len(rfid_objects) == 1:
        rfid = rfid_objects[0]
        filtered_objects = []
        removed_inside_rfid = {'cell': 0, 'point': 0}
        
        for obj in other_objects:
            if obj['class'] in ['cell', 'point']:
                if is_box_inside(obj, rfid):
                    removed_inside_rfid[obj['class']] += 1
                    print(f"    過濾 RFID 內部的 {obj['class']} (conf={obj['confidence']:.4f})")
                else:
                    filtered_objects.append(obj)
            else:
                filtered_objects.append(obj)
        
        if removed_inside_rfid['cell'] > 0 or removed_inside_rfid['point'] > 0:
            print(f"  RFID 內部過濾：移除 {removed_inside_rfid['cell']} 個 cell 和 {removed_inside_rfid['point']} 個 point")
        
        detected_objects = rfid_objects + filtered_objects
    else:
        detected_objects = other_objects
    
    # 重新計算類別計數
    class_counts = {'RFID': 0, 'cell': 0, 'point': 0}
    for obj in detected_objects:
        class_name = obj['class']
        if class_name in class_counts:
            class_counts[class_name] += 1
    
    print(f"  過濾後結果 - RFID: {class_counts['RFID']}, cell: {class_counts['cell']}, point: {class_counts['point']}")
    
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
    print("Faster R-CNN ResNet-50 FPN V2 細胞偵測 - 圖片標記")
    print("=" * 60)
    
    ensure_folders()
    
    image_paths = get_image_paths(START_FOLDER)
    
    if not image_paths:
        print(f"錯誤：在 {START_FOLDER} 資料夾中找不到圖片檔案！")
        return
    
    print(f"找到 {len(image_paths)} 張圖片，開始處理...")
    print("=" * 60)
    
    # 載入模型
    print("正在載入 Faster R-CNN 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    try:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"模型載入成功: {MODEL_PATH}")
    except Exception as e:
        print(f"錯誤：無法載入模型 {MODEL_PATH}")
        print(f"錯誤訊息: {e}")
        return
    
    # 處理所有圖片
    success_count = 0
    for image_path in image_paths:
        if process_image(image_path, model, device):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"處理完成！成功處理 {success_count}/{len(image_paths)} 張圖片")
    print("=" * 60)

if __name__ == "__main__":
    main()

