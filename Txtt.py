# JSON 轉 TXT 標註檔轉換工具
# 從 T1 資料夾讀取 JPG 檔案，尋找對應的 JSON 標註檔，轉換為 YOLO 格式的 TXT，輸出到 T2 資料夾

import json
import os
import cv2
from pathlib import Path
import shutil

def convert_json_to_yolo_txt(json_path, image_path, output_txt_path):
    """
    將 anylabeling 的 JSON 標註檔轉換為 YOLO 格式的 TXT
    
    參數:
    - json_path: JSON 標註檔路徑
    - image_path: 對應的圖片路徑（用於取得圖片尺寸）
    - output_txt_path: 輸出的 TXT 檔案路徑
    
    返回:
    - True 如果轉換成功，False 如果失敗
    """
    # 讀取圖片尺寸
    img = cv2.imread(image_path)
    
    img_height, img_width = img.shape[:2]
    # 讀取 JSON 檔案
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False
    
    # 解析 JSON 並轉換為 YOLO 格式
    yolo_lines = []
    
    # anylabeling 的 JSON 格式可能有多種，嘗試常見的格式
    if 'shapes' in data:
        # 格式 1: shapes 陣列
        shapes = data['shapes']
    elif isinstance(data, list):
        # 格式 2: 直接是陣列
        shapes = data
    else:
        return False
    
    for shape in shapes:
        # 取得標籤（類別名稱）
        label = shape.get('label', 'cell')
        
        # 類別 ID（細胞是 0）
        class_id = 0
        
        # 取得座標點
        points = shape.get('points', [])
        
        if len(points) < 2:
            continue
        
        # 計算邊界框（bounding box）
        # points 可能是 [[x1, y1], [x2, y2]] 或 [[x1, y1], [x2, y2], ...]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # 計算中心點和寬高（像素座標）
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # 轉換為歸一化座標（YOLO 格式）
        norm_center_x = center_x / img_width
        norm_center_y = center_y / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        # 確保座標在 0-1 範圍內
        norm_center_x = max(0, min(1, norm_center_x))
        norm_center_y = max(0, min(1, norm_center_y))
        norm_width = max(0, min(1, norm_width))
        norm_height = max(0, min(1, norm_height))
        
        # YOLO 格式：class_id center_x center_y width height
        yolo_line = f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n"
        yolo_lines.append(yolo_line)
    
    # 寫入 TXT 檔案
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.writelines(yolo_lines)
        return True
    except Exception as e:
        return False

def convert_single_file(json_path, image_path=None, output_path=None):
    """
    轉換單個 JSON 檔案
    
    參數:
    - json_path: JSON 檔案路徑
    - image_path: 圖片路徑（如果為 None，會自動尋找同名的圖片）
    - output_path: 輸出的 TXT 路徑（如果為 None，會自動產生）
    """
    json_path = Path(json_path)
    
    # 自動尋找對應的圖片
    if image_path is None:
        # 嘗試尋找同名的圖片檔案
        json_dir = json_path.parent
        json_name = json_path.stem  # 不含副檔名的檔名
        
        # 嘗試各種圖片格式
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_image = json_dir / f"{json_name}{ext}"
            if possible_image.exists():
                image_path = str(possible_image)
                break       
    
    # 自動產生輸出路徑
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}.txt"
    else:
        output_path = Path(output_path)
    
    # 轉換
    if convert_json_to_yolo_txt(str(json_path), image_path, str(output_path)):
        print(f"[成功] {json_path.name} -> {output_path.name}")
        return True
    else:
        print(f"[失敗] {json_path.name}")
        return False

def process_jpg_file(jpg_path, t1_folder, t2_folder):
    """
    處理單個 JPG 檔案：尋找對應的 JSON，轉換為 TXT，輸出到 T2
    
    參數:
    - jpg_path: JPG 檔案路徑
    - t1_folder: T1 資料夾路徑
    - t2_folder: T2 資料夾路徑
    
    返回:
    - True 如果轉換成功，False 如果失敗
    """
    jpg_path = Path(jpg_path)
    jpg_name = jpg_path.stem  # 不含副檔名的檔名
    
    # 在 T1 資料夾中尋找對應的 JSON 檔案
    json_path = t1_folder / f"{jpg_name}.json"
    
    if not json_path.exists():
        print(f"[跳過] {jpg_path.name} - 找不到對應的 JSON 檔案 ({json_path.name})")
        return False
    
    # 輸出的 TXT 檔案路徑（在 T2 資料夾中）
    txt_output_path = t2_folder / f"{jpg_name}.txt"
    
    # 轉換 JSON 為 TXT
    if convert_json_to_yolo_txt(str(json_path), str(jpg_path), str(txt_output_path)):
        print(f"[成功] {jpg_path.name} -> {txt_output_path.name}")
        return True
    else:
        print(f"[失敗] {jpg_path.name} - 轉換失敗")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("JSON 轉 TXT 標註檔轉換工具")
    print("從 T1 資料夾讀取 JPG 檔案，轉換為 TXT 格式，輸出到 T2 資料夾")
    print("=" * 60)
    print()
    
    # T1 資料夾（輸入）
    t1_folder = Path('T1')
    
    if not t1_folder.exists():
        print(f"錯誤：找不到 T1 資料夾！")
        print(f"請確認 T1 資料夾存在於當前目錄。")
        exit(1)
    
    # T2 資料夾（輸出）
    t2_folder = Path('T2')
    t2_folder.mkdir(exist_ok=True)
    
    # 取得 T1 資料夾中所有 JPG 檔案
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    jpg_files = []
    for ext in image_extensions:
        jpg_files.extend(list(t1_folder.glob(ext)))
    
    if not jpg_files:
        print(f"在 {t1_folder} 資料夾中未找到 JPG 檔案")
        exit(1)
    
    print(f"找到 {len(jpg_files)} 個 JPG 檔案，開始轉換...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for jpg_file in jpg_files:
        if process_jpg_file(jpg_file, t1_folder, t2_folder):
            success_count += 1
        else:
            fail_count += 1
    
    print("=" * 60)
    print(f"Json轉換txt完成！成功: {success_count}, 失敗: {fail_count}")
    print(f"\n所有轉換完成的 TXT 檔案都已儲存在 {t2_folder} 資料夾中")

