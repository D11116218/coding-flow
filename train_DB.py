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
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'name': 'DB_cell_detection1',
    'patience': 25,
}

# 資料擴增配置
AUGMENTATION_CONFIG = {
    # 顏色增強
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    
    # 幾何變換
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.2,
    'shear': 2.0,
    'perspective': 0.0005,
    
    # 翻轉
    'flipud': 0.5,
    'fliplr': 0.5,
    
    # 進階增強
    'mosaic': 0.5,
    'mixup': 0.15,
    'copy_paste': 0.0,
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
        print("請確認標註檔已轉換為 YOLO 格式（.txt）。")
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
    
    print("\n資料擴增設定：")
    print("  - 顏色增強：色調±1.5%、飽和度±70%、亮度±40%")
    print("  - 翻轉：上下翻轉（50%機率）、左右翻轉（50%機率）")
    print("  - 幾何變換：開啟（旋轉±5度、平移±10%、縮放0.8-1.2倍、剪切±2度、透視變換）")
    print("  - 馬賽克增強：開啟（50%機率）")
    print("  - 混合增強：開啟（15%機率）")

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
