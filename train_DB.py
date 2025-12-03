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
PREPROCESSED_FOLDER = 'DB預處理'  # 預處理後的圖片存放資料夾
PREPROCESSED_TRAIN_DIR = os.path.join(PREPROCESSED_FOLDER, 'images', 'train')
PREPROCESSED_VAL_DIR = os.path.join(PREPROCESSED_FOLDER, 'images', 'val')
PREPROCESSED_LABELS_TRAIN_DIR = os.path.join(PREPROCESSED_FOLDER, 'labels', 'train')
PREPROCESSED_LABELS_VAL_DIR = os.path.join(PREPROCESSED_FOLDER, 'labels', 'val')
PREPROCESSED_YAML = 'DB預處理_dataset.yaml'  # 預處理資料集的 yaml 檔案

# 訓練配置
TRAIN_CONFIG = {
    'base_model': 'yolov8n.pt',     # 基礎模型
    'data_yaml': DATASET_YAML,      # 資料集設定檔
    'epochs': 50,                   # 訓練輪數
    'imgsz': 640,                   # 圖片大小
    'batch': 16,                    # 批次大小
    'name': 'DB_cell_detection1',   # 專案名稱（固定為 1，每次覆蓋）
    'patience': 25,                 # 早停耐心值
}

# 資料擴增配置
AUGMENTATION_CONFIG = {
    # 顏色增強
    'hsv_h': 0.015,                 # 色調變化範圍（±1.5%，輕微）
    'hsv_s': 0.7,                   # 飽和度變化範圍（±70%，適中）
    'hsv_v': 0.4,                   # 亮度變化範圍（±40%，適中）
    
    # 幾何變換（保守設定避免小物體變形）
    'degrees': 5.0,                 # 旋轉角度範圍（±5度）
    'translate': 0.1,               # 平移範圍（±10%）
    'scale': 0.2,                   # 縮放範圍（0.8-1.2倍）
    'shear': 2.0,                   # 剪切角度（±2度）
    'perspective': 0.0005,          # 透視變換（0.0005，輕微）
    
    # 翻轉
    'flipud': 0.5,                  # 上下翻轉機率（50%機率）
    'fliplr': 0.5,                  # 左右翻轉機率（50%機率）
    
    # 進階增強
    'mosaic': 0.5,                  # 馬賽克增強機率（50%機率）
    'mixup': 0.15,                  # 混合增強機率（15%機率）
    'copy_paste': 0.0,              # 複製貼上增強機率（0 = 關閉，避免標籤混亂）
}

# 預處理參數（GIMP 風格）
PREPROCESS_PARAMS = {
    # 銳利化參數（Unsharp Mask）
    'sharpen_radius': 1.0,          # 銳化半徑
    'sharpen_amount': 1.0,          # 銳化強度（0.0-5.0）
    'sharpen_threshold': 0.0,       # 銳化閾值
    
    # 降低雜訊參數（非局部均值去噪）
    'denoise_h': 10.0,              # 過濾強度（越大去噪越強，但可能模糊細節）
    'denoise_templateWindowSize': 7, # 模板窗口大小（必須為奇數）
    'denoise_searchWindowSize': 21  # 搜索窗口大小（必須為奇數）
}

# ============================================================================
# 預處理函數
# ============================================================================

def preprocess_image(image):
    """
    預處理圖片（GIMP 風格）
    依序使用：顏色轉灰階（GIMP 亮度方法）、銳利化、降低雜訊
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的圖片（灰階格式）
    """
    # 1. 顏色轉灰階（使用 GIMP 的亮度方法）
    # GIMP 使用亮度（Luminosity）公式：0.299*R + 0.587*G + 0.114*B
    # OpenCV 的 BGR 格式，所以是：0.114*B + 0.587*G + 0.299*R
    b, g, r = cv2.split(image)
    gray = (0.114 * b.astype(np.float32) + 
            0.587 * g.astype(np.float32) + 
            0.299 * r.astype(np.float32)).astype(np.uint8)
    
    # 2. 銳利化（使用 Unsharp Mask 方法，類似 GIMP）
    # Unsharp Mask: 原圖 - 模糊圖，然後加到原圖上
    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        PREPROCESS_PARAMS['sharpen_radius']
    )
    sharpened = cv2.addWeighted(
        gray,
        1.0 + PREPROCESS_PARAMS['sharpen_amount'],
        blurred,
        -PREPROCESS_PARAMS['sharpen_amount'],
        0
    )
    
    # 應用閾值（如果設定）
    if PREPROCESS_PARAMS['sharpen_threshold'] > 0:
        # 計算差異
        diff = cv2.absdiff(gray, blurred)
        # 只對差異大於閾值的區域進行銳化
        mask = diff > PREPROCESS_PARAMS['sharpen_threshold']
        sharpened = np.where(mask, sharpened, gray).astype(np.uint8)
    
    # 3. 降低雜訊（使用非局部均值去噪，類似 GIMP 的降噪功能）
    denoised = cv2.fastNlMeansDenoising(
        sharpened,
        h=PREPROCESS_PARAMS['denoise_h'],
        templateWindowSize=PREPROCESS_PARAMS['denoise_templateWindowSize'],
        searchWindowSize=PREPROCESS_PARAMS['denoise_searchWindowSize']
    )
    
    return denoised

def copy_images_to_preprocessed_folder():
    """複製訓練集圖片和標籤到 DB預處理 資料夾（會覆蓋現有檔案）"""
    print("\n正在複製圖片和標籤到 DB預處理 資料夾...")
    
    # 清空並創建預處理資料夾結構
    if os.path.exists(PREPROCESSED_FOLDER):
        shutil.rmtree(PREPROCESSED_FOLDER)
    
    os.makedirs(PREPROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_LABELS_TRAIN_DIR, exist_ok=True)
    
    if os.path.exists(VAL_IMAGES_DIR):
        os.makedirs(PREPROCESSED_VAL_DIR, exist_ok=True)
        os.makedirs(PREPROCESSED_LABELS_VAL_DIR, exist_ok=True)
    
    # 複製訓練圖片
    train_images = glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*.jpg')) + \
                   glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*.jpeg')) + \
                   glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*.png')) + \
                   glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*.JPG')) + \
                   glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*.JPEG')) + \
                   glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*.PNG'))
    
    train_count = 0
    for img_path in train_images:
        filename = os.path.basename(img_path)
        dest_path = os.path.join(PREPROCESSED_TRAIN_DIR, filename)
        shutil.copy2(img_path, dest_path)
        train_count += 1
    
    # 複製訓練標籤
    train_labels = glob.glob(os.path.join(TRAIN_LABELS_DIR, '*.txt'))
    label_count = 0
    for label_path in train_labels:
        filename = os.path.basename(label_path)
        dest_path = os.path.join(PREPROCESSED_LABELS_TRAIN_DIR, filename)
        shutil.copy2(label_path, dest_path)
        label_count += 1
    
    print(f"  已複製 {train_count} 張訓練圖片和 {label_count} 個標籤到 {PREPROCESSED_TRAIN_DIR}")
    
    # 複製驗證圖片和標籤（如果存在）
    val_count = 0
    val_label_count = 0
    if os.path.exists(VAL_IMAGES_DIR):
        val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, '*.jpg')) + \
                     glob.glob(os.path.join(VAL_IMAGES_DIR, '*.jpeg')) + \
                     glob.glob(os.path.join(VAL_IMAGES_DIR, '*.png')) + \
                     glob.glob(os.path.join(VAL_IMAGES_DIR, '*.JPG')) + \
                     glob.glob(os.path.join(VAL_IMAGES_DIR, '*.JPEG')) + \
                     glob.glob(os.path.join(VAL_IMAGES_DIR, '*.PNG'))
        
        for img_path in val_images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(PREPROCESSED_VAL_DIR, filename)
            shutil.copy2(img_path, dest_path)
            val_count += 1
        
        # 複製驗證標籤
        if os.path.exists(VAL_LABELS_DIR):
            val_labels = glob.glob(os.path.join(VAL_LABELS_DIR, '*.txt'))
            for label_path in val_labels:
                filename = os.path.basename(label_path)
                dest_path = os.path.join(PREPROCESSED_LABELS_VAL_DIR, filename)
                shutil.copy2(label_path, dest_path)
                val_label_count += 1
        
        if val_count > 0:
            print(f"  已複製 {val_count} 張驗證圖片和 {val_label_count} 個標籤到 {PREPROCESSED_VAL_DIR}")
    
    return train_count + val_count

def preprocess_images_in_folder():
    """對 DB預處理 資料夾中的圖片進行預處理（覆蓋）"""
    print("\n正在預處理 DB預處理 資料夾中的圖片...")
    
    # 處理訓練圖片
    train_images = glob.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*.jpg')) + \
                   glob.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*.jpeg')) + \
                   glob.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*.png')) + \
                   glob.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*.JPG')) + \
                   glob.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*.JPEG')) + \
                   glob.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*.PNG'))
    
    train_count = 0
    for img_path in train_images:
        image = cv2.imread(img_path)
        if image is not None:
            preprocessed = preprocess_image(image)
            cv2.imwrite(img_path, preprocessed)  # 覆蓋原檔案
            train_count += 1
    
    print(f"  已預處理 {train_count} 張訓練圖片")
    
    # 處理驗證圖片（如果存在）
    val_count = 0
    if os.path.exists(PREPROCESSED_VAL_DIR):
        val_images = glob.glob(os.path.join(PREPROCESSED_VAL_DIR, '*.jpg')) + \
                     glob.glob(os.path.join(PREPROCESSED_VAL_DIR, '*.jpeg')) + \
                     glob.glob(os.path.join(PREPROCESSED_VAL_DIR, '*.png')) + \
                     glob.glob(os.path.join(PREPROCESSED_VAL_DIR, '*.JPG')) + \
                     glob.glob(os.path.join(PREPROCESSED_VAL_DIR, '*.JPEG')) + \
                     glob.glob(os.path.join(PREPROCESSED_VAL_DIR, '*.PNG'))
        
        for img_path in val_images:
            image = cv2.imread(img_path)
            if image is not None:
                preprocessed = preprocess_image(image)
                cv2.imwrite(img_path, preprocessed)  # 覆蓋原檔案
                val_count += 1
        
        if val_count > 0:
            print(f"  已預處理 {val_count} 張驗證圖片")
    
    return train_count + val_count

def create_preprocessed_yaml():
    """創建預處理資料集的 yaml 檔案"""
    print("\n正在創建預處理資料集設定檔...")
    
    # 讀取原始 yaml 檔案以獲取類別資訊
    with open(DATASET_YAML, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # 創建預處理資料集的 yaml 內容
    yaml_content = f"""# DB預處理 訓練集設定檔

# 資料集路徑（相對於此檔案）
path: ./{PREPROCESSED_FOLDER}  # DB預處理 資料集根目錄
train: images/train  # 訓練圖片路徑（相對於 path）
val: images/val      # 驗證圖片路徑（相對於 path）

# 類別名稱（三個類別）
names:
  0: RFID   # 類別 0：RFID 標籤
  1: cell   # 類別 1：菌落
  2: point  # 類別 2：疙瘩

# 類別數量
nc: 3  # 三個類別
"""
    
    # 寫入 yaml 檔案
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
    
    # 檢查訓練資料
    train_images_exist = os.path.exists(TRAIN_IMAGES_DIR) and len(os.listdir(TRAIN_IMAGES_DIR)) > 0
    train_labels_exist = os.path.exists(TRAIN_LABELS_DIR) and len([f for f in os.listdir(TRAIN_LABELS_DIR) if f.endswith('.txt')]) > 0
    
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
    image_count = len([f for f in os.listdir(TRAIN_IMAGES_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_count = len([f for f in os.listdir(TRAIN_LABELS_DIR) 
                      if f.endswith('.txt')])
    
    val_count = 0
    if os.path.exists(VAL_IMAGES_DIR):
        val_count = len([f for f in os.listdir(VAL_IMAGES_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return {
        'train_images': image_count,
        'train_labels': label_count,
        'val_images': val_count
    }

def print_statistics(stats):
    """列印資料集統計資訊"""
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
    print("\n預處理設定（GIMP 風格）：")
    print("  - 步驟 1：顏色轉灰階（GIMP 亮度方法：0.299*R + 0.587*G + 0.114*B）")
    print(f"  - 步驟 2：銳利化（Unsharp Mask，半徑={PREPROCESS_PARAMS['sharpen_radius']}, 強度={PREPROCESS_PARAMS['sharpen_amount']}）")
    print(f"  - 步驟 3：降低雜訊（非局部均值去噪，強度={PREPROCESS_PARAMS['denoise_h']}）")
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
            # 顏色增強
            hsv_h=AUGMENTATION_CONFIG['hsv_h'],
            hsv_s=AUGMENTATION_CONFIG['hsv_s'],
            hsv_v=AUGMENTATION_CONFIG['hsv_v'],
            
            # 幾何變換
            degrees=AUGMENTATION_CONFIG['degrees'],
            translate=AUGMENTATION_CONFIG['translate'],
            scale=AUGMENTATION_CONFIG['scale'],
            shear=AUGMENTATION_CONFIG['shear'],
            perspective=AUGMENTATION_CONFIG['perspective'],
            
            # 翻轉
            flipud=AUGMENTATION_CONFIG['flipud'],
            fliplr=AUGMENTATION_CONFIG['fliplr'],
            
            # 進階增強
            mosaic=AUGMENTATION_CONFIG['mosaic'],
            mixup=AUGMENTATION_CONFIG['mixup'],
            copy_paste=AUGMENTATION_CONFIG['copy_paste'],
        )
        
        print("\n" + "=" * 60)
        print("訓練完成！")
        print("=" * 60)
        print(f"最後一個檢查點: {results.save_dir}/weights/last.pt")
        print("\n使用方式：")
        print(f"  模型路徑: {results.save_dir}/weights/best.pt")
        
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
    
    # 5. 複製訓練集圖片和標籤到 DB預處理 資料夾（會覆蓋現有檔案）
    copy_images_to_preprocessed_folder()
    
    # 6. 對 DB預處理 資料夾中的圖片進行預處理（覆蓋）
    preprocess_images_in_folder()
    
    # 7. 創建預處理資料集的 yaml 檔案
    create_preprocessed_yaml()
    
    # 8. 訓練模型（使用預處理後的圖片）
    # 臨時修改訓練配置使用預處理資料集
    original_data_yaml = TRAIN_CONFIG['data_yaml']
    TRAIN_CONFIG['data_yaml'] = PREPROCESSED_YAML
    
    results = train_model(model)
    
    # 恢復原始配置
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
