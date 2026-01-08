#!/home/dssignal/coding-flow/.venv/bin/python3
# YOLOv12l 細胞偵測模型訓練程式 - DB 訓練集
# 功能：預處理（轉灰階、銳利化、降低雜訊）、資料擴增（旋轉、平移、鏡像）

import os
import shutil
import cv2
import numpy as np
import glob
from ultralytics import YOLO

# 設置 PyTorch CUDA 記憶體分配優化（避免記憶體碎片化）
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

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
    # 模型大小選擇：'s' (Small), 'm' (Medium), 'l' (Large)
    MODEL_SIZE = 'l'  # 可選擇 's', 'm', 'l'
    
    EPOCHS = 500
    IMGSZ = 640  # 圖片尺寸，統一使用 640（YOLO 標準尺寸，32 的倍數，可處理 616-628 的實際圖片尺寸）
    BATCH = 1  # batch size
    WORKERS = 1  # 數據加載線程數
    NAME = 'DB_cell_detection12'  # 訓練結果資料夾名稱
    PROJECT = 'runs'  # 模型儲存路徑
    PATIENCE = 40  # 早停耐心值（40 個 epoch 無改善才停止，給模型更多訓練機會）
    
    @classmethod
    def get_base_model(cls):
        """根據 MODEL_SIZE 返回對應的預訓練模型路徑"""
        model_map = {
            's': 'yolov12s.pt',
            'm': 'yolov12m.pt',
            'l': 'yolov12l.pt'
        }
        return model_map.get(cls.MODEL_SIZE.lower(), 'yolov12l.pt')
    
    @classmethod
    def get_model_size_name(cls):
        """獲取模型大小的完整名稱"""
        size_map = {
            's': 'Small',
            'm': 'Medium',
            'l': 'Large'
        }
        return size_map.get(cls.MODEL_SIZE.lower(), 'Large')
    
    @classmethod
    def get_augmentation_params(cls):
        """
        獲取資料擴增參數
        包含：旋轉、平移、鏡像、拼湊、加權標記疊加、縮放、剪切
        
        停用某個擴增項目：將該參數的值設為 0 即可
        """
        return {
            'degrees': 3.0,       # 旋轉角度：±3°
            'translate': 0.1,      # 平移：最多 10% 的圖片尺寸
            'fliplr': 0.5,         # 左右翻轉機率：50%（鏡像擴增）
            'mosaic': 0.3,         # 拼湊機率：30%
            'mixup': 0.1,          # 加權標記疊加機率：10%
            'scale': 0.5,          # 縮放範圍：0.5 = 50%-150%（±50%）
            'shear': 2.0,          # 剪切角度：±2°
        }


# 預處理參數（與 A3.py 完全一致）
PREPROCESS_PARAMS = {
    # 顏色轉灰階參數
    'grayscale_radius': 300,        # GIMP Radius = 300
    'grayscale_samples': 4,         # GIMP Samples = 4
    'grayscale_iterations': 10,     # GIMP Iterations = 10
    'grayscale_enhance_shadows': False,  # GIMP Enhance Shadows = 未勾選
    
    # 銳利化參數
    'sharpen_radius': 3.0,          # 銳化半徑
    'sharpen_amount': 1.5,           # 銳化強度（降低：從 2.5 降到 1.5）
    'sharpen_threshold': 0.0,       # 銳化閾值
    
    # 降低雜訊參數
    'denoise_h': 11.0,              # 過濾強度
    'denoise_templateWindowSize': 7, # 模板窗口大小
    'denoise_searchWindowSize': 21,  # 搜索窗口大小
    
    # 對比度增強參數（加深黑點）
    'enhance_contrast': True,        # 是否啟用對比度增強
    'contrast_alpha': 1.2,           # 對比度係數（1.0 = 無變化，>1.0 = 增強對比度，<1.0 = 降低對比度）
    'contrast_beta': 0,              # 亮度調整（0 = 無變化）
}

# ============================================================================
# 預處理函數（與 A3.py 完全一致）
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
    預處理圖片（與 A3.py 完全一致）
    依序使用：顏色轉灰階、銳利化、降低雜訊、對比度增強
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的 BGR 三通道圖片（符合 YOLO 要求）
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
    
    # 與 A3.py 一致：預處理並輸出三通道以符合 YOLO 要求
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

# ============================================================================
# 資料處理函數
# ============================================================================

def find_image_files(directory):
    """查找資料夾中的所有圖片檔案"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)

def find_label_files(directory):
    """查找資料夾中的所有標註檔案"""
    label_files = glob.glob(os.path.join(directory, '*.txt'))
    return sorted(label_files)

def preprocess_images_in_directory(directory):
    """對指定資料夾中的所有圖片進行預處理"""
    image_files = find_image_files(directory)
    count = 0
    
    for image_path in image_files:
        try:
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                print(f"  警告：無法讀取圖片 {image_path}")
                continue
            
            # 預處理
            processed_image = preprocess_image(image)
            
            # 覆蓋原檔案
            cv2.imwrite(image_path, processed_image)
            count += 1
        except Exception as e:
            print(f"  錯誤：處理圖片 {image_path} 時發生錯誤: {e}")
    
    return count

def copy_files(source_dir, dest_dir, file_list):
    """複製檔案列表到目標資料夾"""
    os.makedirs(dest_dir, exist_ok=True)
    for file_path in file_list:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file_path, dest_path)

# ============================================================================
# 訓練函數
# ============================================================================

def check_dataset():
    """檢查資料集是否存在"""
    if not os.path.exists(DatasetConfig.TRAIN_IMAGES_DIR):
        print(f"錯誤：找不到訓練圖片資料夾: {DatasetConfig.TRAIN_IMAGES_DIR}")
        return False
    
    if not os.path.exists(DatasetConfig.TRAIN_LABELS_DIR):
        print(f"錯誤：找不到訓練標註資料夾: {DatasetConfig.TRAIN_LABELS_DIR}")
        return False
    
    return True

def get_statistics():
    """獲取資料集統計資訊"""
    train_images = find_image_files(DatasetConfig.TRAIN_IMAGES_DIR)
    train_labels = find_label_files(DatasetConfig.TRAIN_LABELS_DIR)
    val_images = find_image_files(DatasetConfig.VAL_IMAGES_DIR) if os.path.exists(DatasetConfig.VAL_IMAGES_DIR) else []
    
    return {
        'train_images': len(train_images),
        'train_labels': len(train_labels),
        'val_images': len(val_images)
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

def print_train_config():
    """列印訓練配置資訊"""
    print("\n訓練配置：")
    print(f"  模型大小: {TrainConfig.MODEL_SIZE.upper()} ({TrainConfig.get_model_size_name()})")
    print(f"  Epochs: {TrainConfig.EPOCHS}")
    print(f"  圖片尺寸: {TrainConfig.IMGSZ}")
    print(f"  Batch size: {TrainConfig.BATCH}")
    print(f"  Workers: {TrainConfig.WORKERS}")
    print(f"  Patience: {TrainConfig.PATIENCE}")
    
    print("\n預處理設定（與 A3.py 一致）：")
    print("  1. 轉灰階（GIMP 亮度方法）")
    print("  2. 銳利化（Unsharp Mask）")
    print("  3. 降低雜訊（非局部均值去噪）")
    
    print("\n預處理輸出路徑：")
    print(f"  資料夾: {DatasetConfig.PREPROCESSED_FOLDER}")
    print(f"  YAML: {DatasetConfig.PREPROCESSED_YAML}")

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
    
    # 刪除舊的訓練結果資料夾以確保覆蓋
    output_dir = os.path.join(TrainConfig.PROJECT, 'detect', TrainConfig.NAME)
    if os.path.exists(output_dir):
        print(f"刪除舊的訓練結果資料夾: {output_dir}")
        shutil.rmtree(output_dir)
    
    try:
        # 獲取資料擴增參數
        augmentation_params = TrainConfig.get_augmentation_params()
        # 過濾掉值為 0 的參數（停用的擴增項目）
        augmentation_params = {k: v for k, v in augmentation_params.items() if v != 0}
        
        print(f"  使用資料擴增: {list(augmentation_params.keys())}")
        
        # 訓練參數
        train_params = {
            'data': data_yaml,
            'epochs': TrainConfig.EPOCHS,
            'imgsz': TrainConfig.IMGSZ,
            'batch': TrainConfig.BATCH,
            'workers': TrainConfig.WORKERS,
            'name': TrainConfig.NAME,
            'project': TrainConfig.PROJECT,
            'patience': TrainConfig.PATIENCE,
            'device': 0,  # 明確指定 GPU 0
            'save': True,
            'plots': True,
            **augmentation_params  # 應用資料擴增參數（旋轉、平移、鏡像）
        }
        
        print(f"  開始訓練，使用 GPU 0...")
        print(f"  資料擴增參數: {augmentation_params}")
        
        # 開始訓練
        results = model.train(**train_params)
        
        print("\n訓練完成！")
        print(f"最佳模型已儲存在: {os.path.join(output_dir, 'weights', 'best.pt')}")
        
        return results
        
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 預處理資料夾管理
# ============================================================================

def setup_preprocessed_folder():
    """建立預處理資料夾結構"""
    os.makedirs(DatasetConfig.get_preprocessed_train_dir(), exist_ok=True)
    os.makedirs(DatasetConfig.get_preprocessed_val_dir(), exist_ok=True)
    os.makedirs(DatasetConfig.get_preprocessed_labels_train_dir(), exist_ok=True)
    os.makedirs(DatasetConfig.get_preprocessed_labels_val_dir(), exist_ok=True)

def copy_images_to_preprocessed_folder():
    """複製圖片和標籤到 DB預處理 資料夾"""
    print("\n正在複製圖片和標籤到 DB預處理 資料夾...")
    
    # 建立資料夾結構
    setup_preprocessed_folder()
    
    # 複製訓練圖片
    train_images = find_image_files(DatasetConfig.TRAIN_IMAGES_DIR)
    copy_files(DatasetConfig.TRAIN_IMAGES_DIR, DatasetConfig.get_preprocessed_train_dir(), train_images)
    print(f"  已複製 {len(train_images)} 張訓練圖片")
    
    # 複製訓練標籤
    train_labels = find_label_files(DatasetConfig.TRAIN_LABELS_DIR)
    copy_files(DatasetConfig.TRAIN_LABELS_DIR, DatasetConfig.get_preprocessed_labels_train_dir(), train_labels)
    print(f"  已複製 {len(train_labels)} 個訓練標籤")
    
    # 複製驗證圖片（如果存在）
    if os.path.exists(DatasetConfig.VAL_IMAGES_DIR):
        val_images = find_image_files(DatasetConfig.VAL_IMAGES_DIR)
        if val_images:
            copy_files(DatasetConfig.VAL_IMAGES_DIR, DatasetConfig.get_preprocessed_val_dir(), val_images)
            print(f"  已複製 {len(val_images)} 張驗證圖片")
        
        val_labels = find_label_files(DatasetConfig.VAL_LABELS_DIR) if os.path.exists(DatasetConfig.VAL_LABELS_DIR) else []
        if val_labels:
            copy_files(DatasetConfig.VAL_LABELS_DIR, DatasetConfig.get_preprocessed_labels_val_dir(), val_labels)
            print(f"  已複製 {len(val_labels)} 個驗證標籤")

def preprocess_images_in_folder():
    """對 DB預處理 資料夾中的圖片進行預處理"""
    print("\n正在預處理 DB預處理 資料夾中的圖片...")
    print("  預處理步驟：轉灰階、銳利化、降低雜訊")
    
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
path: ./{DatasetConfig.PREPROCESSED_FOLDER}  # DB預處理 資料集根目錄
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
    
    with open(DatasetConfig.PREPROCESSED_YAML, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  已創建設定檔: {DatasetConfig.PREPROCESSED_YAML}")

# ============================================================================
# 模型載入
# ============================================================================

def load_model():
    """載入預訓練模型"""
    base_model = TrainConfig.get_base_model()
    model_size_name = TrainConfig.get_model_size_name()
    print(f"\n正在載入預訓練模型: {base_model} ({model_size_name})...")
    try:
        model = YOLO(base_model)
        print(f"模型載入成功！({model_size_name})")
        return model
    except Exception as e:
        print(f"錯誤：無法載入模型 {base_model}: {e}")
        return None

# ============================================================================
# 主程式
# ============================================================================

def main():
    """主程式流程"""
    model_size = TrainConfig.MODEL_SIZE.upper()
    model_size_name = TrainConfig.get_model_size_name()
    print("=" * 60)
    print(f"YOLOv12{model_size} ({model_size_name}) 細胞偵測模型訓練 - DB 訓練集")
    print("=" * 60)
    print("預處理：轉灰階、銳利化、降低雜訊（與 A3.py 一致）")
    print("資料擴增：旋轉、平移、鏡像、拼湊、加權標記疊加")
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
    
    # 5. 複製圖片和標籤到 DB預處理 資料夾
    copy_images_to_preprocessed_folder()
    
    # 6. 對 DB預處理 資料夾中的圖片進行預處理
    preprocess_images_in_folder()
    
    # 7. 創建預處理資料集設定檔
    create_preprocessed_yaml()
    
    # 8. 訓練模型
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

