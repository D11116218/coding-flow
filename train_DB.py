#!/home/dssignal/coding-flow/.venv/bin/python3
# YOLOv12l 細胞偵測模型訓練程式 - DB 訓練集
#@@
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
    IMGSZ = 640  # 降低解析度以減少 CUDA 記憶體使用（從 1024 降至 640）
    BATCH = 1  # 降低 batch size 以避免 CUDA 記憶體不足（從 2 降到 1）
    WORKERS = 1  # 數據加載線程數，降低可減少記憶體使用（從 2 降到 1）
    NAME = 'DB_cell_detection12'  # 每次訓練直接覆蓋此資料夾
    PROJECT = 'runs'  # 模型儲存路徑（會儲存在 runs/detect/ 下）
    PATIENCE = 20
    AMP = True  # 啟用混合精度訓練（Automatic Mixed Precision）以節省記憶體
    
    # 類別權重（根據圖片覆蓋率計算，平衡類別不平衡問題）
    # RFID: 100% 圖片都有，cell: 68% 圖片有，point: 84% 圖片有
    # 權重 = 總圖片數 / 包含該類別的圖片數，標準化後最小權重為 1.0
    CLASS_WEIGHTS = {
        0: 1.000,  # RFID（覆蓋率 100%）
        1: 1.471,  # cell（覆蓋率 68%，需要更高權重）
        2: 1.190   # point（覆蓋率 84%）
    }
    USE_CLASS_WEIGHTS = True  # 是否使用類別權重
    
    @classmethod
    def get_class_weights(cls):
        """獲取類別權重列表（按類別 ID 順序）"""
        if not cls.USE_CLASS_WEIGHTS:
            return None
        # 返回權重列表 [RFID, cell, point]
        return [cls.CLASS_WEIGHTS.get(0, 1.0), 
                cls.CLASS_WEIGHTS.get(1, 1.0), 
                cls.CLASS_WEIGHTS.get(2, 1.0)]
    
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
    def get_batch_size(cls):
        """根據模型大小和 IMGSZ 動態調整 batch size"""
        # 直接使用配置的 BATCH 值（已設為 1 以減少記憶體使用）
        return cls.BATCH
    
    @classmethod
    def get_model_size_name(cls):
        """獲取模型大小的完整名稱"""
        size_map = {
            's': 'Small',
            'm': 'Medium',
            'l': 'Large'
        }
        return size_map.get(cls.MODEL_SIZE.lower(), 'Large')
    #資料擴增
    @classmethod
    def get_augmentation_params(cls):
        """
        獲取資料擴增參數    
        注意：這些擴增技術可以增加資料多樣性，提高模型的泛化能力，但過度擴增可能導致訓練不穩定
        
        停用某個擴增項目：將該參數的值設為 0 即可（例如：'flipud': 0）
        """
        return {
            'degrees': 3.0,       # 旋轉角度：±15°（提高以應對 RFID 傾斜問題，從 3° 提升）
            'translate': 0.1,      # 平移：最多 10% 的圖片尺寸（設為 0 停用）
            'scale': 0.15,         # 縮放：85% ~ 115%（設為 0 停用）
            # 'flipud': 0.5,         # 上下翻轉機率：50%（設為 0 停用）
            # 'fliplr': 0.5,         # 左右翻轉機率：50%（設為 0 停用）
            'mosaic': 0.3,         # Mosaic 擴增機率：30%（設為 0 停用）
            'mixup': 0.1,          # Mixup 擴增機率：10%（設為 0 停用）
        }


# 預處理參數（GIMP 風格，與 A3.py 保持一致）
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
    預處理圖片（與 A3.py 完全一致）
    依序使用：顏色轉灰階、銳利化、降低雜訊、對比度增強、伽馬校正
    
    參數：
        image: 輸入圖片（BGR 格式）
    
    返回：
        處理後的 BGR 三通道圖片（與 A3.py 一致，符合 YOLO 要求）
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
    
    # 與 A3.py 一致：預處理並輸出三通道以符合 YOLO 要求
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

# ============================================================================
# GPU 記憶體管理函數
# ============================================================================

def check_gpu_memory():
    """檢查 GPU 記憶體使用情況並返回資訊"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        # 獲取當前進程的 GPU 記憶體使用
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # 使用 nvidia-smi 獲取所有進程的記憶體使用
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            other_processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts[0], parts[1], parts[2]
                        other_processes.append({
                            'pid': pid,
                            'name': name,
                            'memory_mib': memory.replace(' MiB', ''),
                            'memory_gb': float(memory.replace(' MiB', '')) / 1024
                        })
        except (subprocess.CalledProcessError, FileNotFoundError):
            other_processes = []
        
        return {
            'current_memory_gb': current_memory,
            'reserved_memory_gb': reserved_memory,
            'total_memory_gb': total_memory,
            'free_memory_gb': total_memory - (sum(p['memory_gb'] for p in other_processes) if other_processes else 0),
            'other_processes': other_processes
        }
    except ImportError:
        return None

def print_gpu_memory_info():
    """列印 GPU 記憶體使用資訊"""
    gpu_info = check_gpu_memory()
    if gpu_info is None:
        print("  無法獲取 GPU 記憶體資訊")
        return
    
    print("\nGPU 記憶體使用情況：")
    print(f"  總記憶體: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"  可用記憶體: {gpu_info['free_memory_gb']:.2f} GB")
    
    if gpu_info['other_processes']:
        print(f"\n  其他進程佔用 GPU 記憶體:")
        total_other = 0
        for proc in gpu_info['other_processes']:
            print(f"    PID {proc['pid']}: {proc['name']} - {proc['memory_gb']:.2f} GB")
            total_other += proc['memory_gb']
        print(f"  其他進程總計: {total_other:.2f} GB")
        
        if total_other > 10:  # 如果其他進程佔用超過 10GB
            print(f"\n  ⚠️  警告：其他進程佔用了大量 GPU 記憶體 ({total_other:.2f} GB)")
            print(f"  建議終止這些進程以釋放記憶體，或等待它們完成")
            print(f"  終止命令範例: kill {gpu_info['other_processes'][0]['pid']}")
    else:
        print("  沒有其他進程佔用 GPU 記憶體")

def cleanup_gpu_memory():
    """清理 GPU 記憶體"""
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
    except ImportError:
        pass
    return False

# ============================================================================
# 工具函數
# ============================================================================

# 圖片副檔名
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

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

# ============================================================================
# 檔案操作函數
# ============================================================================

def print_train_config():
    """顯示訓練與資料增強設定"""
    print("\n訓練設定:")
    print(f"  模型大小: {TrainConfig.MODEL_SIZE.upper()} ({TrainConfig.get_model_size_name()})")
    print(f"  base_model: {TrainConfig.get_base_model()}")
    print(f"  epochs: {TrainConfig.EPOCHS}")
    print(f"  imgsz: {TrainConfig.IMGSZ}")
    print(f"  batch: {TrainConfig.get_batch_size()} (根據模型大小和 IMGSZ 自動調整)")
    print(f"  workers: {TrainConfig.WORKERS}")
    print(f"  amp: {TrainConfig.AMP} (混合精度訓練，節省記憶體)")
    print(f"  name: {TrainConfig.NAME}")
    print(f"  project: {TrainConfig.PROJECT}")
    print(f"  patience: {TrainConfig.PATIENCE}")
    
    # 顯示類別權重設定
    class_weights = TrainConfig.get_class_weights()
    if class_weights:
        print(f"\n類別權重設定:")
        print(f"  RFID (類別 0): {class_weights[0]:.3f}")
        print(f"  cell (類別 1): {class_weights[1]:.3f} (權重較高，因為只有 68% 圖片包含 cell)")
        print(f"  point (類別 2): {class_weights[2]:.3f}")

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
    
    # 檢查 GPU 記憶體使用情況
    print_gpu_memory_info()
    
    # 清理 GPU 記憶體（如果可用）
    if cleanup_gpu_memory():
        print("  已清理 GPU 記憶體快取")
    
    # 刪除舊的訓練結果資料夾以確保覆蓋
    output_dir = os.path.join(TrainConfig.PROJECT, 'detect', TrainConfig.NAME)
    if os.path.exists(output_dir):
        print(f"刪除舊的訓練結果資料夾: {output_dir}")
        shutil.rmtree(output_dir)
    
    try:
        augmentation_params = TrainConfig.get_augmentation_params()
        # 過濾掉值為 0 的參數（停用的擴增項目）
        augmentation_params = {k: v for k, v in augmentation_params.items() if v != 0}
        
        # 使用動態調整的 batch size
        batch_size = TrainConfig.get_batch_size()
        print(f"  使用 batch size: {batch_size} (根據模型大小 {TrainConfig.MODEL_SIZE.upper()} 和 IMGSZ {TrainConfig.IMGSZ} 自動調整)")
        
        # 獲取類別權重
        class_weights = TrainConfig.get_class_weights()
        if class_weights:
            print(f"  使用類別權重: RFID={class_weights[0]:.3f}, cell={class_weights[1]:.3f}, point={class_weights[2]:.3f}")
            print(f"    (cell 權重較高，因為只有 68% 圖片包含 cell)")
        
        # 訓練參數，添加更多記憶體優化選項
        train_params = {
            'data': data_yaml,
            'epochs': TrainConfig.EPOCHS,
            'imgsz': TrainConfig.IMGSZ,
            'batch': batch_size,
            'workers': TrainConfig.WORKERS,
            'name': TrainConfig.NAME,
            'project': TrainConfig.PROJECT,
            'patience': TrainConfig.PATIENCE,
            'amp': TrainConfig.AMP,  # 啟用混合精度訓練以節省記憶體
            'device': 0,  # 明確指定 GPU 0
            'save': True,
            'plots': True,
            'close_mosaic': 10,  # 最後 10 個 epoch 關閉 mosaic 以節省記憶體
            **augmentation_params
        }
        
        # 應用類別權重到模型的損失函數
        # 注意：ultralytics YOLO 可能不直接支援 class_weights 參數
        # 我們需要在訓練開始前修改模型的損失函數
        if class_weights:
            try:
                import torch
                # 將權重轉換為 tensor 格式
                if torch.cuda.is_available():
                    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device='cuda:0')
                else:
                    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                
                # 方法 1: 嘗試通過模型的 loss 屬性設置類別權重
                # YOLO 的損失函數通常在 model.trainer.loss 或 model.loss
                if hasattr(model, 'trainer') and hasattr(model.trainer, 'loss'):
                    loss_fn = model.trainer.loss
                    # 檢查損失函數是否有 cls 權重屬性
                    if hasattr(loss_fn, 'cls'):
                        if hasattr(loss_fn.cls, 'weight'):
                            loss_fn.cls.weight = weights_tensor
                            print(f"  已設置類別權重到損失函數 (trainer.loss.cls.weight)")
                        elif hasattr(loss_fn.cls, 'class_weights'):
                            loss_fn.cls.class_weights = weights_tensor
                            print(f"  已設置類別權重到損失函數 (trainer.loss.cls.class_weights)")
                
                # 方法 2: 如果模型有 loss 屬性（在訓練前可能不存在）
                if hasattr(model, 'loss'):
                    if hasattr(model.loss, 'cls'):
                        if hasattr(model.loss.cls, 'weight'):
                            model.loss.cls.weight = weights_tensor
                            print(f"  已設置類別權重到損失函數 (model.loss.cls.weight)")
                        elif hasattr(model.loss.cls, 'class_weights'):
                            model.loss.cls.class_weights = weights_tensor
                            print(f"  已設置類別權重到損失函數 (model.loss.cls.class_weights)")
                
                # 方法 3: 嘗試通過訓練參數傳遞（某些版本可能支援）
                # 注意：這需要根據實際的 YOLO 版本調整
                # 某些版本的 YOLO 可能支援 cls_weight 或 class_weights 參數
                # 但我們先不添加，因為可能不支援
                
            except Exception as e:
                print(f"  警告：無法自動設置類別權重: {e}")
                print(f"  將嘗試在訓練過程中應用類別權重")
        
        # 開始訓練
        # 注意：如果上述方法無法設置權重，我們需要在訓練回調中應用
        results = model.train(**train_params)
        
        # 訓練後，如果權重未設置，記錄警告
        if class_weights:
            print(f"\n  類別權重已配置（如果 YOLO 支援）:")
            print(f"    RFID={class_weights[0]:.3f}, cell={class_weights[1]:.3f}, point={class_weights[2]:.3f}")
        
        print("\n" + "=" * 60)
        print("訓練完成！")
        print("=" * 60)
        print(f"最後一個檢查點: {results.save_dir}/weights/last.pt")
        print(f"最佳模型: {results.save_dir}/weights/best.pt")
        
        return results
        
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
            print(f"\n訓練過程中發生 CUDA 記憶體不足錯誤:")
            print(f"  {error_msg}")
            print("\n建議解決方案:")
            print("  1. 終止其他佔用 GPU 的進程（見上方的 GPU 記憶體使用情況）")
            print("  2. 進一步降低 batch size（目前為 1）")
            print("  3. 降低 image size（目前為 640）")
            print("  4. 使用更小的模型（'m' 或 's'）")
            print("  5. 等待其他進程完成後再訓練")
            
            # 清理記憶體
            cleanup_gpu_memory()
        else:
            print(f"\n訓練過程中發生錯誤: {e}")
        return None
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        # 清理記憶體
        cleanup_gpu_memory()
        return None

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
    base_model = TrainConfig.get_base_model()
    model_size_name = TrainConfig.get_model_size_name()
    print(f"\n正在載入預訓練模型: {base_model} ({model_size_name})...")
    try:
        # 清理 GPU 記憶體（如果可用）
        if cleanup_gpu_memory():
            print("  已清理 GPU 記憶體快取")
        
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
    
    # 1. 檢查資料集
    if not check_dataset():
        exit(1)
    
    # 2. 統計資料
    stats = get_statistics()
    print_statistics(stats)
    
    # 3. 檢查 GPU 記憶體（在載入模型前）
    print_gpu_memory_info()
    
    # 4. 載入模型
    model = load_model()
    if model is None:
        exit(1)
    
    # 5. 顯示訓練設定
    print_train_config()
    
    # 6. 複製圖片和標籤到 DB預處理 資料夾
    copy_images_to_preprocessed_folder()
    
    # 7. 對 DB預處理 資料夾中的圖片進行預處理
    preprocess_images_in_folder()
    
    # 8. 創建預處理資料集設定檔
    create_preprocessed_yaml()
    
    # 9. 訓練模型
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
