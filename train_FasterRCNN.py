#!/usr/bin/env python3
"""
Faster R-CNN ResNet-50 FPN V2 細胞偵測模型訓練程式 - DB 訓練集
"""

import os
import shutil
import cv2
import numpy as np
import glob
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F

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

# ============================================================================
# 訓練配置
# ============================================================================

TRAIN_CONFIG = {
    'epochs': 50,
    'batch_size': 4,  # Faster R-CNN 通常需要較小的 batch size
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'step_size': 20,  # 學習率衰減週期
    'gamma': 0.1,     # 學習率衰減因子
    'num_classes': 4,  # 背景(0) + RFID(1) + cell(2) + point(3)
    'model_save_dir': 'runs/FasterRCNN_DB_cell_detection12',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 類別映射
CLASS_MAPPING = {
    0: 'RFID',
    1: 'cell',
    2: 'point'
}

# ============================================================================
# 預處理參數（與 train_DB.py 保持一致）
# ============================================================================

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
# 預處理函數（與 train_DB.py 一致）
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
    """預處理圖片（與 train_DB.py 完全一致）"""
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
# PyTorch Dataset 類別
# ============================================================================

class CellDetectionDataset(Dataset):
    """細胞偵測資料集（YOLO 格式轉 PyTorch 格式）"""
    
    def __init__(self, images_dir, labels_dir, preprocess=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.preprocess = preprocess
        
        # 獲取所有圖片文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"無法讀取圖片: {img_path}")
        
        h, w = img.shape[:2]
        
        # 預處理
        if self.preprocess:
            img = preprocess_image(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 轉換為 RGB 和 tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)
        
        # 讀取標註
        label_path = os.path.join(
            self.labels_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1]) * w
                        center_y = float(parts[2]) * h
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h
                        
                        # 轉換為 [x1, y1, x2, y2] 格式
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # 確保座標在圖片範圍內
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(x1, min(x2, w))
                        y2 = max(y1, min(y2, h))
                        
                        if x2 > x1 and y2 > y1:  # 確保有效的框
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # +1 因為背景是 0
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        return img, target

# ============================================================================
# 訓練函數
# ============================================================================

def collate_fn(batch):
    """自訂 collate 函數處理不同大小的圖片和標註"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} - 平均 Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, data_loader, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def setup_preprocessed_folder():
    """設置預處理資料夾結構（與 train_DB.py 一致）"""
    if os.path.exists(DatasetConfig.PREPROCESSED_FOLDER):
        shutil.rmtree(DatasetConfig.PREPROCESSED_FOLDER)
    
    os.makedirs(DatasetConfig.get_preprocessed_train_dir(), exist_ok=True)
    os.makedirs(DatasetConfig.get_preprocessed_labels_train_dir(), exist_ok=True)
    
    if os.path.exists(DatasetConfig.VAL_IMAGES_DIR):
        os.makedirs(DatasetConfig.get_preprocessed_val_dir(), exist_ok=True)
        os.makedirs(DatasetConfig.get_preprocessed_labels_val_dir(), exist_ok=True)

def copy_and_preprocess_images():
    """複製並預處理圖片（與 train_DB.py 一致）"""
    print("\n正在複製並預處理圖片...")
    
    # 複製訓練圖片和標註
    train_images = glob.glob(os.path.join(DatasetConfig.TRAIN_IMAGES_DIR, '*.*'))
    train_images = [f for f in train_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_path in train_images:
        img = cv2.imread(img_path)
        if img is not None:
            preprocessed = preprocess_image(img)
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            output_path = os.path.join(DatasetConfig.get_preprocessed_train_dir(), os.path.basename(img_path))
            cv2.imwrite(output_path, preprocessed)
    
    # 複製訓練標註
    train_labels = glob.glob(os.path.join(DatasetConfig.TRAIN_LABELS_DIR, '*.txt'))
    for label_path in train_labels:
        shutil.copy2(label_path, DatasetConfig.get_preprocessed_labels_train_dir())
    
    print(f"  已處理 {len(train_images)} 張訓練圖片")
    
    # 複製驗證圖片和標註（如果存在）
    if os.path.exists(DatasetConfig.VAL_IMAGES_DIR):
        val_images = glob.glob(os.path.join(DatasetConfig.VAL_IMAGES_DIR, '*.*'))
        val_images = [f for f in val_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_path in val_images:
            img = cv2.imread(img_path)
            if img is not None:
                preprocessed = preprocess_image(img)
                preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                output_path = os.path.join(DatasetConfig.get_preprocessed_val_dir(), os.path.basename(img_path))
                cv2.imwrite(output_path, preprocessed)
        
        if os.path.exists(DatasetConfig.VAL_LABELS_DIR):
            val_labels = glob.glob(os.path.join(DatasetConfig.VAL_LABELS_DIR, '*.txt'))
            for label_path in val_labels:
                shutil.copy2(label_path, DatasetConfig.get_preprocessed_labels_val_dir())
        
        if val_images:
            print(f"  已處理 {len(val_images)} 張驗證圖片")

def main():
    """主程式流程"""
    print("=" * 60)
    print("Faster R-CNN ResNet-50 FPN V2 細胞偵測模型訓練")
    print("=" * 60)
    
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"使用設備: {device}")
    
    # 設置預處理資料夾
    setup_preprocessed_folder()
    copy_and_preprocess_images()
    
    # 創建資料集
    print("\n載入訓練資料集...")
    train_dataset = CellDetectionDataset(
        DatasetConfig.get_preprocessed_train_dir(),
        DatasetConfig.get_preprocessed_labels_train_dir(),
        preprocess=False  # 已經預處理過了
    )
    print(f"訓練圖片數量: {len(train_dataset)}")
    
    val_dataset = None
    if os.path.exists(DatasetConfig.get_preprocessed_val_dir()):
        print("\n載入驗證資料集...")
        val_dataset = CellDetectionDataset(
            DatasetConfig.get_preprocessed_val_dir(),
            DatasetConfig.get_preprocessed_labels_val_dir(),
            preprocess=False  # 已經預處理過了
        )
        print(f"驗證圖片數量: {len(val_dataset)}")
    
    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
    
    # 載入模型
    print("\n載入 Faster R-CNN ResNet-50 FPN V2 模型...")
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # 修改分類器
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 
        TRAIN_CONFIG['num_classes']
    )
    
    model.to(device)
    print("模型載入完成")
    
    # 優化器和學習率調度器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=TRAIN_CONFIG['learning_rate'],
        momentum=TRAIN_CONFIG['momentum'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAIN_CONFIG['step_size'],
        gamma=TRAIN_CONFIG['gamma']
    )
    
    # 創建儲存目錄
    os.makedirs(TRAIN_CONFIG['model_save_dir'], exist_ok=True)
    best_loss = float('inf')
    
    # 訓練循環
    print("\n開始訓練...")
    print("=" * 60)
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1} - 驗證 Loss: {val_loss:.4f}")
            
            # 儲存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(TRAIN_CONFIG['model_save_dir'], 'best.pth')
                )
                print(f"  ✓ 儲存最佳模型 (Loss: {val_loss:.4f})")
        else:
            # 沒有驗證集時，儲存每個 epoch 的模型
            torch.save(
                model.state_dict(),
                os.path.join(TRAIN_CONFIG['model_save_dir'], f'epoch_{epoch+1}.pth')
            )
        
        lr_scheduler.step()
    
    # 儲存最後的模型
    torch.save(
        model.state_dict(),
        os.path.join(TRAIN_CONFIG['model_save_dir'], 'last.pth')
    )
    
    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"模型儲存在: {TRAIN_CONFIG['model_save_dir']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

