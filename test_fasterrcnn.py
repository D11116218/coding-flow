#!/usr/bin/env python3
"""
測試 Faster R-CNN ResNet-50 FPN V2 模型
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 類別數量（背景 + 3 個類別：RFID, cell, point）
NUM_CLASSES = 4  # 背景(0) + RFID(1) + cell(2) + point(3)

print("=" * 60)
print("Faster R-CNN ResNet-50 FPN V2 模型測試")
print("=" * 60)

# 載入預訓練模型
print("\n1. 載入預訓練模型...")
model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
print("   ✓ 預訓練模型載入成功")

# 修改分類器以適應自訂類別數量
print("\n2. 修改分類器以適應自訂類別...")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
print(f"   ✓ 分類器已修改為 {NUM_CLASSES} 個類別（背景 + RFID + cell + point）")

# 設定為評估模式
model.eval()
print("\n3. 模型已設定為評估模式")

# 顯示模型資訊
print("\n4. 模型資訊：")
print(f"   - 模型類型: Faster R-CNN")
print(f"   - 骨幹網路: ResNet-50 + FPN V2")
print(f"   - 類別數量: {NUM_CLASSES}")
print(f"   - 設備: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

print("\n" + "=" * 60)
print("✓ Faster R-CNN 模型準備完成！")
print("=" * 60)
print("\n使用方式：")
print("1. 訓練：需要創建自訂的 Dataset 類別和訓練循環")
print("2. 推理：使用 model(images) 進行預測")
print("\n注意：Faster R-CNN 需要 PyTorch 風格的數據集，與 YOLO 格式不同")

