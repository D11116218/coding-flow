# 檢查訓練集和驗證集的重疊情況
import os
import cv2
import hashlib
from collections import defaultdict

train_dir = 'DB/images/train'
val_dir = 'DB/images/val'

# 1. 檢查相同檔名
train_files = set(os.listdir(train_dir))
val_files = set(os.listdir(val_dir))
same_filename = train_files & val_files

print("=" * 60)
print("檢查結果")
print("=" * 60)
print(f"\n1. 相同檔名重疊: {len(same_filename)} 個")
if same_filename:
    for f in sorted(same_filename):
        print(f"   - {f}")

# 2. 檢查圖片內容是否相同（即使檔名不同）
def calculate_image_hash(image_path):
    """計算圖片的 hash 值"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        # 使用圖片內容計算 hash
        img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    except Exception as e:
        print(f"  錯誤讀取 {image_path}: {e}")
        return None

def compare_images(img1_path, img2_path):
    """比較兩張圖片是否相同"""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return False
        if img1.shape != img2.shape:
            return False
        # 計算差異
        diff = cv2.absdiff(img1, img2)
        return diff.sum() == 0
    except:
        return False

print(f"\n2. 檢查圖片內容是否相同（即使檔名不同）...")
print("   這可能需要一些時間...")

# 計算所有圖片的 hash
train_hashes = {}
val_hashes = {}

print("   正在計算訓練集圖片 hash...")
for filename in train_files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filepath = os.path.join(train_dir, filename)
        img_hash = calculate_image_hash(filepath)
        if img_hash:
            if img_hash not in train_hashes:
                train_hashes[img_hash] = []
            train_hashes[img_hash].append(filename)

print("   正在計算驗證集圖片 hash...")
for filename in val_files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filepath = os.path.join(val_dir, filename)
        img_hash = calculate_image_hash(filepath)
        if img_hash:
            if img_hash not in val_hashes:
                val_hashes[img_hash] = []
            val_hashes[img_hash].append(filename)

# 檢查 hash 重疊
hash_overlap = set(train_hashes.keys()) & set(val_hashes.keys())

print(f"\n3. 相同內容但不同檔名: {len(hash_overlap)} 組")
if hash_overlap:
    for img_hash in sorted(hash_overlap):
        train_names = train_hashes[img_hash]
        val_names = val_hashes[img_hash]
        print(f"   相同內容:")
        print(f"     訓練集: {', '.join(train_names)}")
        print(f"     驗證集: {', '.join(val_names)}")

# 檢查訓練集內部重複
train_duplicates = {h: names for h, names in train_hashes.items() if len(names) > 1}
print(f"\n4. 訓練集內部重複（相同內容不同檔名）: {len(train_duplicates)} 組")
if train_duplicates:
    for img_hash, names in list(train_duplicates.items())[:5]:
        print(f"   {', '.join(names)}")

# 檢查驗證集內部重複
val_duplicates = {h: names for h, names in val_hashes.items() if len(names) > 1}
print(f"\n5. 驗證集內部重複（相同內容不同檔名）: {len(val_duplicates)} 組")
if val_duplicates:
    for img_hash, names in list(val_duplicates.items())[:5]:
        print(f"   {', '.join(names)}")

print("\n" + "=" * 60)
print("總結")
print("=" * 60)
print(f"訓練集圖片數: {len([f for f in train_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])}")
print(f"驗證集圖片數: {len([f for f in val_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])}")
print(f"相同檔名重疊: {len(same_filename)} 個")
print(f"相同內容重疊: {len(hash_overlap)} 組")
if same_filename or hash_overlap:
    print("\n警告：發現重疊！這會導致資料洩漏，影響模型評估準確性。")
else:
    print("\n未發現重疊，資料集劃分正確。")

