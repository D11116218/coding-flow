import os
import shutil
from pathlib import Path

# 定義來源和目標資料夾
source_dir = Path(r"C:\Users\willes.chen\BB\file_start")
target_dir = Path(r"C:\Users\willes.chen\BB\file_finish")

# 確保目標資料夾存在
target_dir.mkdir(exist_ok=True)

# 計數器
copied_count = 0
skipped_count = 0

# 遍歷 D 資料夾內的每個子資料夾
for subfolder in source_dir.iterdir():
    if subfolder.is_dir():
        # 遍歷子資料夾內的所有檔案
        for file in subfolder.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.txt']:
                target_file = target_dir / file.name
                
                # 如果目標檔案已存在，跳過
                if target_file.exists():
                    print(f"跳過（已存在）: {file.name}")
                    skipped_count += 1
                else:
                    shutil.copy2(file, target_file)
                    print(f"已處理: {file.name}")
                    copied_count += 1

print(f"\n完成！ 已處理 {copied_count} 個檔案，跳過 {skipped_count} 個檔案。")