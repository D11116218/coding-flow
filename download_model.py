#!/usr/bin/env python3
"""
下載 YOLOv12 模型（支援 m 和 x 版本）
使用 GitHub releases 作為下載來源
"""

import os
import sys
import subprocess

def download_model(model_name):
    """下載指定的模型"""
    model_path = f'{model_name}.pt'
    
    print(f"\n{'=' * 60}")
    print(f"正在處理 {model_name}.pt...")
    print(f"{'=' * 60}")
    
    # 如果模型已存在，先檢查大小
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size > 10 * 1024 * 1024:  # 如果文件大於 10MB，認為已下載
            print(f"✓ 模型已存在: {model_path} ({size / 1024 / 1024:.2f} MB)")
            print("  跳過下載。")
            return True
        else:
            print(f"⚠ 發現損壞的模型文件（大小: {size} bytes），將重新下載...")
            os.remove(model_path)
    
    # 從 GitHub releases 下載
    url = f"https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/{model_name}.pt"
    print(f"正在從 GitHub 下載 {model_name}...")
    print(f"URL: {url}")
    
    try:
        # 使用 wget 下載
        result = subprocess.run(
            ['wget', '-O', model_path, url],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 檢查下載是否成功
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            if size > 10 * 1024 * 1024:  # 確認文件大小合理
                print(f"\n✓ {model_name} 下載成功！")
                print(f"  路徑: {os.path.abspath(model_path)}")
                print(f"  大小: {size / 1024 / 1024:.2f} MB")
                
                # 驗證模型是否可以載入
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    print(f"  ✓ 模型驗證成功（可以正常載入）")
                    return True
                except Exception as e:
                    print(f"  ⚠ 模型下載成功但載入失敗: {e}")
                    print(f"  請檢查 ultralytics 版本是否支援 {model_name}")
                    return True  # 仍然認為下載成功
            else:
                print(f"\n✗ {model_name} 下載失敗：文件大小異常 ({size} bytes)")
                return False
        else:
            print(f"\n✗ {model_name} 下載失敗：文件不存在")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 下載失敗: {e}")
        print(f"  請檢查網路連線或手動下載：{url}")
        return False
    except FileNotFoundError:
        print("\n✗ 錯誤：未找到 wget 命令")
        print("  請安裝 wget 或手動下載模型")
        return False
    except Exception as e:
        print(f"\n✗ 下載 {model_name} 時發生錯誤: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv12 模型下載工具")
    print("=" * 60)
    print("\n下載來源：GitHub releases")
    print("  https://github.com/sunsmarterjie/yolov12/releases")
    
    # 下載兩個模型
    models_to_download = ['yolov12m', 'yolov12x']
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"下載完成：{success_count}/{len(models_to_download)} 個模型成功")
    print("=" * 60)
    
    if success_count == len(models_to_download):
        print("\n✓ 所有模型下載成功！")
        print("\n現在您可以在 train_DB.py 和 A3.py 中使用這些模型：")
        print("  - yolov12m.pt (Medium, ~40 MB)")
        print("  - yolov12x.pt (Extra Large, ~114 MB)")
    else:
        print(f"\n⚠ 有 {len(models_to_download) - success_count} 個模型下載失敗")
        sys.exit(1)
