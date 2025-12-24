# 執行說明

## ⚠️ 重要：必須使用 Poetry 環境執行

**問題**：直接使用 `python train_DB.py` 會失敗，因為系統 Python 沒有安裝 `ultralytics`。

## ✅ 正確的執行方式

### 方法 1：使用執行腳本（最簡單）⭐ 推薦

```bash
# 執行訓練程式
./run_train.sh

# 執行 A3.py
./run_A3.sh

# 或使用通用腳本
./run.sh train_DB.py
./run.sh A3.py
```

### 方法 2：使用 `poetry run`

```bash
poetry run python train_DB.py
poetry run python A3.py
```

### 方法 3：激活虛擬環境後執行

```bash
poetry shell
python train_DB.py
python A3.py
# 執行完畢後輸入 exit 退出虛擬環境
```

## 🔍 為什麼會出錯？

- **系統 Python**：`/usr/bin/python` (Python 3.12.3) ❌ 沒有安裝 ultralytics
- **Poetry 環境**：`/home/dssignal/coding-flow/.venv/bin/python` (Python 3.10.18) ✅ 已安裝 ultralytics

## 📝 檢查環境

```bash
# 檢查 Poetry 環境
poetry env info

# 檢查依賴是否安裝
poetry run python -c "import ultralytics; print('✓ ultralytics 可用')"
```

