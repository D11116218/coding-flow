#!/bin/bash
# 通用執行腳本：使用 Poetry 環境執行 Python 腳本
# 用法: ./run.sh train_DB.py 或 ./run.sh A3.py
cd "$(dirname "$0")"
if [ $# -eq 0 ]; then
    echo "用法: ./run.sh <python_script.py> [參數...]"
    exit 1
fi
poetry run python "$@"
