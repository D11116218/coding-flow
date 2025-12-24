#!/bin/bash
# 自動使用 Poetry 虛擬環境執行 train_DB.py

cd "$(dirname "$0")"
exec poetry run python train_DB.py "$@"
