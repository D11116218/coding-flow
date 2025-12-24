#!/bin/bash
# 使用 Poetry 環境執行 train_DB.py
cd "$(dirname "$0")"
poetry run python train_DB.py "$@"

