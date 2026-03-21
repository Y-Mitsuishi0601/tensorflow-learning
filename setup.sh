#!/bin/bash

# エラー発生時にスクリプトを停止
set -e

echo "環境構築を開始します..."

# 1. uvを使用して仮想環境の作成と依存関係のインストール (pyproject.tomlを自動参照)
echo "依存パッケージをインストール中..."
uv sync

# 2. Jupyterカーネルの登録
echo "Jupyterカーネルを登録中..."
uv run python -m ipykernel install --user --name=tf-mac-env --display-name="Python (TF-Mac)"

echo "環境構築が完了しました!"