# TensorFlow Environment for MacOS

Apple Silicon (M1/M2/M3/M4/M5) 搭載Mac上で、Metalを利用したGPUアクセラレーションを有効にしたTensorFlow環境を構築します。
パッケージ管理には高速な `uv` を使用しています。

## 前提条件
- macOS (Apple Silicon搭載機)
- `uv` がインストールされていること (未インストールの場合は `curl -LsSf https://astral.sh/uv/install.sh | sh` を実行)

## 環境構築手順

1. ターミナルでこのディレクトリに移動します。
2. セットアップスクリプトに実行権限を付与し、実行します。

```bash
chmod +x setup.sh
./setup.sh
```

## GPUの動作確認

### 動作確認 (GPUの認識チェック)
環境構築が完了したら、以下のコマンドでTensorFlowがGPUを認識しているか確認してください。

```bash
uv run python verify_gpu.py
```

「GPUが正常に認識されています！」と表示されれば成功です。

### Jupyter Notebookでの利用
Jupyter Notebook（またはJupyterLab、VS Code等）を起動し、新しくノートブックを作成する際に、カーネルとして Python (TF-Mac) を選択してください。


---

## 実行のヒント
ターミナルで上記ファイルを保存したディレクトリに移動し、`README.md` の手順通りに `setup.sh` を実行するだけで、Jupyterへの紐付けまで自動で完了します。

もしVS Code等のエディタ上でJupyterノートブックを開く際、カーネルの選択肢に「Python (TF-Mac)」がすぐに表示されない場合は、エディタを一度再起動してみてください。
