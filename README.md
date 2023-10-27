# プログラム概要
このプログラムは、文章をLLM (Large Language Model) に送信し、校正を受けることができるツールです。ユーザーが入力したテキストをLLMにリクエストとして送信し、大規模言語モデルからのフィードバックを受け取ります。これにより、文法や表現の誤りを修正し、より自然な文章を作成する手助けをします。

# Conda環境
### セットアップ
```
conda create --name TextProofreading python=3.8 -y
conda activate TextProofreading
```
### パッケージのインストール
```
pip install --upgrade pip
pip install -r requirements.txt
```
### 実行例
```
python main.py --api_key=<OpenAIのAPIKey>
```
### 環境削除
```
conda deactivate
conda remove --name TextProofreading --all -y
```

# Docker環境
### セットアップ
```
docker build -t text_proofreading -f Dockerfile .
```
### コンテナ実行
```
docker run -v ${PWD}/:/app/ --rm -it text_proofreading /bin/bash
```
### 実行例
```
python main.py --api_key=<OpenAIのAPIKey>
```
### コンテナ終了・削除
```
exit
docker rmi text_proofreading
```

# コマンドライン
|コマンドライン引数|概要|
| --- | --- |
|`--api_key`|OpenAIのAPI-Key|
|`--data_path`|校正したいテキスト一覧のcsvファイルパス|
|`--template_prompt_path`|LLMへリクエストするプロンプトのファイルパス|
|`--output`|結果を保存したいディレクトリパス|
|`--model`|LLMのモデル デフォルトではgpt-3.5-turboを設定|
|`--batch_size`|LLMへリクエストするバッチ数|
|`--temperature`|LLMの応答のランダム性を制御|
