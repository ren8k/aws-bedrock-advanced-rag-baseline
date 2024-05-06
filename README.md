# Knowledge Bases for Amazon Bedrock を利用した Advanced RAG のベースライン<!-- omit in toc -->

本リポジトリでは，2024/05/01 に公開された AWS 公式ブログの記事「[Amazon Kendra と Amazon Bedrock で構成した RAG システムに対する Advanced RAG 手法の精度寄与検証](https://aws.amazon.com/jp/blogs/news/verifying-the-accuracy-contribution-of-advanced-rag-methods-on-rag-systems-built-with-amazon-kendra-and-amazon-bedrock/)」[^0-0]で紹介されている Advanced RAG の再現実装（Python）を公開している．なお，本実装は先日公開した[本リポジトリ](https://github.com/ren8k/aws-bedrock-rag-baseline)[^0-1]をベースとしており，Naive RAG, Advanced RAG の両方を試行できるコードを用意している．

## TL;DR<!-- omit in toc -->

- boto3 ベースで Advanced RAG の実装を行った（以下概要図）．
- 記事[^0-0]で言及されている，非同期処理による LLM, Retrieve の並列実行にも取り組んでいる．
- Claude3 Haiku, Command R+ を利用した Advanced RAG に対応しており，その他のモデルの利用拡張も容易に行える設計である．
- LLM の引数設定，プロンプトなどは yaml ファイルで管理している．

<img src="./assets/architecture.png" width="600">

## 目次<!-- omit in toc -->

- [背景](#背景)
- [目的](#目的)
- [オリジナリティ](#オリジナリティ)
- [前提](#前提)
- [手順](#手順)
- [手順の各ステップの詳細](#手順の各ステップの詳細)
  - [Knowledge Base for Amazon Bedrock の構築（スキップ可能）](#knowledge-base-for-amazon-bedrock-の構築スキップ可能)
  - [Advanced RAG による質問応答の実行](#advanced-rag-による質問応答の実行)
    - [実行例](#実行例)
    - [advanced\_rag.py のアルゴリズム](#advanced_ragpy-のアルゴリズム)
      - [`config/query/query.yaml`](#configqueryqueryyaml)
      - [`config/prompt_template/query_expansion.yaml`](#configprompt_templatequery_expansionyaml)
  - [CleanUp（スキップ可能）](#cleanupスキップ可能)
- [Next Step](#next-step)
- [References](#references)

## 背景

Advanced RAG の方法論などをまとめた記事は多く存在するが，本日時点（2024/05/06）においてその実装例は非常に少ない．特に，boto3 を利用した Advanced RAG の実装例は本リポジトリでは LangChain を利用せず，boto3 のみを利用して シンプルな RAG を実装した．

## 目的

boto3 のみを利用してシンプルな RAG を実装する．また，Python スクリプトベースで実装し，初学者にも理解しやすく，実用的なベースラインを公開する．

## オリジナリティ

- LangChain を利用せず，boto のみを利用して実装している．
- Knowledge Base を Retrieve API 経由で利用することで，Claude3 Haiku や Command R+で質問応答を行っている．
- 利用する LLM を容易に切り替えられるようシンプルな設計にしている．
  - LLM，Retriever, PromptConfig というクラスを定義しており，機能追加に対して柔軟に対応できるようにしている．
- LLM の設定，プロンプトなどは yaml ファイルで管理している．
  - MLflow などと組み合わせることで，実験管理が容易になると考えられる．

## 前提

- バージニア北部リージョン（`us-east-1`）での実行を前提としている．
- Knowledge Base の DB としては，Pinecone を利用している．
  - Pinecone 無料枠を利用することで，ランニングコストゼロでベクトルデータベースを構築可能．
- `requirements.txt` に記載のライブラリがインストールされている．
  - `pip install -r requirements.txt` でインストール可能．
- `適切な認証情報の設定・ロールの設定がなされている．
  - 設定が面倒な場合，Cloud9 上で実行しても良い．
- Bedrock のモデルアクセスの有効化が適切になされている．
  - 本リポジトリ上のコードでは，Claude3 Haiku, Command R+ を利用している．

## 手順

可能な限り検証コストを抑えるため，Pinecone の無料枠を利用して Knowledge Base を構築し，Advanced RAG の実行を行った．手順は以下の通りである．

- Pinecone を利用した Knowledge Base for Amazon Bedrock の構築（スキップ可能）
- Advanced RAG による質問応答の実行
- Naive RAG による質疑応答の実行

## 手順の各ステップの詳細

### Knowledge Base for Amazon Bedrock の構築（スキップ可能）

本記事[^2-0][^2-1]を参考に，Pinecone アカウントの作成，ベクター DB のインデックスの作成を行う．以下に注意点，および Knowledge Base 作成時の設定を示す．

- 無料枠の場合，バージニア北部(us-east-1)リージョンのみ利用可能である．
- 埋め込みモデルとして`Cohere-embed-multilingual-v3.0`を利用する．
- バージニア北部リージョンにて Secrets Manager を作成し，Pinecone の API キーを保存する．
- 簡単のため，データソースの S3 には以下の 2020 ~ 2023 年度の Amazon の株主向け年次報告書を格納し，これを Embeddings vector に変換している．
  - https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/2022-Shareholder-Letter.pdf
  - https://s2.q4cdn.com/299287126/files/doc_financials/2022/ar/2021-Shareholder-Letter.pdf
  - https://s2.q4cdn.com/299287126/files/doc_financials/2021/ar/Amazon-2020-Shareholder-Letter-and-1997-Shareholder-Letter.pdf
  - https://s2.q4cdn.com/299287126/files/doc_financials/2020/ar/2019-Shareholder-Letter.pdf

### Advanced RAG による質問応答の実行

検索したい内容やプロンプトの雛形を yaml ファイルに定義しておき，python スクリプト（[`advanced_rag.py`](https://github.com/ren8k/aws-bedrock-advanced-rag-baseline/blob/main/src/advanced_rag.py)）を実行することで，Advanced RAG による質問応答を行う．以降，実行例およびコードの解説を行う．

#### 実行例

[`./src`](https://github.com/ren8k/aws-bedrock-advanced-rag-baseline/tree/main/src)ディレクトリに移動し，以下を実行する．

```
python advanced_rag.py --kb-id <Knowledge Base の ID> --relevance-eval
```

以下に，`advanced_rag.py`におけるコマンド引数の説明を行う．

| 引数               | 説明                                                          |
| ------------------ | ------------------------------------------------------------- |
| `--kb-id`          | Knowledge Base の ID                                          |
| `--relevance-eval` | 検索結果の関連度評価を行うか否か（`sotre_true`）              |
| `--region`         | リージョン（default: `us-east-1`）                            |
| `--config-path`    | 設定ファイルパス（default: `../config/config_claude-3.yaml`） |

#### advanced_rag.py のアルゴリズム

記事[^0-0]の内容に従い，以下のフローで Advanced RAG を実行している．

```
step1. クエリ拡張
step2. ベクトル検索
step3. ベクトル検索結果の関連度評価
step4. step3で絞り込んだ結果を元に，プロンプト拡張
step5. LLMによるテキスト生成
```

また，各 step で利用している config ファイルは以下の通りである．(LLM を利用する step で config ファイルを用意している．)

| step  | 処理内容       | config ファイル                                                                                                                 |
| ----- | -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| step1 | クエリ拡張     | - `config/query/query.yaml`<br>- `config/llm/claude-3_query_expansion.yaml` <br>- `config/prompt_template/query_expansion.yaml` |
| step2 | ベクトル検索   | -                                                                                                                               |
| step3 | 関連度評価     | - `config/llm/claude-3_relevance_eval.yaml` <br>- `config/prompt_template/relevance_eval.yaml`                                  |
| step4 | プロンプト拡張 | -                                                                                                                               |
| step5 | テキスト生成   | - `config/llm/claude-3_rag.yaml` <br>- `config/config/prompt_template/rag.yaml`                                                 |

以降，各ステップにおける処理と各ステップで利用する cofig ファイルについて説明する．

<details>
  <summary>step1. クエリ拡張</summary>
  単一のクエリを表記揺れや表現などを考慮した複数のクエリに拡張することで，多様な検索結果を取得する．これにより，生成される回答の適合性を高めることを狙いとしている．以下のconfigファイルに検索したい事項やプロンプトテンプレートを定義する．
  
  - `config/query/query.yaml`: 検索したい事項（この内容が拡張される）
  - `config/prompt_template/query_expansion.yaml`: クエリ拡張のためのプロンプトテンプレート
  - `config/llm/claude-3_query_expansion.yaml`: Claude3の設定
  
  ##### `config/query/query.yaml`
  ```yaml
  "query": "What is Amazon doing in the field of generative AI?"
  ```
  
  ##### `config/prompt_template/query_expansion.yaml`
  ```yaml
  retries: 3
  n_queries: 3
  output_format: JSON形式で、各キーには単一のクエリを格納する。
  template: |
    検索エンジンに入力するクエリを最適化し、様々な角度から検索を行うことで、より適切で幅広い検索結果が得られるようにします。
    具体的には、類義語や日本語と英語の表記揺れを考慮し、多角的な視点からクエリを生成します。

    以下の<question>タグ内にはユーザーの入力した質問文が入ります。
    この質問文に基づいて、{n_queries}個の検索用クエリを生成してください。
    各クエリは30トークン以内とし、日本語と英語を適切に混ぜて使用することで、広範囲の文書が取得できるようにしてください。

    生成されたクエリは、<format>タグ内のフォーマットに従って出力してください。

    <example>
    question: Knowledge Bases for Amazon Bedrock ではどのベクトルデータベースを使えますか？
    query_1: Knowledge Bases for Amazon Bedrock vector databases engine DB
    query_2: Amazon Bedrock ナレッジベース ベクトルエンジン vector databases DB
    query_3: Amazon Bedrock RAG 検索拡張生成 埋め込みベクトル データベース エンジン
    </example>

    <format>
    {output_format}
    </format>

    <question>
    {question}
    </question>

````
</details>

- 検索したい事項を[`./config/query/query.yaml`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/config/query/query.yaml)に記載する．
<details>
<summary>query.yamlの中身（例）</summary>
<br/>

```yaml
"query": "What is Amazon doing in the field of generative AI?"
````

  </details>
  <br/>

- プロンプトテンプレートを[`./config/prompt_template/prompt_template.yaml`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/config/prompt_template/prompt_template.yaml)に記載する．

  - 検索したい事項が`{query}`に，Knowledge Base から Retrieve した内容が`{contexts}`に入るように実装している．
  <details>
  <summary>prompt_template.yamlの中身（例）</summary>
  <br/>

  ```yaml
  "template": |
    Human: You are a financial advisor AI system, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {contexts}
    </context>

    <question>
    {query}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:
  ```

  </details>
  <br/>

- LLM の設定（`bedrock_runtime.invoke_model`の引数等）を[`./config/llm`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/config/llm)ディレクトリ内の yaml ファイルに記載する．以下に，Claude3 Opus を利用する場合の例を解説する．

  - 設定ファイルとしては[`./config/llm/claude-3_cofig.yaml`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/config/llm/claude-3_cofig.yaml)を利用する．
  - 引数の他，`stream` 機能を利用するかどうか，`model_id` を記載する．
  - `messages`には{prompt}を含むように記載する
    - 利用する LLM により引数の要素が異なるため，適宜公式ドキュメント[^3]を参照すること．
    - 例えば，Command R+の場合は，Claude3 とは異なり，プロンプトを`message`に記載する．（`messages`ではない）
    <details>
    <summary>claude-3_cofig.yamlの中身（例）</summary>
    <br/>

  ```yaml
  "anthropic_version": "bedrock-2023-05-31"
  "max_tokens": 1000
  "temperature": 0
  "system": "Respond only in Japanese"
  "messages":
    [{ "role": "user", "content": [{ "type": "text", "text": "{prompt}" }] }]
  "stop_sequences": ["</output>"]

  "stream": false
  "model_id": "anthropic.claude-3-opus-20240229-v1:0"
  ```

  </details>
  <br/>
  <details>
  <summary>command-r-plus_config.yamlの中身（例）</summary>
  <br/>

  ```yaml
  "max_tokens": 1000
  "temperature": 0
  "message": "{prompt}"
  "chat_history":
    [
      { "role": "USER", "message": "Respond only in Japanese" },
      {
        "role": "CHATBOT",
        "message": "Sure. What would you like to talk about?",
      },
    ]
  "stop_sequences": ["</output>"]

  "stream": true
  "model_id": "cohere.command-r-plus-v1:0"
  ```

  </details>
  <br/>

- 利用する Knowledge Base の ID を確認する
  <details>
  <summary>Knowledge Base の IDの確認（例）</summary>
  <br/>

  <img src="./assets/kb_id.png" width="800">

  </details>
  <br/>

- [`./src`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/src)ディレクトリに移動し，`python main.py --kb-id <Knowledge Base の ID>`を実行する
  - LLM，Retriever, PromptConfig というクラスを定義している
  - 利用する LLM の設定ファイルは，[`main.py`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/src/main.py)の 35, 36 行目で指定している

### CleanUp（スキップ可能）

[`./notebook/0_create_ingest_documents_test_kb.ipynb`](https://github.com/ren8k/aws-bedrock-rag-baseline/blob/main/notebook/0_create_ingest_documents_test_kb.ipynb)の下部に`CleanUp`セクションがある．セクションのコメントアウトを外し実行することで，ノートブック上部で作成したリソースを全て削除する．

## Next Step

## References

[^0-0]: [Amazon Kendra と Amazon Bedrock で構成した RAG システムに対する Advanced RAG 手法の精度寄与検証](https://aws.amazon.com/jp/blogs/news/verifying-the-accuracy-contribution-of-advanced-rag-methods-on-rag-systems-built-with-amazon-kendra-and-amazon-bedrock/)
[^0-1]: [ren8k/aws-bedrock-rag-baseline](https://github.com/ren8k/aws-bedrock-rag-baseline)
[^2-0]: [Amazon Bedrock の Knowledge Base を Pinecone 無料枠で構築してみた](https://benjamin.co.jp/blog/technologies/bedrock-knowledgeaase-pinecone/)
[^2-1]: [AWS Marketplace の Pinecone を Amazon Bedrock のナレッジベースとして利用する](https://aws.amazon.com/jp/blogs/news/leveraging-pinecone-on-aws-marketplace-as-a-knowledge-base-for-amazon-bedrock/)
