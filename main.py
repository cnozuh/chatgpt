import asyncio
import argparse
import sys
import itertools
import logging
import os
from pathlib import Path

import pandas as pd

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import OutputParserException
from langchain.output_parsers import (
    ResponseSchema,
    StructuredOutputParser,
    OutputFixingParser
)
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

HUMAN_TEMPLATE= '{input_data}'
"""AIシステムと対話する人間からの情報を表すチャットメッセージ"""

logger = logging.getLogger(__name__)


def configure_logging(logging_level=logging.INFO):
    """
    システムのロギングを設定します
    :param logging_level: ログレベルの定義
    """
    root = logging.getLogger()
    root.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    root.addHandler(handler)


def get_data(file_path):
    """
    読み込んだファイルのテキストをリストとして出力します
    :param file_path: テキスト一覧が保存されているファイルパス
    :return: 読み込んだファイルのテキストをリストとして出力
    """
    df = pd.read_csv(file_path)
    input_text = df.iloc[:,0].to_list()
    return input_text


def get_template(template_path):
    """
    プロンプトのテンプレートファイルを読み込みます
    :param template_path: プロンプトテンプレートが保存されているファイルのパス
    :return: プロンプトのテンプレートが記述されているテキストを出力
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return template


def get_model(model_name, chat_prompt, temperature):
    """
    チャットモデルのラッパーを初期化します
    :param model_name: 利用したいチャットモデルの名前
    :param chat_prompt: llmに送信されるプロンプトテンプレート
    :param temperature: LLMの応答のランダム性を制御
    :return: 定義されたチャットモデル
    """
    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    chain = LLMChain(llm=model, prompt=chat_prompt)
    return model, chain


def configure_schema():
    """
    受信したい応答スキーマを定義します
    :retuen: 定義されたスキーマから作成されたパーサーを出力
    """
    response_schemas = [
        ResponseSchema(name='output', type='list[str]',
            description='A list that holds the output results.'),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser


def configure_prompt(system_template, human_template=HUMAN_TEMPLATE):
    """
    チャットAPIで使用するための構造化プロンプトを生成します
    :param system_template: AIシステムに指示したい情報
    :param human_template: ユーザーの入力情報
    :return: llmに送信されるプロンプトテンプレート
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt


async def send_to_chatllm(model, chain, input_text, output_parser):
    """
    LLMへ送信したプロンプトから生成結果を受信し、出力から構造化された応答を抽出します
    出力構造が定義した形式と異なる場合は、再度LLMに出力の修正を依頼します
    :param model: 初期化されたLLMラッパー
    :param chain :LLMラッパーとプロンプトテンプレートから構成されたLLMチェーン
    :param input_text: 校正したいテキスト一覧
    :param output_parser: 定義されたスキーマから作成されたパーサー
    :return: LLM出力から構造化された応答を抽出されたテキスト一覧
    """
    output = await chain.arun(
        format_instructions=output_parser.get_format_instructions(),
        input_data=input_text,
    )
    try:
        result = output_parser.parse(output)
    except OutputParserException:
        fix_parser = OutputFixingParser.from_llm(parser=output_parser, llm=model)
        result = fix_parser.parse(output)
    return result['output']


def generate_concurrently(model, chain, input_text, output_parser, batch):
    """
    入力テキストをバッチに分割して複数のコルーチンを生成し、awaitableのリストを作成します
    :param model: 初期化されたLLMラッパー
    :param chain: LLMラッパーとプロンプトテンプレートから構成されたLLMチェーン
    :param input_text: 校正したいテキスト一覧
    :param output_parser: 定義されたスキーマから作成されたパーサー
    :param batch: LLMへ送信する単語数を制御するバッチ数
    :yield: 複数のコルーチンのジェネレータを出力
    """
    for num in range(0, len(input_text), batch):
        split_input = input_text[num: num+batch]
        yield send_to_chatllm(model, chain, split_input, output_parser)


def export_results(input_text, output_text, output_path):
    """
    入出力テキストデータを含む表をCSVファイルとして指定パスに保存します
    :param input_text: 校正したいテキスト一覧
    :param output_text: 校正されたテキスト一覧
    :param output_path: 出力結果を保存するパス
    """
    if not output_path.exists():
        output_path.mkdir()
    results_path = output_path / 'results.csv'
    results = list(zip(input_text, output_text))
    df = pd.DataFrame(results, columns=['input_words', 'output_words'])
    df.to_csv(results_path, index=False)


async def main(args):
    """
    テキスト一覧と条件定義したプロンプトをLLMAPIへ送信し、応答された結果をCSVファイルとして出力します
    :param args: コマンドライン引数
    """
    os.environ["OPENAI_API_KEY"] = args.api_key
    input_text = get_data(args.data_path)
    system_template = get_template(args.template_prompt_path)
    output_parser = configure_schema()
    chat_prompt = configure_prompt(system_template)
    model, chain = get_model(args.model, chat_prompt, args.temperature)
    tasks = list(generate_concurrently(model,
                                       chain,
                                       input_text,
                                       output_parser,
                                       args.batch_size))
    results = await asyncio.gather(*tasks)
    results = list(itertools.chain.from_iterable(results))
    export_results(input_text, results, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str,
                        help='OpenAIのAPIキー')
    parser.add_argument('--data_path',  type=Path, default='data/test_words.csv',
                        help='校正対象のテキストファイルのパス')
    parser.add_argument('--template_prompt_path',  type=Path, default='etc/template_prompt/chatmodel_template.txt',
                        help='LLMに入力するプロンプトテンプレートが記述されているテキストファイルのパス')
    parser.add_argument('--output', type=Path, default='output',
                        help='出力結果を格納するディレクトリ名')
    parser.add_argument('--model', default='gpt-3.5-turbo', type=str,
                        help='利用するLLMモデル名')
    parser.add_argument('--batch_size', default=50, type=int,
                        help='LLMへ送信する単語数を制御するサイズ')
    parser.add_argument('--temperature', default=0, type=int,
                        help='LLMの応答のランダム性を制御')
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main(args))
