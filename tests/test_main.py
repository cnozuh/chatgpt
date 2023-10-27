import asyncio
import csv

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import main


@pytest.fixture
def create_text_data(tmpdir):
    """
    一時的なCSVファイルを作成し、そのファイルのパスを返します。
    :param tmpdir:テスト用の一時ディレクトリ
    :yield: 一時CSVファイルのパス
    """
    example_words = [
        ['test_words'],
        ['test1'],
        ['test2'],
        ['test3']
    ]
    tmpfile = tmpdir.join('words.csv')
    with tmpfile.open('w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(example_words)
    yield str(tmpfile)
    tmpdir.remove()


def test_get_data(create_text_data):
    """
    get_data関数がCSVファイルから正しくデータを読み取るかテストします。
    :param create_text_data: create_text_dataフィクスチャから取得したファイルのパス
    """
    expected_words = ['test1', 'test2', 'test3']
    assert main.get_data(create_text_data) == expected_words


@pytest.fixture
def create_template_prompt(tmpdir):
    """
    一時的なテキストファイルを作成し、そのファイルのパスを返します。
    :param tmpdir:テスト用の一時ディレクトリ
    :yield: 一時テキストファイルのパス
    """
    example_text = '文章校正をしてください。'
    tmpfile = tmpdir.join('template.txt')
    with tmpfile.open('w') as txtfile:
        txtfile.write(example_text)
    yield str(tmpfile)
    tmpfile.remove()


def test_get_template(create_template_prompt):
    """
    get_template関数がテキストファイルから正しくデータを読み取るかテストします。
    :param create_template_prompt: create_template_promptフィクスチャから取得したファイルのパス
    """
    expected_text = '文章校正をしてください。'
    assert main.get_template(create_template_prompt) == expected_text


def test_configure_schema():
    """
    configure_schema関数がスキーマを正しく設定するかテストします。
    """
    schema = main.configure_schema()
    assert schema is not None


def test_configure_prompt():
    """
    configure_prompt関数がプロンプトを正しく設定するかテストします。
    """
    system_template = '{system}'
    prompt = main.configure_prompt(system_template)
    assert prompt is not None


@pytest.mark.asyncio
async def test_send_to_chatllm_success():
    """
    send_to_chatllm関数がmockを使用して正しく動作するかテストします。
    """
    with patch('main.ChatOpenAI') as MockChatOpenAI, \
         patch('main.LLMChain') as MockLLMChain:

        mock_output_parser = MagicMock()
        mock_output_parser.parse.return_value = {'output': ['corrected text']}
        mock_model_instance = MockChatOpenAI.return_value
        mock_model_instance.temperature = 0.7
        mock_chain_instance = MockLLMChain.return_value
        # AsyncMockを使用して、非同期のmock関数を返すようにします
        mock_chain_instance.arun = AsyncMock(return_value='mock output')
        result = await main.send_to_chatllm(
            mock_model_instance, mock_chain_instance, ['text'], mock_output_parser
        )
        assert result == ['corrected text']


@pytest.mark.asyncio
async def test_generate_concurrently():
    """
    generate_concurrently関数が入力テキストを正しくバッチに分割し、
    それぞれのバッチでコルーチンを生成するかテストします。
    """
    with patch('main.send_to_chatllm', return_value=['mocked result']):
        input_texts = ['test1', 'test2', 'test3']
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_output_parser = MagicMock()
        generator = main.generate_concurrently(
            mock_model, mock_chain, input_texts, mock_output_parser, 2
        )
        tasks = list(generator)
        # 2つのバッチ（2つのテキストと1つのテキスト）でコルーチンが生成される
        assert len(tasks) == 2

        results = await asyncio.gather(*tasks)
        assert results == [['mocked result'], ['mocked result']]
