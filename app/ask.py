"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"



def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())

    # 1. 履歴ファイルの場所を決める
    history_file = Path("/workspaces/lecture-ai-engineering-2026/lectures/01/chat_history.json")

    # 2. 保存済みの履歴を読み込む
    if history_file.exists():
        with history_file.open("r", encoding="utf-8") as f:
            saved_history = json.load(f)

        history = [
            types.Content.model_validate(item)
            for item in saved_history
        ]
    else:
        history = []

    # 3. 過去の会話を持つチャットを作る
    chat = client.chats.create(
        model=MODEL,
        history=history,
    )

    # 4. 質問を送り、回答を受け取る
    response = chat.send_message(question)

    # 5. トークン数を表示する
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[トークン] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )

    # 6. 今回の会話をファイルに保存する
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with history_file.open("w", encoding="utf-8") as f:
        json.dump(
            [
                item.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for item in chat.get_history()
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    return response.text



def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
