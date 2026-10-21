"""最简 CLI 客户端，连 taste_rag 的 FastAPI 服务。

先用 `uv run uvicorn taste_rag.app:app` 起服务，然后:

    uv run python src/taste_rag/client.py --user u1 --thread t1   # 交互模式
    uv run python src/taste_rag/client.py "现在几点？"            # 单次提问
"""

import argparse
import json

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def ask(client: httpx.Client, base_url: str, user_id: str, thread_id: str, message: str) -> None:
    """发一条消息，把 SSE 流里的 token 实时打印出来。"""
    payload = {"user_id": user_id, "thread_id": thread_id, "message": message}
    try:
        with client.stream("POST", f"{base_url}/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "token":
                    print(event["text"], end="", flush=True)
    except httpx.HTTPError as exc:
        print(f"\n请求失败: {exc}")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="taste-rag chat 客户端")
    parser.add_argument("message", nargs="?", help="单次提问；不传则进入交互模式")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="服务地址")
    parser.add_argument("--user", default="default", help="用户标识")
    parser.add_argument("--thread", default="default", help="会话标识")
    args = parser.parse_args()

    with httpx.Client(timeout=None) as client:
        if args.message:
            ask(client, args.base_url, args.user, args.thread, args.message)
            return

        print(f"[{args.user}/{args.thread}] 输入内容开始对话，exit / quit 结束。")
        while True:
            try:
                message = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if message.strip().lower() in {"exit", "quit"}:
                break
            if message.strip():
                print("AI: ", end="", flush=True)
                ask(client, args.base_url, args.user, args.thread, message)


if __name__ == "__main__":
    main()
