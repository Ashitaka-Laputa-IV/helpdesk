"""最简单的 agentic-chat：create_agent + 内存 checkpointer。

对话历史只活在当前进程里（InMemorySaver），退出即丢，不落盘。
图、工具循环、消息合并都由 create_agent 负责，这里只做命令行外壳。

用法:
    uv run python src/helpdesk/agentic_chat.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from helpdesk import TOOLS, silence_beta_warning

SYSTEM_PROMPT = "你是客服助手，回答要简洁；需要当前时间时调用工具获取，不要自己猜。"
THREAD_ID = "default"


def main() -> None:
    silence_beta_warning()
    load_dotenv()

    agent = create_agent(
        init_chat_model(os.environ["MODEL"]),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        # 内存 checkpointer：维持多轮历史，但进程结束就没了
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("输入内容开始对话，输入 exit / quit 结束。")
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]}, config
        )
        print(f"AI: {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
