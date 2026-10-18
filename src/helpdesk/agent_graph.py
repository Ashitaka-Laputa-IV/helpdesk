"""create_agent 与 graph 编排结合的最简实现。

图：START -> agent -> END，其中 agent 节点就是一个 create_agent 产物。
对话历史只存内存（InMemorySaver），退出即丢，不落盘。

用法:
    uv run python src/helpdesk/agent_graph.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from helpdesk import TOOLS, silence_beta_warning

SYSTEM_PROMPT = "你是客服助手，回答要简洁；需要当前时间时调用工具获取，不要自己猜。"
THREAD_ID = "default"


def build_graph(model):
    """把 create_agent 产出的 agent 当成图节点，外面套一层 StateGraph。"""
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    silence_beta_warning()
    load_dotenv()

    graph = build_graph(init_chat_model(os.environ["MODEL"]))
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("输入内容开始对话，输入 exit / quit 结束。")
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break

        result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
        print(f"AI: {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
