"""最简 chat 图 + 工具调用：使用 langgraph + langchain。

图结构：START -> chatbot -> (tools -> chatbot)* -> END
chatbot 节点调用模型；当模型请求调用工具时，先由 tools 节点执行，再回到 chatbot。
对话历史通过 SqliteSaver checkpointer 持久化到 .chat.db，并按 (user_id, thread_id) 隔离。
"""

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

DB_PATH = Path(__file__).resolve().parents[2] / ".chat.db"
DEFAULT_USER = "default"
DEFAULT_THREAD = "default"


def thread_key(user_id: str, thread_id: str) -> str:
    """拼出 checkpointer 的 thread_id，保证不同用户/会话互不串线。"""
    return f"{user_id}:{thread_id}"


def thread_config(user_id: str, thread_id: str) -> dict:
    """构造传给图的 config。"""
    return {"configurable": {"thread_id": thread_key(user_id, thread_id)}}


def silence_beta_warning() -> None:
    """v3 流式协议目前是实验性的，屏蔽其 beta 警告避免污染输出。"""
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)


@tool
def get_current_time() -> str:
    """获取当前的日期与时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_current_time]


def load_model():
    """从环境变量初始化对话模型。"""
    load_dotenv()
    return init_chat_model(os.environ["MODEL"])


def build_graph(model, checkpointer=None):
    """构建并编译 chat 图。"""
    model_with_tools = model.bind_tools(TOOLS)

    def chatbot(state: MessagesState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
    builder.add_edge("tools", "chatbot")
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(description="命令行 chat")
    parser.add_argument("--user", default=DEFAULT_USER, help="用户标识")
    parser.add_argument("--thread", default=DEFAULT_THREAD, help="会话标识")
    args = parser.parse_args()

    silence_beta_warning()
    model = load_model()
    config = thread_config(args.user, args.thread)

    # 历史存 .chat.db，按 (user, thread) 隔离；只传新消息，LangGraph 自动接上历史
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(model, checkpointer)
        print(f"[{args.user}/{args.thread}] 输入内容开始对话，输入 exit / quit 结束。")
        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            print("AI: ", end="", flush=True)
            stream = graph.stream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                version="v3",
            )
            # 投影消费：messages 投影按节点给出逐 token 文本
            for _, item in stream.interleave("messages"):
                if item.node == "chatbot":
                    for token in item.text:
                        print(token, end="", flush=True)
            print()


if __name__ == '__main__':
    main()
