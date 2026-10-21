"""最简 chat 图 + 工具调用：使用 langgraph + langchain。

图结构：START -> chatbot -> (tools -> chatbot)* -> END
chatbot 节点调用模型；当模型请求调用工具时，先由 tools 节点执行，再回到 chatbot。
对话历史通过 SqliteSaver checkpointer 持久化到 .db/chat.db，并按 (user_id, thread_id) 隔离。

通用的东西都在 core：工具在 core.tools，会话与模型装配在 core.session / core.runtime。
"""

import argparse
from typing import Literal

from langchain.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from taste_rag.core import runtime
from taste_rag.core.paths import db_path
from taste_rag.core.session import DEFAULT_THREAD, DEFAULT_USER, thread_config
from taste_rag.core.tools import TOOLS

DB_PATH = db_path("chat.db")


def build_graph(model, checkpointer=None, system_prompt: str | None = None):
    """构建并编译 chat 图。

    system_prompt 为 None 时不插入 system 消息（保持最简）；
    需要固定人设时传入，节点会在每轮对话前补上。
    """
    model_with_tools = model.bind_tools(TOOLS)

    def chatbot(state: MessagesState) -> dict:
        messages = state["messages"]
        if system_prompt:
            messages = [SystemMessage(content=system_prompt), *messages]
        return {"messages": [model_with_tools.invoke(messages)]}

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

    runtime.silence_warnings()
    model = runtime.load_model()
    config = thread_config(args.user, args.thread)

    # 历史存 .db/chat.db，按 (user, thread) 隔离；只传新消息，LangGraph 自动接上历史
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(model, checkpointer)
        runtime.run_cli(
            lambda question: runtime.stream_answer(graph, config, question),
            prompt="You: ",
            reply_prefix="AI: ",
            intro=f"[{args.user}/{args.thread}] 输入内容开始对话，输入 exit / quit 结束。",
        )


if __name__ == '__main__':
    main()
