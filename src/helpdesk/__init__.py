"""最简 chat 图 + 工具调用：使用 langgraph + langchain。

图结构：START -> chatbot -> (tools -> chatbot)* -> END
chatbot 节点调用模型；当模型请求调用工具时，先由 tools 节点执行，再回到 chatbot。
"""

import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


@tool
def get_current_time() -> str:
    """获取本机当前的日期与时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_current_time]


def build_graph(model):
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
    return builder.compile()


def main() -> None:
    load_dotenv()
    model = init_chat_model(os.environ["MODEL"])
    graph = build_graph(model)

    messages = []
    print("输入内容开始对话，输入 exit / quit 结束。")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        messages.append(HumanMessage(content=user_input))
        messages = graph.invoke({"messages": messages})["messages"]
        print(f"AI: {messages[-1].text}")


if __name__ == '__main__':
    main()
