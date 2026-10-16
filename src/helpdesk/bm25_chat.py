"""用 langgraph 编排的问答 chat：检索走 bm25s，答案由模型基于检索到的语料生成。

图结构：START -> retrieve -> chatbot -> END
retrieve 节点用 bm25s 取回 top-k 片段写入 context，chatbot 节点据此作答。

用法:
    uv run python -m helpdesk.bm25_chat
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from .bm25 import build_index, load_chunks, search

DB_PATH = Path(__file__).resolve().parents[2] / ".bm25_chat.db"
THREAD_ID = "default"
TOP_K = 5
SYSTEM_PROMPT = "你是客服助手，只依据给定资料回答问题；资料中没有的信息，直接说明不知道。"


class ChatState(MessagesState):
    """messages 记录对话；context 存放本轮检索到的资料。"""

    context: str


def build_graph(model, retriever, chunks, checkpointer=None):
    """构建并编译 chat 图。"""

    def retrieve(state: ChatState) -> dict:
        question = state["messages"][-1].text
        hits = search(retriever, chunks, question, TOP_K)
        return {"context": "\n\n".join(f"[{source}] {text}" for _, source, text in hits)}

    def chatbot(state: ChatState) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\n资料：\n{state['context']}"
        reply = model.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(ChatState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    # v3 流式协议目前是实验性的，屏蔽其 beta 警告避免污染 CLI 输出
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)

    load_dotenv()
    chunks = load_chunks()
    retriever = build_index(chunks)
    print(f"已索引 {len(chunks)} 个分块")

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(
            init_chat_model(os.environ["MODEL"]), retriever, chunks, checkpointer
        )
        config = {"configurable": {"thread_id": THREAD_ID}}

        print("输入问题开始问答，输入 exit / quit 结束。")
        while True:
            try:
                question = input("Q: ")
            except (EOFError, KeyboardInterrupt):
                break
            if question.strip().lower() in {"exit", "quit"}:
                break

            print("A: ", end="", flush=True)
            stream = graph.stream_events(
                {"messages": [HumanMessage(content=question)]},
                config,
                version="v3",
            )
            for _, item in stream.interleave("messages"):
                if item.node == "chatbot":
                    for token in item.text:
                        print(token, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
