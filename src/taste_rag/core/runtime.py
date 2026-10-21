"""运行时装配：模型、嵌入、重排、检索栈，以及交互式 CLI。

各 rag_* 的 main() 只声明自己需要什么，不再重复写环境准备与流式输出样板。
"""

import os
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core._api import LangChainBetaWarning
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker

from taste_rag.core.corpus import load_chunks
from taste_rag.core.errors import explain_error
from taste_rag.core.vector_store import open_store

DEFAULT_ROUTER_MAX_TOKENS = 128  # 留出余量：历史较长时模型偶尔会多写几个 token
DEFAULT_ROUTER_MODE = "llm"


@dataclass
class RetrievalStack:
    """一套可直接传给图的检索组件。"""

    bm25: Any
    chunks: list
    retriever: Any
    reranker: Any | None = None


def silence_beta_warning() -> None:
    """v3 流式协议目前是实验性的，屏蔽其 beta 警告避免污染输出。"""
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def silence_serializer_warning() -> None:
    """屏蔽结构化输出的噪声。

    with_structured_output 把解析结果放进消息的 parsed 字段，LangGraph 在
    checkpoint/事件序列化时会报 PydanticSerializationUnexpectedValue。
    解析结果本身是正确的，只是序列化器不认识这个类型。
    """
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def silence_warnings() -> None:
    """屏蔽 beta 流式协议告警与结构化输出的序列化告警。"""
    silence_beta_warning()
    silence_serializer_warning()


def load_model():
    """对话模型。"""
    load_dotenv()  # 各入口都从这里取模型，统一在这里读 .env
    return init_chat_model(os.environ["MODEL"])


def load_router_model(max_tokens: int = DEFAULT_ROUTER_MAX_TOKENS):
    """路由器模型：可指向另一个服务（例如非思考的小模型），默认复用对话模型。"""
    load_dotenv()
    return init_chat_model(
        os.environ.get("ROUTER_MODEL", os.environ["MODEL"]),
        base_url=os.environ.get("ROUTER_BASE_URL", os.environ["OPENAI_API_BASE"]),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=max_tokens,
    )


def load_embeddings() -> OpenAIEmbeddings:
    """嵌入模型。"""
    load_dotenv()
    return OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        check_embedding_ctx_length=False,
    )


def load_reranker():
    """重排模型。"""
    load_dotenv()
    return init_reranker(
        os.environ["RERANKER_MODEL"], base_url=os.environ["RERANKER_BASE_URL"]
    )


def build_bm25() -> tuple[Any, list]:
    """加载语料并建 bm25 索引，返回 (索引, 分块)。"""
    # 延迟导入：bm25s 只有在真正检索时才需要，也避免 `-m taste_rag.core.bm25` 重复加载
    from taste_rag.core.bm25 import build_index

    chunks = load_chunks()
    print(f"已索引 {len(chunks)} 个分块")
    return build_index(chunks), chunks


def build_retrieval_stack(
    semantic_k: int, with_reranker: bool = False
) -> RetrievalStack:
    """建 bm25 索引、连向量库，按需加载重排器。"""
    bm25, chunks = build_bm25()
    retriever = open_store(load_embeddings()).as_retriever(
        search_kwargs={"k": semantic_k}
    )
    return RetrievalStack(
        bm25=bm25,
        chunks=chunks,
        retriever=retriever,
        reranker=load_reranker() if with_reranker else None,
    )


def stream_answer(
    graph, config: dict, question: str, node: str = "chatbot"
) -> Iterator[str]:
    """跑一轮图，只产出指定节点（默认 chatbot）的 token。"""
    stream = graph.stream_events(
        {"messages": [HumanMessage(content=question)]}, config, version="v3"
    )
    for _, item in stream.interleave("messages"):
        if item.node == node:
            yield from item.text


def run_cli(
    answer: Callable[[str], Iterator[str]],
    prompt: str = "Q: ",
    reply_prefix: str = "A: ",
    intro: str = "输入问题开始问答，输入 exit / quit 结束。",
) -> None:
    """交互式问答：输入问题，逐 token 打印回答。

    单轮失败只打印一句提示并继续，不把 traceback 甩给用户。
    """
    print(intro)
    while True:
        try:
            question = input(prompt)
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break

        # 前缀延迟到首个 token 再打印，否则会被节点日志插在中间
        started = False
        try:
            for token in answer(question):
                if not started:
                    print(reply_prefix, end="", flush=True)
                    started = True
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            print("\n[中断] 已停止本轮输出。")
            continue
        except Exception as error:  # noqa: BLE001 - 单轮失败不应结束会话
            if started:
                print()
            print(f"[错误] {explain_error(error)}")
            continue
        print()
