"""RAG Demo：summarize -> router -> (bm25 ∥ semantic) -> rank -> chatbot。

自包含实现，不依赖项目内其它模块；所有可调参数与提示词都来自 rag-demo.toml。

用法:
    uv run python rag-demo.py
    uv run python rag-demo.py --ranking rerank
"""

from __future__ import annotations

import argparse
import tomllib
import uuid
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal

import bm25s
from langchain.agents.middleware.summarization import count_tokens_approximately
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_reranker.rerankers import init_reranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

RankingMode = Literal["rrf", "rerank", "none"]
RouterMode = Literal["llm", "always"]

SUMMARY_KIND = "summary"
SUMMARY_HEADER = "（以下是更早对话的摘要，不是用户新说的话）\n"


# ---------------------------------------------------------------- 配置


@dataclass(frozen=True)
class ModelSettings:
    name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingSettings:
    name: str
    base_url: str
    api_key: str
    check_ctx_length: bool


@dataclass(frozen=True)
class RerankerSettings:
    name: str
    base_url: str


@dataclass(frozen=True)
class RetrievalSettings:
    docs_dir: Path
    docs_glob: str
    index_dir: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    index_batch_size: int
    bm25_k: int
    semantic_k: int
    rrf_k: int
    top_n: int
    ranking: RankingMode
    router_mode: RouterMode


@dataclass(frozen=True)
class MemorySettings:
    trigger_tokens: int
    keep_messages: int


@dataclass(frozen=True)
class PromptSettings:
    router: str
    summarize: str
    answer: str
    plain: str


@dataclass(frozen=True)
class RuntimeSettings:
    thread_id: str


@dataclass(frozen=True)
class Settings:
    chat: ModelSettings
    router: ModelSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    retrieval: RetrievalSettings
    memory: MemorySettings
    prompts: PromptSettings
    runtime: RuntimeSettings


def _model_settings(raw: dict) -> ModelSettings:
    """把 toml 的模型段落转成 dataclass，max_tokens 非正数表示不限制。"""
    return ModelSettings(
        name=raw["name"],
        base_url=raw["base_url"],
        api_key=raw["api_key"],
        temperature=float(raw["temperature"]),
        max_tokens=int(raw["max_tokens"]),
    )


def load_settings(path: Path) -> Settings:
    """读取配置；配置里的相对路径以文件所在目录为基准。"""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    retrieval = raw["retrieval"]

    return Settings(
        chat=_model_settings(raw["chat"]),
        router=_model_settings(raw["router"]),
        embedding=EmbeddingSettings(**raw["embedding"]),
        reranker=RerankerSettings(**raw["reranker"]),
        retrieval=RetrievalSettings(
            docs_dir=base / retrieval["docs_dir"],
            docs_glob=retrieval["docs_glob"],
            index_dir=base / retrieval["index_dir"],
            collection=retrieval["collection"],
            chunk_size=int(retrieval["chunk_size"]),
            chunk_overlap=int(retrieval["chunk_overlap"]),
            index_batch_size=int(retrieval["index_batch_size"]),
            bm25_k=int(retrieval["bm25_k"]),
            semantic_k=int(retrieval["semantic_k"]),
            rrf_k=int(retrieval["rrf_k"]),
            top_n=int(retrieval["top_n"]),
            ranking=retrieval["ranking"],
            router_mode=retrieval["router_mode"],
        ),
        memory=MemorySettings(**raw["memory"]),
        prompts=PromptSettings(**raw["prompts"]),
        runtime=RuntimeSettings(**raw["runtime"]),
    )


# ---------------------------------------------------------------- 状态


class ChatState(MessagesState):
    """messages 之外，只保留本轮检索的中间结果。"""

    need_retrieval: bool
    bm25_docs: list[dict]
    semantic_docs: list[dict]
    context: str


class RouteDecision(BaseModel):
    """路由输出。"""

    need_retrieval: bool = Field(description="true=需要查阅文档库；false=直接回答")


# ---------------------------------------------------------------- 主体


class RagDemo:
    """检索资源 + 图。命令行与测试共用同一套构建逻辑。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.prompts = settings.prompts

        self._chat_model = self._build_model(settings.chat)
        self._router_model = self._build_model(settings.router)
        self._structured_router = self._router_model.with_structured_output(RouteDecision)

        self._splits = self._split_documents()
        self._chunks = [
            {"source": split.metadata["source"], "text": split.page_content}
            for split in self._splits
        ]

        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding.name,
            base_url=settings.embedding.base_url,
            api_key=settings.embedding.api_key,
            check_embedding_ctx_length=settings.embedding.check_ctx_length,
        )
        self._store = Chroma(
            collection_name=settings.retrieval.collection,
            embedding_function=self._embeddings,
            persist_directory=str(settings.retrieval.index_dir),
        )
        self._ensure_index()
        self._retriever = self._store.as_retriever(
            search_kwargs={"k": settings.retrieval.semantic_k}
        )

        self._bm25 = self._build_bm25(self._chunks)
        self._reranker = init_reranker(
            settings.reranker.name, base_url=settings.reranker.base_url
        )

        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph(self._checkpointer)

    # -- 资源

    @staticmethod
    def _build_model(options: ModelSettings):
        """按配置初始化聊天模型；max_tokens<=0 视为不限制。"""
        return init_chat_model(
            options.name,
            base_url=options.base_url,
            api_key=options.api_key,
            temperature=options.temperature,
            max_tokens=options.max_tokens if options.max_tokens > 0 else None,
        )

    def _split_documents(self) -> list[Document]:
        """读语料并切分，向量库与 bm25 共用同一批分块。"""
        retrieval = self.settings.retrieval
        documents = [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path.relative_to(retrieval.docs_dir))},
            )
            for path in sorted(retrieval.docs_dir.glob(retrieval.docs_glob))
        ]
        return RecursiveCharacterTextSplitter(
            chunk_size=retrieval.chunk_size, chunk_overlap=retrieval.chunk_overlap
        ).split_documents(documents)

    def _ensure_index(self) -> None:
        """向量库为空时才写入，避免每次启动重复嵌入。"""
        if self._store.get(limit=1)["ids"]:
            return
        batch = self.settings.retrieval.index_batch_size
        for start in range(0, len(self._splits), batch):
            self._store.add_documents(self._splits[start : start + batch])
        print(f"[index] 已写入 {len(self._splits)} 个分块")

    @staticmethod
    def _build_bm25(chunks: list[dict]):
        """对分块建 bm25 索引。"""
        tokens = bm25s.tokenize([chunk["text"] for chunk in chunks], show_progress=False)
        index = bm25s.BM25()
        index.index(tokens, show_progress=False)
        return index

    # -- 检索

    def _bm25_search(self, query: str) -> list[dict]:
        tokens = bm25s.tokenize(query, show_progress=False)
        indices, _ = self._bm25.retrieve(
            tokens, k=self.settings.retrieval.bm25_k, show_progress=False
        )
        return [self._chunks[index] for index in indices[0]]

    def _semantic_search(self, query: str) -> list[dict]:
        return [
            {"source": doc.metadata["source"], "text": doc.page_content}
            for doc in self._retriever.invoke(query)
        ]

    def _dedupe(self, docs: list[dict], top_n: int) -> list[dict]:
        """按 (来源, 文本) 去重并截断。"""
        picked: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for doc in docs:
            key = (doc["source"], doc["text"])
            if key in seen:
                continue
            seen.add(key)
            picked.append(doc)
            if len(picked) == top_n:
                break
        return picked

    def _fuse_rrf(self, bm25_docs: list[dict], semantic_docs: list[dict]) -> list[dict]:
        """名次倒数融合；同分时保持首次出现顺序（bm25 在前）。"""
        k = self.settings.retrieval.rrf_k
        scores: dict[tuple[str, str], float] = {}
        first: dict[tuple[str, str], dict] = {}
        for ranked in (bm25_docs, semantic_docs):
            for rank, doc in enumerate(ranked, 1):
                key = (doc["source"], doc["text"])
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                first.setdefault(key, doc)
        ordered = sorted(scores.items(), key=lambda item: -item[1])
        return [first[key] for key, _ in ordered[: self.settings.retrieval.top_n]]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """交叉编码器重排。"""
        documents = [
            Document(page_content=doc["text"], metadata={"source": doc["source"]})
            for doc in candidates
        ]
        ranked = self._reranker.compress_documents(documents, query=query)
        return [
            {"source": doc.metadata["source"], "text": doc.page_content}
            for doc in ranked[: self.settings.retrieval.top_n]
        ]

    # -- 节点

    def _summarize(self, state: ChatState) -> dict:
        """历史超预算时，把较早对话压成一条摘要消息。"""
        messages = state["messages"]
        if count_tokens_approximately(messages) <= self.settings.memory.trigger_tokens:
            return {}

        summaries = [m for m in messages if m.additional_kwargs.get("kind") == SUMMARY_KIND]
        plain = [m for m in messages if m.additional_kwargs.get("kind") != SUMMARY_KIND]
        cut = max(0, len(plain) - self.settings.memory.keep_messages)
        # 不把 AI 的 tool_calls 与它的 ToolMessage 拆开
        while 0 < cut < len(plain) and plain[cut].type == "tool":
            cut -= 1
        if cut <= 0:
            return {}

        old, remaining = plain[:cut], plain[cut:]
        previous = "\n".join(m.text for m in summaries if m.text)
        prompt = self.prompts.summarize.format(
            previous=previous or "（无）",
            dialogue="\n".join(f"{m.type}: {m.text}" for m in old if m.text),
        )
        try:
            summary = (self._chat_model.invoke(prompt).text or "").strip()
        except Exception:  # noqa: BLE001 - 压缩失败不应阻塞对话
            return {}
        if not summary:
            return {}

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(
                    content=f"{SUMMARY_HEADER}{summary}",
                    additional_kwargs={"kind": SUMMARY_KIND},
                ),
                *remaining,
            ]
        }

    def _route(self, state: ChatState) -> dict:
        """决定是否检索：always 一律查，llm 交给模型判断。"""
        if self.settings.retrieval.router_mode == "always":
            need = True
        else:
            try:
                decision = self._structured_router.invoke(
                    [SystemMessage(content=self.prompts.router), *state["messages"]]
                )
                need = decision.need_retrieval
            except Exception:  # noqa: BLE001 - 路由失败时保守检索，避免漏查
                need = True
        return {"need_retrieval": need, "context": ""}

    def _bm25_node(self, state: ChatState) -> dict:
        return {"bm25_docs": self._bm25_search(state["messages"][-1].text)}

    def _semantic_node(self, state: ChatState) -> dict:
        return {"semantic_docs": self._semantic_search(state["messages"][-1].text)}

    def _rank(self, state: ChatState) -> dict:
        """按配置的排序策略挑出进入上下文的文档。"""
        query = state["messages"][-1].text
        bm25_docs = state.get("bm25_docs") or []
        semantic_docs = state.get("semantic_docs") or []
        candidates = self._dedupe(bm25_docs + semantic_docs, len(bm25_docs) + len(semantic_docs))

        mode = self.settings.retrieval.ranking
        if mode == "rerank":
            picked = self._rerank(query, candidates)
        elif mode == "none":
            picked = candidates[: self.settings.retrieval.top_n]
        else:
            picked = self._fuse_rrf(bm25_docs, semantic_docs)

        return {"context": "\n\n".join(f"[{d['source']}] {d['text']}" for d in picked)}

    def _answer(self, state: ChatState) -> dict:
        """有上下文时按资料回答，否则走无资料提示词。"""
        context = state.get("context") or ""
        system = (
            self.prompts.answer.format(context=context) if context else self.prompts.plain
        )
        reply = self._chat_model.invoke([SystemMessage(content=system), *state["messages"]])
        return {"messages": [reply]}

    # -- 图

    @staticmethod
    def _next_nodes(state: ChatState) -> list[str] | str:
        return ["bm25", "semantic"] if state["need_retrieval"] else "chatbot"

    def _build_graph(self, checkpointer):
        builder = StateGraph(ChatState)
        builder.add_node("summarize", self._summarize)
        builder.add_node("router", self._route)
        builder.add_node("bm25", self._bm25_node)
        builder.add_node("semantic", self._semantic_node)
        builder.add_node("rank", self._rank)
        builder.add_node("chatbot", self._answer)

        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "router")
        builder.add_conditional_edges("router", self._next_nodes)
        builder.add_edge("bm25", "rank")
        builder.add_edge("semantic", "rank")
        builder.add_edge("rank", "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile(checkpointer=checkpointer)

    # -- 对外接口

    def invoke(self, question: str, thread_id: str | None = None) -> dict:
        """跑完一轮，返回最终状态。"""
        config = {"configurable": {"thread_id": thread_id or self.settings.runtime.thread_id}}
        return self._graph.invoke({"messages": [HumanMessage(content=question)]}, config)

    def state(self, thread_id: str | None = None) -> dict:
        """读取某个会话的当前状态（供命令行调试与测试使用）。"""
        config = {"configurable": {"thread_id": thread_id or self.settings.runtime.thread_id}}
        return self._graph.get_state(config).values

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[str]:
        """流式产出回答片段，只取 chatbot 节点的 token。"""
        config = {"configurable": {"thread_id": thread_id or self.settings.runtime.thread_id}}
        for message, metadata in self._graph.stream(
            {"messages": [HumanMessage(content=question)]}, config, stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "chatbot" and message.text:
                yield message.text


def silence_serializer_warnings() -> None:
    """流式输出 + with_structured_output 时，parsed 字段会触发 Pydantic 序列化告警。"""
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def main() -> None:
    silence_serializer_warnings()
    parser = argparse.ArgumentParser(description="RAG Demo")
    parser.add_argument("--config", default="rag-demo.toml", help="配置文件路径")
    parser.add_argument("--ranking", choices=["rrf", "rerank", "none"], help="覆盖排序策略")
    parser.add_argument("--router-mode", choices=["llm", "always"], help="覆盖路由模式")
    args = parser.parse_args()

    settings = load_settings(Path(args.config).resolve())
    if args.ranking or args.router_mode:
        settings = replace(
            settings,
            retrieval=replace(
                settings.retrieval,
                ranking=args.ranking or settings.retrieval.ranking,
                router_mode=args.router_mode or settings.retrieval.router_mode,
            ),
        )

    demo = RagDemo(settings)
    thread_id = f"{settings.runtime.thread_id}-{uuid.uuid4().hex[:8]}"

    print("输入问题开始问答，输入 exit / quit 结束。")
    while True:
        try:
            question = input("Q: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        print("A: ", end="", flush=True)
        for token in demo.stream(question, thread_id):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
