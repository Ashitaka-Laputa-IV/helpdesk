"""taste-rag：LangChain / LangGraph 的 RAG 实验集。

模块分层：
    corpus                     语料目录与切块
    bm25 / vector_store        稀疏索引、向量库
    retrieval                  召回 / 去重 / 融合 / 重排原语
    runtime                    模型与检索栈装配、交互式 CLI
    chat                       基础多轮 chat（含 CLI、持久化历史）
    chat_create_agent          对照：纯 create_agent 版 chat
    chat_manual_graph          对照：手工 StateGraph 版 chat
    rag_common                 共用提示词、状态与节点
    rag_basic                  最简 2-step RAG
    rag_bm25_rerank            多查询改写 + bm25 + 重排
    rag_router                 Router + 双路召回 + 排序（rerank / RRF / 无）
    rag_agent                  Agentic RAG（create_agent + 检索工具）
    summarize                  上下文压缩
    app / client               FastAPI 服务与 HTTP 客户端

这里只做公共导出，方便 `from taste_rag import thread_config` 之类的调用。
"""

from taste_rag.chat import (
    DB_PATH,
    DEFAULT_THREAD,
    DEFAULT_USER,
    TOOLS,
    build_graph,
    get_current_time,
    load_model,
    main,
    silence_beta_warning,
    thread_config,
    thread_key,
)

__all__ = [
    "DB_PATH",
    "DEFAULT_THREAD",
    "DEFAULT_USER",
    "TOOLS",
    "build_graph",
    "get_current_time",
    "load_model",
    "main",
    "silence_beta_warning",
    "thread_config",
    "thread_key",
]
