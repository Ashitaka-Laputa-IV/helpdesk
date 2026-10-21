# taste-rag

基于 **LangChain 1.4 / LangGraph** 的本地 RAG 实验项目：同一套语料与模型下，对比「固定工作流」与「Agentic」两条问答路线，并附带多轮 chat 与 HTTP 服务。

- **workflow**：固定编排 —— Router 判断是否检索 → bm25 + 向量双路召回 → 排序（rerank / RRF / 无）→ 生成
- **agentic**：流程交给模型 —— 只提供一个检索工具，何时查、查几次由 `create_agent` 自主决定

两条路线共用同一套检索原语（`core/retrieval.py`）与上下文压缩阈值。

## 特性

| 能力 | 说明 |
|---|---|
| 双路召回 | bm25s 稀疏 + Chroma 向量，按 `(来源, 原文)` 去重 |
| 三种排序 | `rerank`（cross-encoder）/ `rrf`（名次倒数融合）/ `none`（沿用召回顺序） |
| 检索判断 | `router_mode=llm` 让模型判断；`always` 一律检索，用于对照 |
| 多轮历史 | SqliteSaver，按 `(user, thread)` 隔离 |
| 上下文压缩 | workflow 用 `core/summarize.py`；agentic 用官方 `SummarizationMiddleware` |
| 流式输出 | CLI 逐 token 打印；HTTP 走 SSE |
| 错误处理 | 服务未启动只提示一句，单轮失败不退出会话 |

## 目录结构

```
src/taste_rag/
├── core/                 基础设施
│   ├── corpus.py         语料目录与切块
│   ├── bm25.py           bm25s 索引与查询
│   ├── vector_store.py   Chroma 开库与建索引
│   ├── retrieval.py      召回 / 去重 / rerank / RRF / 拼接上下文
│   ├── summarize.py      上下文压缩（StateGraph 版）
│   ├── session.py        thread_id 约定
│   ├── paths.py          .db/ 路径
│   ├── tools.py          get_current_time / TOOLS
│   ├── errors.py         异常转成一句提示
│   └── runtime.py        模型 / 嵌入 / 重排 / 检索栈装配 + 交互式 CLI
├── chat/
│   ├── langgraph_chat.py 手工 ReAct 图（带 CLI、Sqlite 历史）
│   └── agent_chat.py     纯 create_agent 版（带 CLI、Sqlite 历史）
├── rag/
│   ├── common.py         共用提示词、状态、节点
│   ├── workflow.py       Router + 双路召回 + 排序
│   └── agentic.py        create_agent + 检索工具 + 总结中间件
└── api/
    ├── server.py         FastAPI（SSE 流式）
    └── client.py         命令行客户端
```

## 环境要求

- Python ≥ 3.13，依赖用 [uv](https://docs.astral.sh/uv/) 管理
- 三个 OpenAI 兼容的本地服务：对话 / 嵌入 / 重排（示例端口 `9938` / `9931` / `9932`）

## 配置

```bash
cp .env.example .env
```

| 变量 | 说明 |
|---|---|
| `MODEL` / `OPENAI_API_BASE` | 对话模型与服务地址 |
| `OPENAI_API_KEY` | 本地服务可填任意值 |
| `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` | 嵌入模型与服务地址 |
| `RERANKER_MODEL` / `RERANKER_BASE_URL` | 重排模型与服务地址 |
| `ROUTER_MODE` | `llm`（默认，模型判断）/ `always`（一律检索） |
| `ROUTER_MODEL` / `ROUTER_BASE_URL` | 路由模型；不配则复用 `MODEL` / `OPENAI_API_BASE` |
| `SUMMARY_TRIGGER_TOKENS` | 历史超过多少 token 触发压缩（默认 1500） |
| `SUMMARY_KEEP_MESSAGES` | 压缩后保留最近多少条原文（默认 6） |

## 快速开始

```bash
uv sync
cp .env.example .env    # 按实际服务地址修改

# 1) 下载语料：约 180 篇 markdown 写入 data/langchain-docs/
uv run python data/download_docs.py

# 2) 建向量索引（首次，耗时取决于语料量）
uv run python -c "from taste_rag.core.runtime import load_embeddings; from taste_rag.core.vector_store import ensure_indexed, open_store; print('写入分块:', ensure_indexed(open_store(load_embeddings())))"
```

> bm25 索引每次启动在内存里构建（6700 分块，不到 1 秒），无需持久化。

## 运行

| 入口 | 命令 | 历史文件 |
|---|---|---|
| 基础 chat | `uv run taste-rag` | `.db/chat.db` |
| create_agent chat | `uv run python -m taste_rag.chat.agent_chat` | `.db/agent_chat.db` |
| workflow RAG | `uv run python -m taste_rag.rag.workflow` | `.db/router_chat.db` |
| agentic RAG | `uv run python -m taste_rag.rag.agentic` | `.db/agentic_rag.db` |
| HTTP 服务 | `uv run uvicorn taste_rag.api.server:app --reload` | `.db/chat.db` |
| HTTP 客户端 | `uv run python -m taste_rag.api.client "现在几点？"` | — |

chat 类型的入口支持 `--user` / `--thread`，输入 `exit` / `quit` 结束。

## HTTP 接口

```bash
uv run uvicorn taste_rag.api.server:app --reload
uv run python -m taste_rag.api.client --user u1 --thread t1
```

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/chat` | body `{user_id, thread_id, message}`，SSE 返回 `token` / `error` / `end` 事件 |
| `GET` | `/threads/{user_id}/{thread_id}` | 该会话的历史消息 |
| `GET` | `/health` | 健康检查 |

## 测试

脚本在 `tests/`，按需运行（大多需要本地服务在线）：

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `build_eval_set.py` | 生成检索评测集 `data/eval/queries.json` | 对话模型 |
| `benchmark_retrieval.py` | 检索链路耗时与 Recall / MRR | 嵌入、重排 |
| `compare_ranking.py` | 无排序 / rerank / RRF 横评 | 嵌入、重排 |
| `sweep_fusion.py` | `bm25_k / sem_k / rrf_k / top_n` 参数扫描 | 嵌入、重排 |
| `benchmark_ttft.py` | 四种方案首字延迟（none / rerank / rrf / agent） | 全部服务 |
| `compare_rag.py` | workflow vs agentic 端到端对比 | 全部服务 |
| `compare_chatbots.py` | 纯 `create_agent` vs 手工 ReAct 图 | 对话模型 |
| `compare_router_decision.py` | 检索判断 vs 总是检索 | 全部服务 |

多数脚本支持 `--limit` / `--repeat` / `--tier` 等参数，`--help` 可查看。

## 数据与产物

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `data/langchain-docs/` | 下载的 markdown 语料 | 否 |
| `data/eval/queries.json` | 评测集 | 是 |
| `.chroma/` | Chroma 向量索引 | 否 |
| `.db/` | SQLite 检查点（各入口的会话历史） | 否 |

## 设计文档

- `docs/agentic-vs-workflow.md`：两条 RAG 路线的实现与对比
- `docs/rerank-vs-rrf.md`：三种排序方案的召回质量与首字延迟
- `docs/rrf参数调优.md`：RRF 各参数扫描结果
- `docs/TODO.md`：已完成项、待办与触发条件

## 已知限制

- 4B 级模型对提示词敏感：agentic 的「是否检索」依赖提示词，边界问法仍可能不调用工具
- Agentic 首字中位约 **3.5s**（首字前多一次模型决策），RRF 约 **0.6s**
- 逐字答案不稳定，评测看语义与关键词，不看完全一致

## License

MIT
