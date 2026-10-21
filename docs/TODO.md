# TODO

> 更新于 2026-10-21。设计说明见同目录三份报告：
> `agentic-vs-workflow.md`、`rerank-vs-rrf.md`、`rrf参数调优.md`。

## 已完成（架构）

- [x] 项目改名 `taste-rag`，包名 `taste_rag`
- [x] 目录分层：`core`（检索 / 运行时）、`chat`、`rag`、`api`
- [x] 检索原语收敛到 `core/retrieval.py`：
      `bm25_candidates` / `sem_candidates` / `dedupe` / `rrf_fuse` / `rerank` / `format_context`
- [x] 依赖装配与交互式 CLI 收敛到 `core/runtime.py`
- [x] 两版 router 合并为 `rag/workflow.py`，`ranking` 可选 `rerank` / `rrf` / `none`
- [x] Agentic RAG 接入官方 `SummarizationMiddleware`
- [x] 单轮失败不再抛 traceback：`core/errors.py` + `runtime.run_cli`
- [x] SQLite 检查点统一到 `.db/`（`core/paths.py`）
- [x] 删除 `rag/bm25_rerank.py`、`rag/graph_basic.py`、`tests/compare_ranking_e2e.py`
- [x] `docs/` 只保留三份核心报告

## 待办

### 工程整洁

- [ ] `README.md` 仍是空文件：补项目简介、目录结构、快速开始（`.env` 配置 + 三条链路的启动命令）
- [ ] `pyproject.toml` 的 `description` 还是占位符 `Add your description here`
- [ ] `.db/` 内 `bm25_chat.db`、`router_chat_rrf.db` 属于已删除模块，可直接清理
- [ ] `core/vector_store.py` 的 `load_documents` / `ensure_indexed` 已无调用方：
      是保留（目前唯一的重建索引入口）还是删除，待定

### 路径与命名

- [ ] 仓库目录仍叫 `helpdesk`，改名后可同步为 `taste-rag`
- [ ] Chroma 集合名仍是 `helpdesk`：改名需重建 7008 个分块（数分钟）
- [ ] `.chroma` 是否也并入 `.db/chroma`，与 SQLite 检查点放在一起

### 行为与质量

- [ ] Agentic 的「要不要检索」完全依赖提示词：4B 模型在边界问法上仍可能不调工具。
      可加代码兜底——首轮没有 `tool_calls` 且问题非寒暄时，强制检索一次再让模型作答
- [ ] `tests/compare_chatbots.py`、`tests/compare_router_decision.py` 暂无对应文档，
      决定是补报告还是删除
- [ ] 错误提示已覆盖连接 / 超时 / 鉴权 / 限流 / 404，未覆盖「模型返回空内容」

## 暂不做（满足条件再动）

### RRF 换用官方 `EnsembleRetriever`

现状：`core/retrieval.py` 的 `rrf_fuse()` 自研十几行，两路等权融合，无额外依赖。

满足任一条再考虑切换：

- [ ] 需要 >2 路融合（例如加标题检索、时间衰减）
- [ ] 需要给每一路调权重（官方支持 `weights`，自研需自己扩展）
- [ ] bm25s 也封装成 `BaseRetriever`，适配成本接近零

切换成本：

- 需为 bm25s 写 `BaseRetriever` 适配器，并在 `Document` 与 `{"source", "text"}` 之间转换
- 需引入 `langchain-classic` 依赖
- `tests/compare_ranking.py`、`tests/sweep_fusion.py` 里的本地 `rrf_order()` 要同步替换

已验证的事实：

- 官方 `c` 默认 60，本语料实测 `rrf_k=20` 的 MRR 更高（0.723 vs 0.690），切换时须显式传 `c=20`
- 自研版同分时保持首次出现顺序（稳定排序）；官方按 `id_key` 聚合，平局行为未验证，切换后需跑评测集回归

## 已知数据（详见 docs）

- 首字中位：无排序 745ms ｜ rerank 1716ms ｜ RRF 631ms ｜ Agentic 3509ms
- 检索质量（英文改写档 top-5）：RRF 83.3% ＞ rerank 76.7%
- 端到端：workflow 与 agentic 的检索判定正确率相当；agentic 省掉固定 router 调用，但首字更慢
