# RAG 问答图

对应实现：`src/helpdesk/router_chat_rrf.py`（RRF 版）/ `src/helpdesk/router_chat.py`（rerank 版）。

```mermaid
graph TD
    Start([Start]) --> Summarize["summarize<br/><small>历史超预算时压缩成摘要</small>"]

    Summarize --> Router{"router<br/><small>llm：模型判断要不要查<br/>always：一律查，不判断</small>"}

    Router -- "要查" --> BM25["bm25 召回<br/><small>k=25</small>"]
    Router -- "要查" --> Semantic["语义召回<br/><small>k=15</small>"]
    Router -- "不查" --> Chat["chatbot"]

    BM25 --> Rank
    Semantic --> Rank

    Rank["rank 检索后排序<br/><small>rrf（k=20）/ rerank / none</small>"] --> Chat

    Chat --> End([End])
```

## 说明

- **router 只有两种模式**：`llm` 交给模型判断，`always` 一律检索。寒暄、致谢不再特判，
  与其他问题走同一条判断链路。
- **两路召回是并行的**：`bm25` 与 `semantic` 从 router 同时出发，各自返回候选，
  统一交给 `rank` 节点，而不是各自再走一条排序分支。
- **`rank` 是编译期开关**（`build_graph(..., ranking=...)`），不是运行期分支，
  所以图上画成一个节点、三种取值：
  - `rrf`：按名次倒数融合（`k=20`），默认值；
  - `rerank`：cross-encoder 重排（`bge-reranker-v2-m3`）；
  - `none`：沿用召回顺序（bm25 在前、向量补位），仅用于对比。
- **不检索时直接进 `chatbot`**：不经过召回与排序，延迟最低。
- 节点名与代码一致：`summarize` / `router` / `bm25` / `semantic` / `rank` / `chatbot`。
