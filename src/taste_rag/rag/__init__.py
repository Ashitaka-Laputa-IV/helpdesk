"""RAG 各实现。

common        共用提示词、状态与节点
graph_basic   最简两段式：向量召回 -> 重排 -> 生成
bm25_rerank   多查询改写 + bm25 多路 + 重排
router        Router 决定是否检索，排序可选 rerank / RRF / 无
agentic       Agentic RAG：create_agent + 检索工具
"""
