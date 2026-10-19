# rag-demo 运行效果报告

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `rag-demo.py` | 自包含实现：`summarize → router → (bm25 ∥ semantic) → rank → chatbot` |
| `rag-demo.toml` | 全部参数与提示词，代码内无字面量配置 |
| `test-rag-demo.py` | 独立测试：importlib 加载、统计首字/耗时/判定/关键词 |

`rag-demo.py` 不 import 项目内任何模块（`src/helpdesk` 未被引用），只依赖第三方库：
langchain / langgraph / chroma / bm25s / langchain-reranker。

## 2. 参数外置

凡是可调的东西都在 `rag-demo.toml`：

| 分区 | 内容 |
|---|---|
| `[chat]` / `[router]` | 模型名、base_url、key、temperature、max_tokens |
| `[embedding]` / `[reranker]` | 嵌入与重排服务地址、`check_ctx_length` |
| `[retrieval]` | 语料目录/glob、索引目录/集合、chunk_size/overlap、批量写入、`bm25_k`、`semantic_k`、`rrf_k`、`top_n`、排序策略、路由模式 |
| `[memory]` | 触发压缩的 token 阈值、保留消息条数 |
| `[prompts]` | 路由、摘要、有资料回答、无资料回答四段提示词 |
| `[runtime]` | 默认 thread_id |

命令行支持覆盖：`--ranking rrf|rerank|none`、`--router-mode llm|always`、`--config`。

## 3. 测试方法

```bash
uv run python test-rag-demo.py --repeat 3                 # 默认 RRF
uv run python test-rag-demo.py --repeat 2 --ranking rerank
uv run python test-rag-demo.py --repeat 2 --ranking none
```

- 语料：6 个英文问题，覆盖 寒暄 / 必检索 / 历史追问 / 致谢 / 重复提问
- 每轮独立 `thread_id`，轮内保持多轮上下文
- 首字延迟（TTFT）：流式取到 `chatbot` 节点第一个 token 为止
- 正式测试前先跑一次 `Hello` 预热，排除建连开销
- 判定正确 = `state.context` 非空 与 期望值一致；关键词 = 答案命中预设词之一

## 4. 结果

### 4.1 三种排序策略对比

| 策略 | 样本 | 检索判定 | 关键词 | 首字中位 | 首字平均 | 总耗时/轮 |
|---|---|---|---|---|---|---|
| **rrf（默认）** | 18 | 18/18 | 18/18 | **579ms** | **610ms** | 13.1s |
| none（不排序） | 12 | 12/12 | 12/12 | 617ms | 672ms | 9.6s |
| rerank | 12 | 12/12 | 12/12 | 762ms | 1199ms | 17.2s |

### 4.2 RRF 逐轮明细（单轮示例）

| 问题 | 检索 | 首字 | 总耗时 | 回答摘要 |
|---|---|---|---|---|
| Hello | 否 | 607ms | 0.7s | Hello! How can I assist you today? |
| How do I define a tool in LangChain? | 是 | 1062ms | 4.4s | 用 `@tool` 装饰器定义，附示例 |
| What is the basic flow of vector store retrieval? | 是 | 1587ms | 3.7s | 索引阶段 + 查询阶段 |
| What technical question did I just ask? | 否 | 972ms | 1.5s | 回答"你刚才问的是向量检索流程" |
| Thanks | 否 | 445ms | 0.5s | You're welcome! |
| How do I define a tool…（重复提问） | 否 | 527ms | 2.1s | 直接复用历史答案，未检索 |

关键点：

- **历史追问与重复提问都不触发检索**，靠多轮上下文直接回答；
- 未检索轮首字稳定在 0.4~0.6s，检索轮因为要跑召回 + 生成，首字升到 1~1.6s；
- 三种策略的判定与关键词都满分，差别只在**首字延迟**与**答案质量**。

### 4.3 排序策略的取舍

- **rrf**：首字最低且无额外依赖，作为默认值。
- **rerank**：首字平均 1199ms，比 rrf 慢近一倍（重排本身约 800ms），且需额外服务。
- **none**：总耗时最低，但出现质量退化——某轮对"如何定义工具"回答了
  *"LangChain does not provide a specific, universal method…"*，
  说明按 bm25 优先的原始顺序把正确答案挤出了上下文。
  **关键词检查通过是因为答案里仍出现了 tool 一词，这项检查偏宽松。**

## 5. 提示词优化过程（本次重点）

| 版本 | 改动 | 检索判定 |
|---|---|---|
| v1 | 「技术问题要查，寒暄和历史可答不查」 | **3/6** |
| v2 | 改为**四步顺序判定**（闲聊 → 问历史 → 历史已有答案 → 技术细节），并附 6 条示例 | **6/6** |
| v3 | 回答语言从「固定中文」改为「跟随提问语言」 | 判定不变，关键词由 5/6 → **6/6** |

失败原因与修正：

1. **v1 判定混乱**：把「向量检索流程」判成不查、把「我刚才问了什么」判成要查。
   小模型对抽象规则不敏感，改成**可执行的四步流程 + 示例**后一次性修正。
2. **v3 之前关键词不匹配**：问题用英文、回答被强制中文，检查用的英文关键词自然命中不了。
   改成跟随提问语言后，答案与提问一致，也符合客服场景。
3. **路由 `max_tokens=128` 被截断**：结构化输出中途截断抛 `LengthFinishReasonError`，
   调到 256 并在代码里对路由异常做「保守检索」兜底。

## 6. 结论

1. **实现可用**：18/18 判定、18/18 关键词，多轮上下文与压缩逻辑正常。
2. **默认 rrf 合理**：首字中位 579ms，与不排序（617ms）基本持平，但避免了 none 的质量退化。
3. **提示词是小模型路由的关键**：同一模型、同一数据，仅改判定结构就从 3/6 提升到 6/6。
4. **rerank 不适合作为默认**：首字翻倍，当前评测集上未见质量优势。

## 7. 复现

```bash
uv run python test-rag-demo.py --repeat 3
uv run python rag-demo.py --ranking rrf        # 交互式，流式输出
```

## 8. 局限

- 评测集只有 6 个问题、3 轮，样本小，结论看方向。
- 「关键词命中」是粗粒度检查，无法区分"答对"与"提到词但答错"；
  `none` 策略的退化正是靠人工看答案才发现。
- 索引复用了项目已有的 `.chroma`（7008 块）；换目录时首次运行需重建，耗时较长。
- 首字与耗时受本地模型服务负载影响，绝对值仅供量级参考。
