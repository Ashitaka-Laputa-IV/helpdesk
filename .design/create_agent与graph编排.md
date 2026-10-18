# create_agent 与 graph 编排：两个 chatbot 的对比

## 目标

同一个模型、同一套语料，对比两个多轮 chatbot：

| | 实现 | 构建函数 |
|---|---|---|
| **A** | 纯 `create_agent` | `agentic_chat.build_agent(model)` |
| **B** | `create_agent` + `StateGraph` 编排 | `agent_graph.build_graph(model)` |

两者都只用内存历史（`InMemorySaver`），不落盘；两个模块只提供构建函数，对话由测试脚本驱动。

## 实现差异

**A. 纯 create_agent**：图、工具循环、内存历史全部由 `create_agent` 内部管理。

```python
def build_agent(model):
    return create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                        checkpointer=InMemorySaver())
```

**B. create_agent + graph 编排**：把 A 的产物当成外层图的一个节点。

```python
def build_graph(model):
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)     # 编译好的 agent 直接当节点
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver())
```

B 的图结构：

```mermaid
graph TD
    Start([Start]) --> Agent[agent<br/>create_agent]
    Agent --> END([END])
```

`agent` 节点内部仍是标准 ReAct 循环（模型 → 工具 → 模型），由 `create_agent` 负责。

## 测试

脚本：`tests/compare_chatbots.py`

```bash
uv run python tests/compare_chatbots.py
```

执行方式：**串行**——A 先跑完全部 8 轮，B 再跑一遍；两者使用独立的 `thread_id`，互不影响。
每轮记录：耗时、消息条数、本轮工具调用、校验结果。

### 语料（8 轮）

| 轮次 | 输入 | 考察点 |
|---|---|---|
| 1 | 你好，我叫李浩瑞 | 正常应答 |
| 2 | 我是一名后端工程师，主要写 Python | 记录事实 |
| 3 | 现在几点？ | **工具调用** |
| 4 | 我叫什么名字？ | **历史保持**（第 1 轮） |
| 5 | 我是做什么工作的？ | **历史保持**（第 2 轮） |
| 6 | 我主要用什么编程语言？ | **历史保持**（第 2 轮） |
| 7 | 把刚才说的名字、职业和语言合成一句话 | 多轮归纳 |
| 8 | 现在几点？ | 再次工具调用 |

### 结果

**逐轮耗时**（毫秒）

| 轮次 | A 纯 create_agent | B graph 编排 | 差值 | 消息条数（两者相同） |
|---|---|---|---|---|
| 1 | 253 | 236 | -16 | 2 |
| 2 | 480 | 552 | +73 | 4 |
| 3 | 691 | 675 | -16 | 8 |
| 4 | 150 | 127 | -23 | 10 |
| 5 | 211 | 219 | +9 | 12 |
| 6 | 176 | 162 | -14 | 14 |
| 7 | 218 | 256 | +38 | 16 |
| 8 | 688 | 690 | +2 | 20 |

**汇总**

| 指标 | A | B |
|---|---|---|
| 总耗时 | 2.87s | 2.92s |
| 平均每轮 | 358ms | 365ms |
| 最快 / 最慢 | 150ms / 691ms | 127ms / 690ms |
| 校验通过 | 8/8 | 8/8 |

**关键观察**

- 两者消息条数逐轮完全相同（2, 4, 8, 10, 12, 14, 16, 20），说明图编排没有改变状态演进。
- 第 3、8 轮都真实调用了 `get_current_time`，返回时间与系统时间一致。
- 第 4~7 轮都正确回忆出第 1、2 轮的姓名/职业/语言，历史保持正常。
- 总耗时差 53ms（1.8%），小于同一实现自身的轮次抖动，属于噪声。
- 答案措辞并非逐字相同（完全一致 0/8），这是模型采样的随机性，不是编排差异：
  两者校验均 8/8，语义一致。

## 结论

1. **功能完全等价**：消息条数、工具调用、事实回忆、校验结果逐轮一致。
2. **耗时无实质差异**：2.87s vs 2.92s，差值在噪声范围内。
3. **成本几乎为零**：B 只比 A 多一层单节点包装，没有额外模型调用。
4. **收益在可编排性**：A 要插入检索、改写、审核得改中间件或内部行为；
   B 直接在图上加节点和边即可，例如 `START -> rewrite -> agent -> check -> END`。

因此：**只做纯对话选 A；预判后续要编排选 B，现在不额外付出代价。**
