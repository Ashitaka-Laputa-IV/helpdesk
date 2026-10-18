# create_agent 与 graph 编排：两个 chatbot 的对比

## 目标

用同一个模型、同一套对话语料，对比两个多轮 chatbot：

| | 实现 | 文件 |
|---|---|---|
| **A** | 纯 `create_agent` | `src/helpdesk/agentic_chat.py` |
| **B** | `create_agent` + `StateGraph` 编排 | `src/helpdesk/agent_graph.py` |

两者都只用内存历史（`InMemorySaver`），不落盘。

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

### 语料

```text
你好，我叫李浩瑞
我叫什么名字？
现在几点？
```

三轮分别考察：正常应答、**历史保持**、**工具调用**。两者使用相同的 `MODEL`、同一套 `TOOLS`（`get_current_time`）。

### 结果

| 轮次 | 输入 | A. 纯 create_agent | B. graph 编排 |
|---|---|---|---|
| 1 | 你好，我叫李浩瑞 | 你好，李浩瑞！有什么可以帮你的吗？ | 同左 |
| 2 | 我叫什么名字？ | 你叫李浩瑞。 | 同左 |
| 3 | 现在几点？ | 现在是2026年10月18日15时54分04秒。 | 现在是2026年10月18日15时54分05秒。 |

耗时与消息轨迹：

| 轮次 | A 耗时 / 消息数 | B 耗时 / 消息数 | 序列 |
|---|---|---|---|
| 1 | 0.23s / 2 | 0.20s / 2 | `human, ai` |
| 2 | 0.13s / 4 | 0.13s / 4 | `human, ai, human, ai` |
| 3 | 0.67s / 8 | 0.68s / 8 | `human, ai, human, ai, human, ai, tool, ai` |

第 3 轮两者都触发了 `get_current_time`，工具返回值与系统时间一致，确认是真实调用。

## 结论

1. **功能完全等价**：输出、消息条数、消息序列、工具调用行为逐轮一致，只有时间戳因为执行时刻不同差了 1 秒。
2. **耗时无实质差异**：三轮回合分别 1.03s / 1.01s，差值在噪声范围（单次调用抖动就有 ±0.05s）。
3. **成本几乎为零**：B 比 A 只多一层单节点包装，没有引入额外的模型调用或状态转换。
4. **收益在可编排性**：A 想插入检索、改写、审核等环节得改 `create_agent` 的内部行为或加中间件；
   B 直接在图上加节点和边即可，比如 `START -> rewrite -> agent -> check -> END`。

因此：**只做纯对话选 A（少一层抽象）；预判后续要编排选 B（现在不额外付出代价）。**
