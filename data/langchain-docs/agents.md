> ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Agents

> Build a LangChain agent: a model calling tools in a loop until a task is complete, shaped by a harness of prompt, tools, and middleware.

An agent is a model calling tools in a loop until a given task is complete.

<img src="https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/core_agent_loop.svg?fit=max&auto=format&n=jtty0O--UJOKG0nK&q=85&s=4b4cbb497b6273758a565de1bc90ece0" alt="Core agent loop diagram" style={{height: "300px", width: "auto", justifyContent: "center"}} className="rounded-lg block mx-auto" width="1060" height="760" data-path="oss/images/core_agent_loop.svg" />

A harness is everything around that loop: the prompt, the tools, and any middleware that shapes the model's behavior.

<Note>
  **Agent = Model + Harness**

  The job of a harness: get the model the right context at the right time for the given task.
</Note>

[`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) is a highly configurable harness. At its simplest, you can create one with:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools)
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="openai:gpt-5.5", tools=tools)
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools)
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools)
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools)
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools)
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools)
  ```
</CodeGroup>

Building on that, you can configure the basics directly with the `model=`, `tools=`, and `system_prompt=` parameters. For more advanced capabilities, extend the harness with [middleware](#configure-the-harness).

<Tip>
  [Deep Agents](/oss/python/deepagents/overview) builds on `create_agent` and comes with commonly useful capabilities already assembled, such as planning, file system tools, subagents, and memory. Use `create_agent` when you need to configure the harness yourself.
</Tip>

## Core components

<img src="https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/agent_model_harness.svg?fit=max&auto=format&n=jtty0O--UJOKG0nK&q=85&s=5ac6a7e0343af7cb5ba3ca632e2224af" alt="Agent model and harness components diagram" style={{height: "280px", width: "auto", justifyContent: "center"}} className="rounded-lg block mx-auto" width="1200" height="760" data-path="oss/images/agent_model_harness.svg" />

### Model

Pass a model identifier string (`"provider:model"`) or an initialized model instance to select the model for your agent. See [Models](/oss/python/langchain/models) for parameters, provider setup, and dynamic model selection.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools)
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="openai:gpt-5.5", tools=tools)
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools)
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools)
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools)
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools)
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent

  agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools)
  ```
</CodeGroup>

### Tools

To provide the agent with tools, pass any Python callable, LangChain tool, or tool dict. See [Tools](/oss/python/langchain/tools) for tool definition, context access, and dynamic tool selection.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="google_genai:gemini-3.6-flash", tools=[search])
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="openai:gpt-5.5", tools=[search])
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="anthropic:claude-sonnet-5", tools=[search])
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=[search])
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=[search])
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=[search])
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for information."""
      return f"Results for: {query}"


  agent = create_agent(model="ollama:north-mini-code-1.0", tools=[search])
  ```
</CodeGroup>

### System prompt

Shape how the agent approaches tasks. The system prompt parameter accepts a string or `SystemMessage`. For dynamic prompts at runtime, use [middleware](/oss/python/langchain/middleware).

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="openai:gpt-5.5",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=tools,
      system_prompt="You are a helpful assistant. Be concise and accurate.",
  )
  ```
</CodeGroup>

### Structured output

Return a validated schema from the agent using `response_format=`. See [Structured output](/oss/python/langchain/structured-output) for strategies and examples.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="openai:gpt-5.5", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel
  from langchain.agents import create_agent


  class Answer(BaseModel):
      summary: str
      confidence: float


  agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools, response_format=Answer)
  result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
  result["structured_response"]  # Answer(summary=..., confidence=...)
  ```
</CodeGroup>

### Agent state

Every agent manages its execution context through [`AgentState`](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentState), a typed dictionary that holds the current conversation history and any custom fields your tools and middleware need.

The built-in field is:

| Field | Type | Description |
| - | - | - |
| `messages` | `list[BaseMessage]` | The full conversation history for the current thread. Append-only: new messages are added, never replaced. |

`AgentState` is also the type signature for every node-style middleware hook (`before_model`, `after_model`, and similar). Hooks receive the current state and can return a dict of updates to merge back into it.

To add custom fields (for example, a `user_id` or a counter), subclass `AgentState` and pass the subclass to `create_agent` via `state_schema=`:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import AgentState, create_agent


  class MyState(AgentState):
      user_id: str
      call_count: int


  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[],
      state_schema=MyState,  # [!code highlight]
  )
  ```
</CodeGroup>

For full details, examples, and middleware-level state schemas, see [Short-term memory](/oss/python/langchain/short-term-memory#customizing-agent-memory) and [Custom middleware](/oss/python/langchain/middleware/custom#state-updates).

## Invocation

<Tip>
  Trace each step of this loop, debug tool calls, and evaluate agent outputs with [LangSmith](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-langchain-agents). Follow the [tracing quickstart](/langsmith/trace-with-langchain) to get set up. We recommend you also set up [LangSmith Engine](/langsmith/engine) which monitors your traces, detects issues, and proposes fixes.
</Tip>

You can invoke an agent with a message. Behind the scenes that passes an update to the agent's [`State`](/oss/python/langgraph/graph-api#state). All agents include a [sequence of messages](/oss/python/langgraph/use-graph-api#messagesstate) in their state; to invoke the agent, pass a new message along with a `thread_id` so the agent can persist and resume conversation history:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[],
      checkpointer=InMemorySaver(),
  )

  config = {"configurable": {"thread_id": str(uuid7())}}

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config=config,
  )

  # A follow-up turn on the same conversation: reuse the same thread_id to keep history
  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
      config=config,
  )
  ```
</CodeGroup>

<Card title="View example trace" icon="chart-line" href="https://smith.langchain.com/public/455ace73-fce4-413b-b47e-613b605ae01d/r" arrow horizontal>
  Open a public LangSmith run for this example.
</Card>

<Note>
  Persisting conversation history with `thread_id` requires the agent to be configured with a [checkpointer](/oss/python/langchain/long-term-memory). When deployed on [LangSmith](/langsmith/deployment), a checkpointer is provisioned automatically. Locally, pass one explicitly, for example `create_agent(..., checkpointer=InMemorySaver())`.
</Note>

If you also need to pass per-run configuration (such as a user ID, API keys, or feature flags) to tools and middleware, pass it as `context` alongside `config`. Define the shape of that data with `context_schema` and access it through `runtime.context`:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain_core.utils.uuid import uuid7
  from langgraph.checkpoint.memory import InMemorySaver


  @dataclass
  class Context:
      user_id: str


  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[],
      context_schema=Context,
      checkpointer=InMemorySaver(),
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid7())}},
      context=Context(user_id="user-123"),
  )
  ```
</CodeGroup>

<Card title="View example trace" icon="chart-line" href="https://smith.langchain.com/public/2ccaff61-e004-4356-81f3-4c061ef7a93a/r" arrow horizontal>
  Open a public LangSmith run for this example.
</Card>

`thread_id` scopes the *conversation* (message history, checkpoints), while `context` carries *per-run* data your tools and middleware read at invocation time. Both are commonly passed together. See [tool context](/oss/python/langchain/tools#context) and [Runtime](/oss/python/langchain/runtime) for more.

## Streaming

`invoke` returns the final response at the end of a run. If an agent executes multiple tool calls, users often need progress updates before completion. Use streaming to surface intermediate messages and tool activity as they happen.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import AIMessage, HumanMessage


stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]},
    version="v3",
)
for snapshot in stream.values:
    # Each snapshot contains the full state at that point
    latest_message = snapshot["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

<Card title="View example trace" icon="chart-line" href="https://smith.langchain.com/public/eb78aa8e-9575-4c73-8e79-bad82b30ef1d/r" arrow horizontal>
  Open a public LangSmith run for this example.
</Card>

<Tip>
  For streaming modes, event types, and UI patterns, see [Streaming](/oss/python/langchain/streaming).
</Tip>

## Configure the harness

`create_agent` is highly extensible. Middleware is the primitive for customization: each piece handles one concern, hooks into the agent loop at the right moment, and composes freely with any other. Take exactly what your use case needs and skip the rest.

Common patterns are prebuilt as first-class middleware. You can build anything else as [custom middleware](/oss/python/langchain/middleware/custom).

<img src="https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/agent_harness_capabilities.svg?fit=max&auto=format&n=jtty0O--UJOKG0nK&q=85&s=0ff671d72badd0844826660dfcb04391" alt="Agent harness capabilities by category" style={{height: "300px", width: "auto", justifyContent: "center"}} className="rounded-lg block mx-auto" width="1500" height="360" data-path="oss/images/agent_harness_capabilities.svg" />

As agents take on complex work, they need support across a few key areas. The middleware ecosystem provides:

<CardGroup cols={2}>
  <Card title="Execution environment" icon="bolt" href="#execution-environment">
    Tools, filesystem, sandboxes, and code execution
  </Card>

  <Card title="Context management" icon="database" href="#context-management">
    Summarization, memory, skills, and prompt caching
  </Card>

  <Card title="Planning and delegation" icon="sitemap" href="#planning-and-delegation">
    Todo lists and subagents for parallel, isolated work
  </Card>

  <Card title="Fault tolerance" icon="shield" href="#fault-tolerance">
    Retries, fallbacks, and call limits
  </Card>

  <Card title="Guardrails" icon="lock" href="#guardrails">
    PII detection and content controls
  </Card>

  <Card title="Steering" icon="user" href="#steering">
    Human-in-the-loop approval before high-impact actions
  </Card>
</CardGroup>

<Tip>
  `create_deep_agent` pre-assembles this stack for long-running coding and research tasks (filesystem, summarization, subagents, and prompt caching included by default). See [Deep Agents](/oss/python/deepagents/harness) for the full prebuilt harness.
</Tip>

### Execution environment

Agents are especially useful when they can take action rather than just generate text. The execution environment gives the agent a workspace: tools it can call, a filesystem for reading and writing files across turns, and code execution for running scripts or shell commands.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware

  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[search],
      middleware=[FilesystemMiddleware(backend=StateBackend())],
  )
  ```
</CodeGroup>

See [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware), [Sandboxes](/oss/python/deepagents/sandboxes), [Interpreters](/oss/python/deepagents/interpreters).

<Note>
  This example imports from the `deepagents` package. Install it with:

  <CodeGroup>
    ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install deepagents
    ```

    ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    uv add deepagents
    ```
  </CodeGroup>
</Note>

### Context management

Every model call has a fixed context window. As an agent runs, that window fills with accumulating history, tool results, and intermediate steps. Summarization compresses history before overflow hits; memory loads persistent instructions at startup so knowledge carries across sessions; skills surface domain knowledge on demand rather than loading everything upfront.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "google_genai:gemini-3.6-flash"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "openai:gpt-5.5"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "anthropic:claude-sonnet-5"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "openrouter:z-ai/glm-5.2"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "fireworks:accounts/fireworks/models/glm-5p2"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "baseten:zai-org/GLM-5.2"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

  backend = StateBackend()
  model = "ollama:north-mini-code-1.0"

  agent = create_agent(
      model=model,
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          SummarizationMiddleware(model=model, backend=backend),
          MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
          SkillsMiddleware(backend=backend, sources=["./skills/"]),
      ],
  )
  ```
</CodeGroup>

See [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware), [`MemoryMiddleware`](https://reference.langchain.com/python/deepagents/middleware/memory/MemoryMiddleware), [Skills](/oss/python/langchain/multi-agent/skills), [Context engineering](/oss/python/deepagents/context-engineering).

<Note>
  This example imports from the `deepagents` package. Install it with:

  <CodeGroup>
    ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install deepagents
    ```

    ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    uv add deepagents
    ```
  </CodeGroup>
</Note>

### Planning and delegation

Complex tasks often exceed what one context window can handle. Delegation lets the main agent break work into pieces, hand them to subagents that each run in their own isolated context, and stay focused on coordination rather than execution. Work can run in parallel; the main agent's context stays clean.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from deepagents.middleware.subagents import SubAgentMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import TodoListMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  backend = StateBackend()

  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[search],
      middleware=[
          FilesystemMiddleware(backend=backend),
          TodoListMiddleware(),
          SubAgentMiddleware(
              backend=backend,
              subagents=[
                  {
                      "name": "researcher",
                      "description": "Searches and returns a structured summary.",
                      "system_prompt": "Use the search tool to research the question and summarize key points.",
                      "tools": [search],
                      "model": "anthropic:claude-sonnet-4-6",
                      "middleware": [],
                  }
              ],
          ),
      ],
  )
  ```
</CodeGroup>

See [Subagents](/oss/python/langchain/multi-agent/subagents).

<Note>
  This example imports from the `deepagents` package. Install it with:

  <CodeGroup>
    ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install deepagents
    ```

    ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    uv add deepagents
    ```
  </CodeGroup>
</Note>

### Name your agent

Optionally use an identifier for the agent. This is especially useful when embedding the agent as a subgraph in [multi-agent](/oss/python/langchain/multi-agent) systems.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools, name="research_assistant")
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="openai:gpt-5.5", tools=tools, name="research_assistant")
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools, name="research_assistant")
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools, name="research_assistant")
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools, name="research_assistant")
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools, name="research_assistant")
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools, name="research_assistant")
  ```
</CodeGroup>

### Fault tolerance

Agents in production encounter failures that rarely appear in development: rate limits, model timeouts, transient API errors. Fault tolerance middleware handles these at the infrastructure level so your tools and business logic don't need try/catch around every call.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[search],
      middleware=[
          ModelRetryMiddleware(max_retries=3),
          ToolRetryMiddleware(max_retries=2),
      ],
  )
  ```
</CodeGroup>

See [`ModelRetryMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/model_retry/ModelRetryMiddleware), [`ToolRetryMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/tool_retry/ToolRetryMiddleware), [Prebuilt middleware](/oss/python/langchain/middleware/built-in).

### Guardrails

Some policies can't live in a prompt—they need to be enforced deterministically regardless of what the model does. Guardrails intercept data as it flows through the agent loop, applying compliance rules or content policies before tool results reach the model's context.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import PIIMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[search],
      middleware=[PIIMiddleware("email")],
  )
  ```
</CodeGroup>

See [`PIIMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware), [Prebuilt middleware](/oss/python/langchain/middleware/built-in).

### Steering

Full autonomy isn't always appropriate. Steering lets you place humans at specific decision points—before destructive writes, expensive API calls, or anything requiring judgment—without restructuring your agent. The agent pauses and waits; a human approves, edits, or rejects; execution continues.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="google_genai:gemini-3.6-flash",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="openai:gpt-5.5",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="anthropic:claude-sonnet-5",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="openrouter:z-ai/glm-5.2",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="fireworks:accounts/fireworks/models/glm-5p2",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="baseten:zai-org/GLM-5.2",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  from langchain.tools import tool


  @tool
  def search(query: str) -> str:
      """Search for a query and return a short summary."""
      return f"Search results for: {query}"


  agent = create_agent(
      model="ollama:north-mini-code-1.0",
      tools=[search],
      middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
  )
  ```
</CodeGroup>

See [`HumanInTheLoopMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/human_in_the_loop/HumanInTheLoopMiddleware), [Human-in-the-loop](/oss/python/langchain/human-in-the-loop).

### Middleware resources

<CardGroup cols={3}>
  <Card title="Middleware overview" icon="route" href="/oss/python/langchain/middleware/overview" arrow>
    How the middleware stack works and when hooks fire
  </Card>

  <Card title="Prebuilt middleware" icon="package" href="/oss/python/langchain/middleware/built-in" arrow>
    Full reference with configuration examples
  </Card>

  <Card title="Custom middleware" icon="code" href="/oss/python/langchain/middleware/custom" arrow>
    Write your own hooks for business logic, PII scrubbing, and more
  </Card>
</CardGroup>

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to your agent of choice via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/agents.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
