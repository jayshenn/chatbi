# 阶段 4：LangChain Agents + LangGraph 编排 + LiteLLM + LLMOps

> 预计时间：4 周

## 学习目标

- 掌握 LangChain v1 Agent 和 Tools 机制
- 使用 LangGraph 实现有状态的多步骤工作流
- 配置 LiteLLM 多模型路由
- 接入 Langfuse 实现可观测性
- 使用 Ragas 评估 RAG/Agent 效果

## 前置条件

- 完成 [阶段 3](phase-3-service-webdemo-text2sql.md)
- RAG 和 Text-to-SQL 服务已可用

---

## Part 1: LangChain v1 Agents + Tools

### Step 1: 安装依赖

```bash
# 更新 pyproject.toml
# dependencies = [
#     ...
#     "langchain>=0.3.0",
#     "langchain-core>=0.3.0",
#     "langchain-openai>=0.2.0",
#     "langgraph>=0.2.0",
# ]

uv pip install -e ".[dev]"
```

### Step 2: 创建 Tools

创建 `src/chatbi/agents/__init__.py`：

```python
"""Agent 模块"""
```

创建 `src/chatbi/agents/tools.py`：

```python
"""Agent 工具定义"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chatbi.rag_llamaindex.index import load_llamaindex
from chatbi.rag_llamaindex.query import LlamaIndexQueryEngine
from chatbi.services.text_to_sql import text_to_sql_service
from chatbi.sql.database import execute_query


# ========== RAG 工具 ==========

class RAGLookupInput(BaseModel):
    """RAG 查询输入"""
    question: str = Field(..., description="要查询的问题，关于指标定义、表结构、数仓设计等")


@tool("rag_lookup", args_schema=RAGLookupInput)
def rag_lookup(question: str) -> str:
    """
    查询知识库文档。

    用于查询指标口径、表结构说明、数仓分层设计等文档内容。
    当用户询问某个指标的定义、某张表的字段含义、或数据相关的概念时使用。
    """
    try:
        index = load_llamaindex()
        engine = LlamaIndexQueryEngine(index)
        result = engine.query(question)

        sources = "\n".join([f"- {s['metadata'].get('file_name', '未知')}" for s in result.sources])
        return f"{result.answer}\n\n引用来源：\n{sources}"
    except Exception as e:
        return f"查询知识库失败: {str(e)}"


# ========== SQL 工具 ==========

class GenerateSQLInput(BaseModel):
    """生成 SQL 输入"""
    question: str = Field(..., description="需要查询数据的自然语言问题")


@tool("generate_sql", args_schema=GenerateSQLInput)
def generate_sql(question: str) -> str:
    """
    根据自然语言问题生成 SQL 查询语句。

    用于将用户的数据查询需求转换为 SQL。
    只生成 SQL 不执行，用于需要用户确认或进一步修改的场景。
    """
    try:
        result = text_to_sql_service.generate_sql(question)
        return f"SQL:\n```sql\n{result.sql}\n```\n\n解释: {result.explanation}"
    except Exception as e:
        return f"生成 SQL 失败: {str(e)}"


class RunSQLInput(BaseModel):
    """执行 SQL 输入"""
    sql: str = Field(..., description="要执行的 SQL 查询语句，必须是 SELECT 语句")


@tool("run_sql", args_schema=RunSQLInput)
def run_sql(sql: str) -> str:
    """
    执行 SQL 查询并返回结果。

    只能执行 SELECT 查询，禁止执行 INSERT/UPDATE/DELETE 等操作。
    返回查询结果的前 20 行。
    """
    try:
        # 安全检查
        sql_upper = sql.upper()
        if any(word in sql_upper for word in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE"]):
            return "错误：不允许执行修改数据的操作"

        rows = execute_query(sql)

        if not rows:
            return "查询结果为空"

        # 格式化输出
        if len(rows) > 20:
            rows = rows[:20]
            truncated = True
        else:
            truncated = False

        import pandas as pd
        df = pd.DataFrame(rows)
        result = df.to_markdown(index=False)

        if truncated:
            result += "\n\n(结果已截断，仅显示前 20 行)"

        return result
    except Exception as e:
        return f"执行 SQL 失败: {str(e)}"


class QueryDataInput(BaseModel):
    """查询数据输入"""
    question: str = Field(..., description="需要查询数据的自然语言问题")


@tool("query_data", args_schema=QueryDataInput)
def query_data(question: str) -> str:
    """
    自然语言查询数据库。

    将自然语言问题转换为 SQL 并执行，返回查询结果和解释。
    适用于需要从数据库获取具体数据的场景。
    """
    try:
        result = text_to_sql_service.query(question)

        import pandas as pd
        df = pd.DataFrame(result.rows)

        output = f"执行的 SQL:\n```sql\n{result.sql}\n```\n\n"
        output += f"查询结果 ({result.row_count} 行):\n{df.head(10).to_markdown(index=False)}\n\n"
        output += f"分析: {result.result_explanation}"

        return output
    except Exception as e:
        return f"查询数据失败: {str(e)}"


# ========== 工具集合 ==========

ALL_TOOLS = [
    rag_lookup,
    generate_sql,
    run_sql,
    query_data,
]
```

### Step 3: 创建 Agent

创建 `src/chatbi/agents/analytics_agent.py`：

```python
"""数据分析 Agent"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chatbi.config import get_settings
from chatbi.agents.tools import ALL_TOOLS


SYSTEM_PROMPT = """你是一个智能数据分析助手，可以帮助用户：

1. **查询文档**：查询指标定义、表结构、数仓设计等知识库内容
2. **查询数据**：通过自然语言查询数据库，获取具体的数据结果

## 可用工具

- `rag_lookup`: 查询知识库文档，用于了解指标定义、表结构等
- `generate_sql`: 生成 SQL 但不执行，用于需要用户确认的场景
- `run_sql`: 执行 SQL 查询
- `query_data`: 自然语言查询数据，自动生成并执行 SQL

## 工作流程

1. 理解用户问题，判断需要查文档还是查数据
2. 如果涉及指标定义或概念，先用 `rag_lookup` 查询相关文档
3. 如果需要查询具体数据，使用 `query_data` 或 `generate_sql` + `run_sql`
4. 综合所有信息，给出完整的回答

## 注意事项

- 如果不确定某个指标的定义，先查文档再查数据
- 回答要准确、简洁，包含数据来源
- 如果遇到错误，尝试分析原因并调整查询
"""


def create_analytics_agent() -> AgentExecutor:
    """创建数据分析 Agent"""
    settings = get_settings()

    # 初始化 LLM
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.3,
    )

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Agent
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    # 创建 AgentExecutor
    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    return executor


# 全局 Agent 实例
_agent_executor: AgentExecutor | None = None


def get_agent_executor() -> AgentExecutor:
    """获取 Agent 执行器"""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = create_analytics_agent()
    return _agent_executor


def run_agent(question: str, chat_history: list | None = None) -> dict:
    """
    运行 Agent

    Args:
        question: 用户问题
        chat_history: 对话历史

    Returns:
        {"output": str, "intermediate_steps": list}
    """
    executor = get_agent_executor()
    result = executor.invoke({
        "input": question,
        "chat_history": chat_history or [],
    })
    return result
```

### Step 4: 添加 Agent API

创建 `src/chatbi/api/agent.py`：

```python
"""Agent API 路由"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbi.agents.analytics_agent import run_agent

router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentQueryRequest(BaseModel):
    """Agent 查询请求"""
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: list[dict] | None = None


class AgentQueryResponse(BaseModel):
    """Agent 查询响应"""
    answer: str
    intermediate_steps: list[dict] | None = None
    query_time_ms: int


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(request: AgentQueryRequest):
    """
    Agent 统一入口

    智能判断用户意图，自动选择查文档或查数据。
    """
    start_time = time.time()

    try:
        # 转换 chat_history 格式
        chat_history = []
        if request.chat_history:
            for msg in request.chat_history:
                if msg.get("role") == "user":
                    chat_history.append(("human", msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    chat_history.append(("ai", msg.get("content", "")))

        result = run_agent(request.question, chat_history)

        query_time_ms = int((time.time() - start_time) * 1000)

        # 处理中间步骤
        steps = []
        if result.get("intermediate_steps"):
            for action, observation in result["intermediate_steps"]:
                steps.append({
                    "tool": action.tool,
                    "input": action.tool_input,
                    "output": str(observation)[:500],  # 截断
                })

        return AgentQueryResponse(
            answer=result["output"],
            intermediate_steps=steps,
            query_time_ms=query_time_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

注册路由：

```python
# main.py
from chatbi.api.agent import router as agent_router
app.include_router(agent_router)
```

---

## Part 2: LangGraph 有状态工作流

### Step 5: 定义工作流状态

创建 `src/chatbi/workflows/__init__.py`：

```python
"""工作流模块"""
```

创建 `src/chatbi/workflows/state.py`：

```python
"""工作流状态定义"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AnalyticsState(TypedDict):
    """数据分析工作流状态"""

    # 消息历史（使用 add_messages reducer 自动追加）
    messages: Annotated[list, add_messages]

    # 用户原始问题
    question: str

    # 意图分类：rag / sql / hybrid
    intent: str | None

    # RAG 检索结果
    rag_context: str | None
    rag_sources: list[dict] | None

    # SQL 相关
    generated_sql: str | None
    sql_error: str | None
    sql_retry_count: int
    query_result: list[dict] | None

    # 最终答案
    final_answer: str | None

    # 错误信息
    error: str | None
```

### Step 6: 创建工作流节点

创建 `src/chatbi/workflows/nodes.py`：

```python
"""工作流节点定义"""

import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from chatbi.config import get_settings
from chatbi.workflows.state import AnalyticsState
from chatbi.rag_llamaindex.index import load_llamaindex
from chatbi.rag_llamaindex.query import LlamaIndexQueryEngine
from chatbi.services.text_to_sql import text_to_sql_service
from chatbi.sql.database import execute_query


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.3,
    )


# ========== 意图识别节点 ==========

INTENT_PROMPT = """分析用户问题的意图，判断属于以下哪种类型：

1. **rag** - 查询文档/知识库，例如：指标定义、表结构说明、概念解释
2. **sql** - 查询具体数据，例如：统计数据、查询记录、数据分析
3. **hybrid** - 混合查询，需要先查文档了解概念再查数据

用户问题：{question}

请只返回一个词：rag、sql 或 hybrid
"""


def route_intent(state: AnalyticsState) -> AnalyticsState:
    """意图识别节点"""
    llm = get_llm()

    response = llm.invoke([
        HumanMessage(content=INTENT_PROMPT.format(question=state["question"]))
    ])

    intent = response.content.strip().lower()
    if intent not in ["rag", "sql", "hybrid"]:
        intent = "hybrid"  # 默认混合模式

    return {**state, "intent": intent}


# ========== RAG 节点 ==========

def rag_node(state: AnalyticsState) -> AnalyticsState:
    """RAG 检索节点"""
    try:
        index = load_llamaindex()
        engine = LlamaIndexQueryEngine(index)
        result = engine.query(state["question"])

        return {
            **state,
            "rag_context": result.answer,
            "rag_sources": result.sources,
        }
    except Exception as e:
        return {**state, "error": f"RAG 检索失败: {str(e)}"}


# ========== SQL 生成节点 ==========

def generate_sql_node(state: AnalyticsState) -> AnalyticsState:
    """SQL 生成节点"""
    try:
        # 如果有之前的错误，加入上下文
        question = state["question"]
        if state.get("sql_error"):
            question = f"{question}\n\n注意：之前生成的 SQL 执行出错：{state['sql_error']}，请修正。"

        result = text_to_sql_service.generate_sql(question)

        return {
            **state,
            "generated_sql": result.sql,
            "sql_error": None,
        }
    except Exception as e:
        return {**state, "error": f"SQL 生成失败: {str(e)}"}


# ========== SQL 执行节点 ==========

def run_sql_node(state: AnalyticsState) -> AnalyticsState:
    """SQL 执行节点"""
    if not state.get("generated_sql"):
        return {**state, "error": "没有可执行的 SQL"}

    try:
        rows = execute_query(state["generated_sql"])
        return {
            **state,
            "query_result": rows,
            "sql_error": None,
            "sql_retry_count": state.get("sql_retry_count", 0),
        }
    except Exception as e:
        retry_count = state.get("sql_retry_count", 0) + 1
        return {
            **state,
            "sql_error": str(e),
            "sql_retry_count": retry_count,
            "query_result": None,
        }


# ========== 结果汇总节点 ==========

SUMMARIZE_PROMPT = """请根据以下信息回答用户的问题。

用户问题：{question}

{rag_section}

{sql_section}

请给出完整、准确的回答：
"""


def summarize_node(state: AnalyticsState) -> AnalyticsState:
    """结果汇总节点"""
    llm = get_llm()

    # 构建 RAG 部分
    rag_section = ""
    if state.get("rag_context"):
        sources = ""
        if state.get("rag_sources"):
            sources = "\n".join([f"- {s.get('metadata', {}).get('file_name', '未知')}"
                                for s in state["rag_sources"]])
        rag_section = f"""
## 文档查询结果
{state['rag_context']}

引用来源：
{sources}
"""

    # 构建 SQL 部分
    sql_section = ""
    if state.get("query_result") is not None:
        import pandas as pd
        df = pd.DataFrame(state["query_result"])
        if len(df) > 0:
            result_table = df.head(10).to_markdown(index=False)
        else:
            result_table = "查询结果为空"

        sql_section = f"""
## 数据查询结果
执行的 SQL：
```sql
{state.get('generated_sql', '')}
```

查询结果（{len(state['query_result'])} 行）：
{result_table}
"""

    prompt = SUMMARIZE_PROMPT.format(
        question=state["question"],
        rag_section=rag_section,
        sql_section=sql_section,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    return {**state, "final_answer": response.content}


# ========== 错误处理节点 ==========

def error_node(state: AnalyticsState) -> AnalyticsState:
    """错误处理节点"""
    error_msg = state.get("error", "未知错误")
    return {
        **state,
        "final_answer": f"抱歉，处理您的请求时遇到了问题：{error_msg}"
    }
```

### Step 7: 构建工作流图

创建 `src/chatbi/workflows/analytics_workflow.py`：

```python
"""数据分析工作流"""

from langgraph.graph import StateGraph, END

from chatbi.workflows.state import AnalyticsState
from chatbi.workflows.nodes import (
    route_intent,
    rag_node,
    generate_sql_node,
    run_sql_node,
    summarize_node,
    error_node,
)


def should_retry_sql(state: AnalyticsState) -> str:
    """判断是否需要重试 SQL"""
    if state.get("sql_error") and state.get("sql_retry_count", 0) < 2:
        return "retry"
    elif state.get("sql_error"):
        return "error"
    return "continue"


def route_by_intent(state: AnalyticsState) -> str:
    """根据意图路由"""
    intent = state.get("intent", "hybrid")
    if state.get("error"):
        return "error"
    return intent


def build_analytics_workflow() -> StateGraph:
    """构建数据分析工作流"""

    # 创建图
    workflow = StateGraph(AnalyticsState)

    # 添加节点
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("rag", rag_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("run_sql", run_sql_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("error", error_node)

    # 设置入口
    workflow.set_entry_point("route_intent")

    # 添加条件边
    workflow.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {
            "rag": "rag",
            "sql": "generate_sql",
            "hybrid": "rag",  # 混合模式先查 RAG
            "error": "error",
        }
    )

    # RAG -> 判断是否需要查数据
    def after_rag(state: AnalyticsState) -> str:
        if state.get("error"):
            return "error"
        if state.get("intent") == "hybrid":
            return "sql"
        return "summarize"

    workflow.add_conditional_edges(
        "rag",
        after_rag,
        {
            "sql": "generate_sql",
            "summarize": "summarize",
            "error": "error",
        }
    )

    # SQL 生成 -> 执行
    workflow.add_edge("generate_sql", "run_sql")

    # SQL 执行 -> 判断结果
    workflow.add_conditional_edges(
        "run_sql",
        should_retry_sql,
        {
            "retry": "generate_sql",
            "continue": "summarize",
            "error": "error",
        }
    )

    # 汇总 -> 结束
    workflow.add_edge("summarize", END)
    workflow.add_edge("error", END)

    return workflow


# 编译工作流
analytics_app = build_analytics_workflow().compile()


def run_workflow(question: str) -> dict:
    """
    运行工作流

    Args:
        question: 用户问题

    Returns:
        最终状态
    """
    initial_state = AnalyticsState(
        messages=[],
        question=question,
        intent=None,
        rag_context=None,
        rag_sources=None,
        generated_sql=None,
        sql_error=None,
        sql_retry_count=0,
        query_result=None,
        final_answer=None,
        error=None,
    )

    result = analytics_app.invoke(initial_state)
    return result
```

### Step 8: 添加工作流 API

创建 `src/chatbi/api/workflow.py`：

```python
"""工作流 API"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbi.workflows.analytics_workflow import run_workflow

router = APIRouter(prefix="/workflow", tags=["Workflow"])


class WorkflowQueryRequest(BaseModel):
    """工作流查询请求"""
    question: str = Field(..., min_length=1, max_length=1000)


class WorkflowQueryResponse(BaseModel):
    """工作流查询响应"""
    answer: str
    intent: str | None
    sql: str | None
    sources: list[str] | None
    query_time_ms: int


@router.post("/query", response_model=WorkflowQueryResponse)
async def query_workflow(request: WorkflowQueryRequest):
    """
    LangGraph 工作流查询

    使用有状态工作流处理复杂查询，支持自动路由、错误重试。
    """
    start_time = time.time()

    try:
        result = run_workflow(request.question)

        sources = None
        if result.get("rag_sources"):
            sources = [s.get("metadata", {}).get("file_name", "未知")
                      for s in result["rag_sources"]]

        return WorkflowQueryResponse(
            answer=result.get("final_answer", "处理失败"),
            intent=result.get("intent"),
            sql=result.get("generated_sql"),
            sources=sources,
            query_time_ms=int((time.time() - start_time) * 1000),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

注册路由：

```python
# main.py
from chatbi.api.workflow import router as workflow_router
app.include_router(workflow_router)
```

---

## Part 3: LiteLLM 多模型路由

### Step 9: 配置 LiteLLM

更新 `litellm_config.yaml`：

```yaml
model_list:
  # 主力模型
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  # 快速模型
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # Embedding 模型
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY

  # 可选：添加其他模型
  # - model_name: claude-3-sonnet
  #   litellm_params:
  #     model: anthropic/claude-3-sonnet-20240229
  #     api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  drop_params: true

router_settings:
  routing_strategy: simple-shuffle  # 或 least-busy, usage-based-routing

general_settings:
  master_key: sk-litellm-master-key
```

### Step 10: 创建模型路由模块

创建 `src/chatbi/llm/model_router.py`：

```python
"""模型路由管理"""

from enum import Enum
from functools import lru_cache

from openai import OpenAI

from chatbi.config import get_settings


class TaskType(Enum):
    """任务类型"""
    REASONING = "reasoning"      # 复杂推理
    GENERATION = "generation"    # 内容生成
    EMBEDDING = "embedding"      # 向量化
    SIMPLE = "simple"            # 简单任务


# 任务类型到模型的映射
MODEL_MAPPING = {
    TaskType.REASONING: "gpt-4o",
    TaskType.GENERATION: "gpt-4o-mini",
    TaskType.EMBEDDING: "text-embedding-3-small",
    TaskType.SIMPLE: "gpt-4o-mini",
}


class ModelRouter:
    """模型路由器"""

    def __init__(self):
        self.settings = get_settings()
        self._usage_log = []

    def get_client(self) -> OpenAI:
        """获取 OpenAI 客户端"""
        if self.settings.use_litellm:
            return OpenAI(
                api_key=self.settings.litellm_api_key,
                base_url=self.settings.litellm_base_url,
            )
        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

    def get_model(self, task_type: TaskType) -> str:
        """根据任务类型获取模型"""
        return MODEL_MAPPING.get(task_type, self.settings.model_name)

    def chat_completion(
        self,
        messages: list[dict],
        task_type: TaskType = TaskType.GENERATION,
        **kwargs,
    ) -> str:
        """
        带路由的 Chat Completion

        Args:
            messages: 消息列表
            task_type: 任务类型
            **kwargs: 其他参数
        """
        client = self.get_client()
        model = self.get_model(task_type)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        # 记录使用情况
        self._log_usage(model, response.usage)

        return response.choices[0].message.content or ""

    def get_embedding(self, text: str) -> list[float]:
        """获取 Embedding"""
        client = self.get_client()
        model = self.get_model(TaskType.EMBEDDING)

        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    def _log_usage(self, model: str, usage):
        """记录使用情况"""
        if usage:
            self._usage_log.append({
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            })

    def get_usage_summary(self) -> dict:
        """获取使用汇总"""
        from collections import defaultdict
        summary = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

        for log in self._usage_log:
            model = log["model"]
            summary[model]["calls"] += 1
            summary[model]["prompt_tokens"] += log["prompt_tokens"]
            summary[model]["completion_tokens"] += log["completion_tokens"]

        return dict(summary)


# 全局路由器实例
model_router = ModelRouter()
```

---

## Part 4: LLMOps - Langfuse + Ragas

### Step 11: 集成 Langfuse

```bash
# 更新 pyproject.toml
# dependencies = [
#     ...
#     "langfuse>=2.50.0",
# ]

uv pip install -e ".[dev]"
```

创建 `src/chatbi/observability/__init__.py`：

```python
"""可观测性模块"""
```

创建 `src/chatbi/observability/langfuse_client.py`：

```python
"""Langfuse 集成"""

import os
from functools import wraps
from typing import Any, Callable

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# 初始化 Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    enabled=bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
)


def trace_llm_call(name: str = "llm_call"):
    """
    LLM 调用追踪装饰器

    Usage:
        @trace_llm_call("sql_generation")
        def generate_sql(question: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace = langfuse.trace(name=name)
            try:
                result = func(*args, **kwargs)
                trace.update(output=str(result)[:1000])
                return result
            except Exception as e:
                trace.update(status_message=str(e), level="ERROR")
                raise
            finally:
                langfuse.flush()
        return wrapper
    return decorator


def log_generation(
    name: str,
    input_text: str,
    output_text: str,
    model: str,
    latency_ms: int,
    metadata: dict | None = None,
):
    """记录 LLM 生成"""
    langfuse.generation(
        name=name,
        input=input_text,
        output=output_text,
        model=model,
        metadata=metadata or {},
        usage={
            "latencyMs": latency_ms,
        },
    )
    langfuse.flush()


def log_span(
    name: str,
    input_data: Any,
    output_data: Any,
    metadata: dict | None = None,
):
    """记录 Span"""
    span = langfuse.span(
        name=name,
        input=input_data,
        metadata=metadata or {},
    )
    span.update(output=output_data)
    langfuse.flush()
```

### Step 12: 集成 Ragas 评估

```bash
# 更新 pyproject.toml
# dependencies = [
#     ...
#     "ragas>=0.2.0",
# ]

uv pip install -e ".[dev]"
```

创建 `src/chatbi/evaluation/__init__.py`：

```python
"""评估模块"""
```

创建 `src/chatbi/evaluation/ragas_eval.py`：

```python
"""Ragas 评估"""

import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


def load_eval_dataset(file_path: str) -> list[dict]:
    """
    加载评估数据集

    数据格式：
    [
        {
            "question": "...",
            "ground_truth": "...",
            "contexts": ["..."],  # 可选
        }
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    运行 RAG 评估

    Args:
        questions: 问题列表
        answers: 模型回答列表
        contexts: 检索到的上下文列表
        ground_truths: 标准答案列表
    """
    # 构建数据集
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # 运行评估
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return result.to_pandas().to_dict()


def create_eval_report(results: dict, output_path: str):
    """生成评估报告"""
    import pandas as pd

    df = pd.DataFrame(results)

    # 计算平均分
    summary = df.mean().to_dict()

    report = f"""# RAG 评估报告

## 总体得分

| 指标 | 分数 |
|------|------|
| Faithfulness（忠实度） | {summary.get('faithfulness', 0):.3f} |
| Answer Relevancy（答案相关性） | {summary.get('answer_relevancy', 0):.3f} |
| Context Precision（上下文精确度） | {summary.get('context_precision', 0):.3f} |
| Context Recall（上下文召回率） | {summary.get('context_recall', 0):.3f} |

## 详细结果

{df.to_markdown(index=False)}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 评估报告已保存到: {output_path}")
```

创建评估脚本 `scripts/evaluate_agent.py`：

```python
"""Agent 评估脚本"""

import json
import requests
from pathlib import Path

from chatbi.evaluation.ragas_eval import run_rag_evaluation, create_eval_report


API_BASE_URL = "http://localhost:8000"


def load_test_cases(file_path: str) -> list[dict]:
    """加载测试用例"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_file: str, output_file: str):
    """运行评估"""
    test_cases = load_test_cases(test_file)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"开始评估 {len(test_cases)} 个测试用例...")

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['question'][:50]}...")

        # 调用 API
        response = requests.post(
            f"{API_BASE_URL}/workflow/query",
            json={"question": case["question"]},
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            questions.append(case["question"])
            answers.append(data["answer"])
            contexts.append(case.get("contexts", [data.get("sql", "")]))
            ground_truths.append(case["ground_truth"])
        else:
            print(f"  ❌ 请求失败: {response.text}")

    # 运行 Ragas 评估
    print("\n运行 Ragas 评估...")
    results = run_rag_evaluation(questions, answers, contexts, ground_truths)

    # 生成报告
    create_eval_report(results, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="./data/eval/test_cases.json")
    parser.add_argument("--output", default="./data/eval/report.md")

    args = parser.parse_args()
    run_evaluation(args.test_file, args.output)
```

创建示例测试数据 `data/eval/test_cases.json`：

```json
[
    {
        "question": "什么是 GMV？",
        "ground_truth": "GMV 是成交总额，指一定时间内的成交金额总和，计算时不包含取消和退款订单。",
        "contexts": []
    },
    {
        "question": "用户表有哪些字段？",
        "ground_truth": "用户表包含 id、name、email、created_at、status 等字段。",
        "contexts": []
    },
    {
        "question": "统计活跃用户数量",
        "ground_truth": "需要查询 users 表中 status = 'active' 的用户数量。",
        "contexts": []
    }
]
```

---

## Part 5: 完整集成测试

### Step 13: 测试所有功能

```bash
# 1. 启动所有服务
docker-compose up -d

# 2. 初始化数据
python -m chatbi.sql.init_testdb
python -m chatbi.rag.build_index --docs ./data/docs

# 3. 测试各 API

# RAG 查询
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "GMV 的计算公式是什么？"}'

# SQL 查询
curl -X POST http://localhost:8000/sql/query \
  -H "Content-Type: application/json" \
  -d '{"question": "统计每个用户的订单总金额"}'

# Agent 查询
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "用户留存率是什么？请查询最近的日活数据"}'

# Workflow 查询
curl -X POST http://localhost:8000/workflow/query \
  -H "Content-Type: application/json" \
  -d '{"question": "查询订单状态为已完成的总金额，并解释 GMV 的含义"}'

# 4. 运行评估
python scripts/evaluate_agent.py
```

---

## 验收检查清单

### Part 1: LangChain Agent
- [ ] 工具函数正确定义并可调用
- [ ] Agent 能根据问题选择正确的工具
- [ ] `/agent/query` 接口正常工作

### Part 2: LangGraph Workflow
- [ ] 工作流状态正确传递
- [ ] 意图识别节点工作正常
- [ ] SQL 错误重试机制生效
- [ ] `/workflow/query` 接口正常工作

### Part 3: LiteLLM
- [ ] LiteLLM 服务正常启动
- [ ] 模型路由根据任务类型选择正确模型
- [ ] 使用统计功能可用

### Part 4: LLMOps
- [ ] Langfuse 可以记录调用链路
- [ ] Ragas 评估脚本可运行
- [ ] 评估报告正确生成

---

## 毕业项目完成

恭喜！完成本阶段后，你已经构建了一个完整的智能数据分析平台，包含：

- RAG 文档问答
- Text-to-SQL 数据查询
- 多工具 Agent
- LangGraph 工作流
- 多模型路由
- 可观测性和评估

下一步建议：
1. 完善 Web Demo 界面
2. 添加更多业务文档和测试数据
3. 优化 Prompt 提升准确率
4. 部署到生产环境
