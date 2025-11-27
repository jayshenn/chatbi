# 阶段 0（第 0–1 周）现代 Python 工程 + LLM 基础设施

### 要学什么

1. **Python 工程化**

   * 依赖管理工具：`uv`
   * 了解项目结构：

     * `pyproject.toml`
     * `src/`、`tests/`
   * 环境变量管理：

     * 用 `.env` + `python-dotenv` 或直接 docker-compose 环境变量管理 `OPENAI_API_KEY`、`LITELLM_API_KEY` 等。
   * 基础工具：

     * `ruff`：静态检查
     * `black`：代码格式化
     * `pytest`：最基本的单元测试写法

2. **FastAPI + Docker**

   * FastAPI 基础：

     * 创建 `app = FastAPI()`
     * 定义路由：`GET /ping` 返回 `"pong"`；`POST /echo` 返回输入。
   * Docker 基础：

     * 写一个 `Dockerfile` 能把 FastAPI 包进镜像。
     * 用 `docker build` + `docker run` 跑起来。
     * 简单写一个 `docker-compose.yml`（可选）。

3. **LLM 调用基本模式**

   * 使用官方 `openai` SDK 或 `httpx` 调用一个 OpenAI 风格 API（可以是真·OpenAI，也可以是其他兼容服务）。
   * 理解参数：

     * `base_url`：切换不同提供商
     * `api_key`：从环境变量读取
   * 明确「请求结构 → 响应结构」：messages、role、content 等。

4. **LiteLLM 网关（建议启用）**

   * 安装并启动 LiteLLM（本地或 Docker）。
   * 修改 LLM 调用代码，统一通过 LiteLLM 的 `base_url` 调模型。
   * 做一次实验：

     * 改配置（不是改代码）把模型从 A 换成 B，确认功能不受影响。

### 要做什么 / 产出什么

* 一个基础项目骨架（支持 Docker）：

  * FastAPI 服务：

    * `GET /ping`：健康检查
    * `POST /echo`：原样返回请求体
    * `POST /llm-test`：调用 LLM，返回模型的回复（例如复述一句话）。
  * 所有敏感配置（model 名称、base_url、api_key）从 `.env` / 环境变量读取。
* 一个 README：

  * 说明如何 `uv install`、如何 `docker-compose up`、如何测试接口。

---

# 阶段 1（第 1–3 周）Python 实战 + Prompt 工程 + 结构化输出

### 要学什么

1. **Python 强化（贴近数据工程使用场景）**

   * 文件读写：

     * CSV / JSON / Markdown 的读写封装。
   * `pandas`：

     * DataFrame 基础操作：过滤、分组聚合、排序、join。
   * 异步编程：

     * `async/await` 基本语法。
     * FastAPI 中 async 路由的写法。

2. **Prompt 工程基础**

   * Prompt 模板结构：

     * 角色：你是谁（例如“数据分析专家”）。
     * 任务：用户要解决的问题。
     * 约束：格式、风格、语言、禁止事项。
   * Few-shot：

     * 为一个任务准备 2–3 个高质量示例。
   * 要求模型输出 **JSON**：

     * 通过明确说明「输出必须是 JSON，对象结构为 …」。

3. **Pydantic v2 + 结构化输出**

   * Pydantic v2 模型定义：

     * 定义清晰的字段名、类型、枚举。
   * 把 LLM 输出的 JSON：

     * 用 `json.loads` 解析，再用 Pydantic 模型校验。
   * 对异常情况的处理：

     * JSON 解析失败如何重试 / 报错。
     * 缺字段 / 类型不对如何处理。

### 小项目：SQL 解释助手

**目标：** 输入一条复杂 SQL，输出结构化的解释。

1. 学习内容结合项目设计输出结构，比如：

   ```python
   class SqlExplanation(BaseModel):
       target_metrics: list[str]
       filters: list[str]
       group_by: list[str]
       joins: list[str]
       business_explanation: str
   ```

2. 处理流程：

   * 输入：`sql: str` + 可选 sample 数据（CSV）。
   * 调用 LLM：

     * Prompt 中提供 SQL + 要求输出为 `SqlExplanation` 对应 JSON。
   * 使用 Pydantic 校验，返回结构化结果。

3. API / CLI 封装：

   * CLI：`python sql_explainer.py --sql "..."`
   * FastAPI：

     * `POST /explain_sql`：

       * 请求：`{"sql": "...", "sample_data": optional}`
       * 响应：`SqlExplanation` JSON。

### 阶段产出

* 一个 `sql_explainer` 包 / 模块：

  * 可直接 `import` 用于后续项目。
* 一个 FastAPI 接口 + 一个简单 CLI。
* 一组 Prompt 模板 + 对应 Pydantic 模型。

---

# 阶段 2（第 4–6 周）RAG from scratch + LlamaIndex + 向量数据库

## 2.1 手写 RAG（不用框架，掌握底层逻辑）

### 要学什么

1. **文档体系设计**

   * 为你熟悉的数仓/业务领域准备文档：

     * 指标口径说明（如 GMV/DAU/留存等）。
     * 表结构说明（每个表字段含义）。
     * 数仓分层（ODS/DWD/DWS/ADS）设计说明。
   * 文档格式可以是 Markdown、PDF、HTML，内容真实可信即可。

2. **文档解析与切分**

   * 使用 Python：

     * Markdown/HTML 直接解析。
     * PDF 可用 `pypdf` 或其他工具提取文字。
   * 编写切块逻辑：

     * 按章节标题+段落切块。
     * 控制每块长度（按 token 或字符），保留适当 overlap（例如 50–100 token）。

3. **Embedding + 向量检索**

   * 选择一个 OpenAI 风格的 embedding 模型。
   * 先在内存里简单实现：

     * 用 `numpy` 计算余弦相似度。
   * 然后接入向量库：

     * 使用 Chroma / Qdrant 的 Python SDK 或 HTTP API。
     * 实现基本操作：插入向量、相似度检索。

4. **RAG 查询流程**

   * 完整流程：

     * 用户问题 → embedding → top-k 文档检索
     * 构造 Prompt：问题 + 文档片段（带上 source 标识）
     * 调用 LLM 生成答案
   * 控制：

     * 要求回答只基于文档，不要瞎编。
     * 在回答里给出引用的片段 / 文档名。

### 要做什么 / 产出什么

* 数据准备脚本：

  * `prepare_docs.py`：加载文档、切块、写入向量库。
* RAG 查询脚本 / 模块：

  * `rag_query.py`：输入问题，返回：

    * `answer: str`
    * `sources: List[{"doc_id": str, "score": float, "snippet": str}]`
* 在 FastAPI 中添加接口：

  * `POST /ask_rag_raw`：调用手写 RAG 流程。

---

## 2.2 使用 LlamaIndex 重写 RAG

### 要学什么

1. **LlamaIndex 基础概念**

   * Document / Node / Index / QueryEngine。
   * VectorStoreIndex 的创建、持久化、加载。
   * 自定义 text splitter。

2. **把手写 RAG 映射到 LlamaIndex**

   * 将之前的文档加载、切块配置到 LlamaIndex 的 Loader + NodeParser。
   * 向量库改为：LlamaIndex 驱动 Chroma / Qdrant。
   * 自定义 QueryEngine：

     * 控制检索数量。
     * 调整 Prompt（如增加 domain-specific 指令）。

3. **检索结果可解释性**

   * 确保响应中可以拿到：

     * 原始文档 ID/路径。
     * 文本片段（方便前端展示）。
   * 封装统一结构返回给 FastAPI。

### 要做什么 / 产出什么

* 一个 `rag_llamaindex` 模块：

  * `build_index.py`：构建/更新索引。
  * `query_engine.py`：封装 query 接口。
* FastAPI 接口：

  * `POST /ask_rag`：

    * 内部调用 LlamaIndex QueryEngine。
    * 返回答案 + sources。
* 小型 Demo：

  * 在命令行或简单 UI 中验证：

    * 「GMV 指标的计算口径？」
    * 「用户留存相关的事实表名称和主键？」

---

# 阶段 3（第 7–8 周）服务化、Web Demo、Text-to-SQL / SQL Agent 雏形

## 3.1 RAG API 化 + Web Demo

### 要学什么

1. **API 设计与模型定义**

   * 请求/响应 Schema：

     * 请求：`{"question": str}`
     * 响应：`{"answer": str, "sources": [...]}`
   * 用 Pydantic 定义数据模型，方便文档与校验。

2. **Web Demo（Streamlit / Gradio / 简易前端）**

   * 实现一个聊天式界面：

     * 左侧：对话历史。
     * 右侧：显示当前回答引用的文档列表、片段。

3. **基础日志记录**

   * 记录到日志 / 文件 / 数据库：

     * 用户问题。
     * 检索到的文档 id / snippet。
     * LLM 调用耗时。

### 要做什么 / 产出什么

* 完整的 RAG Web 应用：

  * 后端：FastAPI 提供 `/ask_rag`。
  * 前端：Streamlit/Gradio 调用后端 API。
* 一个基本的日志系统，用以后续分析失败案例。

---

## 3.2 Text-to-SQL / SQL Agent 雏形

### 要学什么

1. **Schema 表达 & Prompt 设计**

   * 为一个核心主题域定义 schema：

     * `tables.json` / Markdown 文档，描述每张表：

       * 表名、字段名、类型、含义、主键、外键。
   * Prompt 结构：

     * 输入：schema 描述 + 自然语言问题 + 少量示例（NL → SQL）。
     * 输出约束：

       * 只允许 SELECT。
       * 必须带 LIMIT。
       * 禁止 DDL/DML。

2. **SQL 生成与执行**

   * 使用 LLM：

     * 输出结构化 JSON：`{"sql": "SELECT ...", "explanation": "..."}`。
   * 用 SQLAlchemy：

     * 连接一个数据库（本地 Postgres/MySQL 均可）。
     * 执行只读 SQL，捕获异常。
   * 将查询结果转成 DataFrame，再转成 JSON。

3. **结果解释**

   * 提供 DataFrame 的前几行给 LLM：

     * 让模型生成自然语言解释：

       * 说明指标含义、趋势、对比等。

### 要做什么 / 产出什么

* Text-to-SQL 模块：

  * `generate_sql(question, schema) -> sql`。
  * `execute_sql(sql) -> dataframe`。
  * `explain_result(df, question) -> explanation`.
* FastAPI 接口：

  * `POST /ask_sql`：

    * 请求：`{"question": str}`
    * 响应：`{"sql": str, "rows": [...], "explanation": str}`。
* 前端整合：

  * 在 Web Demo 中增加一个 Tab「问数」，展示：

    * 原问题、生成的 SQL、表格结果、中文解释。

---

# 阶段 4（第 9–12 周）LangChain v1 Agents + LangGraph 编排 + LiteLLM + LLMOps

## 4.1 LangChain v1 Agents + Tools + Messages

### 要学什么

1. **LangChain v1 基础使用**

   * 安装与导入规范：

     * `langchain`：Agent / 工具 / 消息。
     * `langchain-core`：底层抽象（Runnable 等）。
   * 使用 `init_chat_model` 初始化模型（可透传到 LiteLLM 网关）。

2. **定义工具（Tools）**

   * 使用 `@tool` 装饰器，把已有函数包装成工具，例如：

     * `run_sql(sql: str)`：执行只读 SQL。
     * `rag_lookup(question: str)`：查询知识库。
     * `fetch_kpi(name: str, start_date: str, end_date: str)`：预定义指标查询。
   * 为每个工具写清晰的文档字符串，帮助 LLM 正确选择工具。

3. **使用 `create_agent` 构建多工具 Agent**

   * 从 `langchain.agents` 引入 `create_agent`。
   * 构建一个 Agent：

     * system prompt：智能数据分析助手。
     * tools：前面定义的所有工具。
   * 调用模式：

     * `agent.invoke({"messages": [...]})`。
     * 获取最终消息 / 中间 reasoning。

4. **消息与内容块（content_blocks）**

   * 了解 LangChain 的消息模型：

     * HumanMessage / AIMessage / SystemMessage。
   * 理解多模态扩展与 content blocks（了解即可，为未来做准备）。

### 要做什么 / 产出什么

* 一个 `analytics_agent` 模块：

  * 使用 `create_agent` 封装多工具 Agent。
  * 支持根据用户问题自动选择：

    * 仅问文档（走 RAG）。
    * 问数据（走 Text-to-SQL）。
    * 同时查文档 + 查数据（组合工具）。
* FastAPI 接口：

  * `POST /ask_agent`：统一对外的 Agent 接口。
* 用脚本 / Notebook 做一些测试用例：

  * 简单问题（查口径）、
  * 中等问题（单表查询）、
  * 复杂问题（多表 join + 指标解释）。

---

## 4.2 LangGraph：有状态、多步骤 Agent 编排

### 要学什么

1. **LangGraph 概念**

   * State：整个对话/任务的状态（messages、中间 SQL、错误等）。
   * Node：一个个处理单元（如「意图识别」、「生成 SQL」、「执行 SQL」、「结果解释」、「RAG 检索」）。
   * Edge：节点之间的控制流（正常流转、错误回退、结束条件）。

2. **Functional API 实战**

   * 把已有函数包装成节点：

     * `def route_intent(state) -> state`：根据问题判定走「问文档」 or 「问数」 or「混合」。
     * `def generate_sql_node(state) -> state`。
     * `def run_sql_node(state) -> state`。
     * `def rag_node(state) -> state`。
     * `def summarize_node(state) -> state`。
   * 使用 LangGraph 编译成一个图：

     * 定义初始状态结构（Pydantic/TypedDict）。
     * 定义节点和边。
     * 编译为可调用的应用。

3. **错误处理与自纠**

   * 在节点中捕获 SQL 错误 / LLM 输出异常：

     * 标记错误信息到 state。
   * 图中定义错误路径：

     * 遇到 SQL 错误 → 回到 `generate_sql_node`，提示「上一次 SQL 报错信息」，要求模型修正。

4. **对话状态与时间旅行**

   * 在 state 中保留对话历史、最近几次工具调用。
   * 使用 LangGraph 的持久化能力：

     * 以 thread_id（会话 id）为 key 存储状态。
     * 支持回放 / time-travel（可选实践）。

### 要做什么 / 产出什么

* 一个 LangGraph 应用：

  * 包含多个节点的完整智能分析流程图。
  * 支持多轮对话（上下文记忆）。
* FastAPI 集成：

  * 将 LangGraph app 封装为依赖 / 服务，并暴露 `/ask_workflow` 等接口。
* 一份简单的「图结构示意文档」：

  * 节点列表 + 流程图说明（文字或 mermaid 图）。

---

## 4.3 LiteLLM 进阶：多模型路由 + 成本控制

### 要学什么

1. **统一 Chat / Embedding 调用**

   * 将所有 LLM 调用（包括：

     * Chat
     * Embedding
     * 可能的 Rerank）
       统一通过 LiteLLM。

2. **多模型配置**

   * 在 LiteLLM 中配置：

     * 高质量模型（如更强的 GPT / Claude）。
     * 高性价比模型（如一些国产或开源部署模型）。
   * 通过环境变量决定当前使用哪条「模型策略」。

3. **模型路由策略（简单版本）**

   * 在代码中封装一个小组件：

     * 根据任务类型选择模型（例如推理用强模型，embedding 用便宜模型）。
   * 为未来做 A/B Test 留出接口（简单记录「使用了哪个模型」）。

### 要做什么 / 产出什么

* 一个 `model_provider` 模块：

  * 提供统一的 `get_chat_model(task_type)`、`get_embedding_model()` 等函数。
* 在所有调用模型的地方，统一走 `model_provider`。
* 简单记录每次调用使用的模型名，以及大致 token 使用量（配合后面的 Langfuse）。

---

## 4.4 LLMOps：可观测（Langfuse）+ 评估（Ragas）

### 要学什么

1. **Langfuse 可观测性**

   * 部署 Langfuse（本地 Docker 或简单服务器）。
   * 在代码中接入 Langfuse SDK：

     * 为每一次 `/ask_agent` / `/ask_workflow` 调用创建 trace。
     * 为每个 LLM 调用、工具调用记录 span：

       * 输入（可脱敏）。
       * 输出。
       * 时延、token 消耗。

2. **Ragas 评估 RAG / Agent**

   * 准备一个评测集（30–50 条）：

     * 问题、参考答案、参考文档或 SQL。
   * 使用 Ragas:

     * 运行评估指标（如答案相关性、上下文利用度等）。
   * 将评估结果固化：

     * 输出为报表 / Markdown，记录不同版本模型/Prompt 的表现。

### 要做什么 / 产出什么

* 在项目中接入 Langfuse：

  * 可以在 UI 中看到每次调用的完整链路。
* 一个评测脚本：

  * `evaluate_agent.py`：

    * 读取评测集。
    * 调用 `/ask_agent`。
    * 用 Ragas 计算指标，输出报告（CSV/Markdown）。

---

## 4.5 毕业项目：企业级 SQL 智能分析中台

### 最终系统目标

* 项目整体：

  * 后端：Python + FastAPI。
  * Agent & Workflow：LangChain v1 Agents（`create_agent`）+ LangGraph。
  * RAG：手写 RAG + LlamaIndex + 向量库（Chroma/Qdrant）。
  * 数据库：SQLAlchemy + 任意 OLTP/OLAP（MySQL/Postgres/云数仓均可）。
  * 模型层：LiteLLM 接入多家 OpenAI-compatible 模型。
  * 可观测：Langfuse。
  * 评估：Ragas。
  * 部署：Docker / Docker Compose。

### 系统能力

1. **自然语言问文档（RAG）**

   * 问指标口径 / 表结构 / 数仓分层等。
   * 返回回答 + 引用文档。

2. **自然语言问数据（Text-to-SQL）**

   * 根据问题生成 SQL。
   * 执行 SQL 返回结果。
   * 给出自然语言解释和注意事项。

3. **复合分析（多工具 Agent）**

   * Agent 能决定：

     * 先查文档确定概念 → 再生成 SQL。
     * 查询多个时间段 / 维度 → 汇总后输出分析结论。
   * 能在 SQL 报错 / 无数据等情况下自我纠错或给出友好提示。

### 要做什么 / 产出什么

* 一个完整的代码仓库（建议公开到 GitHub）：

  * `backend/`：FastAPI + LangGraph + LangChain + RAG + Text-to-SQL。
  * `frontend/`：Streamlit/Gradio 或简单 SPA。
  * `docker-compose.yml`：

    * 包括 API 服务、向量库、LiteLLM、Langfuse。
* 完整文档：

  * 系统架构图。
  * 各子模块说明。
  * 如何本地一键启动。
  * 示例问题与示例回答。
* 一份简历描述/项目总结：

  * 明确写出：云厂商无绑定、LangChain v1 + LangGraph、LlamaIndex、LiteLLM、Langfuse、Ragas 等关键词。
