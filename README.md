# ChatBI

基于自然语言的数据分析学习项目 - Text-to-SQL、RAG、AI Agents

## 项目简介

ChatBI 是一个 LLM 工程学习项目，目标是从零构建一个智能数据分析平台。通过这个项目，学习和实践 RAG（检索增强生成）、Text-to-SQL、AI Agent 等 LLM 应用开发技术。

> 本项目基于 [LLM Engineer Roadmap](docs/llm-engineer-roadmap-for-data-engineer.md) 学习路线图进行开发。

## 核心功能

### 1. 自然语言问文档（RAG）
- 查询指标口径、表结构、数仓分层等文档
- 返回精准答案并附带引用来源

### 2. 自然语言问数据（Text-to-SQL）
- 根据自然语言问题自动生成 SQL
- 执行查询并返回结构化结果
- 提供自然语言解释和数据洞察

### 3. 复合分析（多工具 Agent）
- 智能判断查询意图，自动选择最优工具
- 支持先查文档确定概念，再生成 SQL 查询
- 支持多维度、多时间段的组合分析
- 具备自我纠错能力

## 技术栈

| 类别 | 技术选型 |
|------|----------|
| 后端框架 | Python + FastAPI |
| Agent 编排 | LangChain v1 + LangGraph |
| RAG 框架 | LlamaIndex |
| 向量数据库 | Chroma / Qdrant |
| 数据库连接 | SQLAlchemy |
| 模型网关 | LiteLLM（支持多模型路由） |
| 可观测性 | Langfuse |
| 评估框架 | Ragas |
| 容器化 | Docker / Docker Compose |

## 项目结构

```
chatbi/
├── backend/          # FastAPI 后端服务
│   ├── api/          # API 路由
│   ├── agents/       # LangChain Agent 定义
│   ├── workflows/    # LangGraph 工作流
│   ├── rag/          # RAG 相关模块
│   └── sql/          # Text-to-SQL 模块
├── frontend/         # 前端界面（Streamlit/Gradio）
├── docs/             # 项目文档
├── tests/            # 测试用例
└── docker-compose.yml
```

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- 任意 OpenAI 兼容的 LLM API

### 安装依赖

```bash
# 使用 uv 安装依赖
uv install
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，配置以下变量：
# - OPENAI_API_KEY / LITELLM_API_KEY
# - DATABASE_URL
# - VECTOR_DB_URL
```

### 启动服务

```bash
# 使用 Docker Compose 一键启动
docker-compose up -d

# 或本地开发模式
uvicorn backend.main:app --reload
```

### 测试接口

```bash
# 健康检查
curl http://localhost:8000/ping

# 问文档（RAG）
curl -X POST http://localhost:8000/ask_rag \
  -H "Content-Type: application/json" \
  -d '{"question": "GMV 指标的计算口径是什么？"}'

# 问数据（Text-to-SQL）
curl -X POST http://localhost:8000/ask_sql \
  -H "Content-Type: application/json" \
  -d '{"question": "查询最近7天的日活跃用户数"}'

# Agent 统一入口
curl -X POST http://localhost:8000/ask_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "用户留存相关的表有哪些？请查询最近一周的次日留存率"}'
```

## API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 学习路线图

本项目配套详细的分阶段实操指南，按顺序学习即可完成整个项目：

| 阶段 | 主题 | 预计时间 |
|------|------|----------|
| [阶段 0](docs/phase-0-python-engineering-llm-infra.md) | Python 工程化 + LLM 基础设施 | 1-2 周 |
| [阶段 1](docs/phase-1-python-prompt-structured-output.md) | Prompt 工程 + 结构化输出 | 2-3 周 |
| [阶段 2](docs/phase-2-rag-llamaindex-vectordb.md) | RAG + LlamaIndex + 向量数据库 | 3 周 |
| [阶段 3](docs/phase-3-service-webdemo-text2sql.md) | 服务化 + Web Demo + Text-to-SQL | 2 周 |
| [阶段 4](docs/phase-4-langchain-langgraph-llmops.md) | LangChain Agents + LangGraph + LLMOps | 4 周 |

> 完整学习大纲：[LLM Engineer Roadmap](docs/llm-engineer-roadmap-for-data-engineer.md)

## License

MIT
