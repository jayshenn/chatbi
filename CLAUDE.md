# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ChatBI 是一个 LLM 工程学习项目，目标是从零构建智能数据分析平台。基于 `docs/llm-engineer-roadmap-for-data-engineer.md` 学习路线图进行开发。

核心学习目标：
- RAG（检索增强生成）- 自然语言问文档
- Text-to-SQL - 自然语言问数据
- 多工具 Agent - 复合分析

## 技术栈

- **后端**: Python + FastAPI
- **Agent 编排**: LangChain v1 + LangGraph
- **RAG**: LlamaIndex
- **向量数据库**: Chroma / Qdrant
- **数据库**: SQLAlchemy
- **模型网关**: LiteLLM
- **可观测性**: Langfuse
- **评估**: Ragas
- **部署**: Docker / Docker Compose

## 常用命令

```bash
# 依赖管理（使用 uv）
uv install

# 启动服务
docker-compose up -d
uvicorn backend.main:app --reload

# 代码检查
ruff check .
black --check .

# 运行测试
pytest
pytest tests/test_xxx.py -v  # 单个测试文件
pytest -k "test_name"        # 按名称匹配
```

## 架构说明

项目采用分层架构：

```
backend/
├── api/          # FastAPI 路由层
├── agents/       # LangChain Agent 定义（多工具编排）
├── workflows/    # LangGraph 工作流（状态机、节点、边）
├── rag/          # RAG 模块（文档解析、向量化、检索）
└── sql/          # Text-to-SQL（schema 管理、SQL 生成、执行）
```

### 核心 API 端点

- `GET /ping` - 健康检查
- `POST /ask_rag` - RAG 文档问答
- `POST /ask_sql` - Text-to-SQL 数据查询
- `POST /ask_agent` - 统一 Agent 入口（自动路由）

### LangGraph 工作流

Agent 使用 LangGraph 编排多步骤流程：
- 意图识别 → 选择工具（RAG/SQL/混合）
- SQL 生成 → 执行 → 错误时自动修正重试
- 结果解释 → 返回

## 开发规范

- 所有 LLM 调用通过 LiteLLM 网关统一管理
- 敏感配置（API Key、数据库连接）从环境变量读取
- Pydantic v2 定义请求/响应模型
- 使用 Langfuse 记录调用链路