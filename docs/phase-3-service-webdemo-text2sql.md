# 阶段 3：服务化、Web Demo、Text-to-SQL / SQL Agent 雏形

> 预计时间：2 周

## 学习目标

- 完善 RAG 服务的 API 设计
- 构建可交互的 Web Demo（Streamlit/Gradio）
- 实现 Text-to-SQL 核心功能
- 添加基础日志记录

## 前置条件

- 完成 [阶段 2](phase-2-rag-llamaindex-vectordb.md)
- RAG 服务已可用

---

## Part 1: RAG API 化 + Web Demo

### Step 1: 完善 API 设计

#### 1.1 统一响应模型

创建 `src/chatbi/models/response.py`：

```python
"""统一响应模型"""

from typing import Any, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """统一 API 响应"""

    success: bool = True
    data: T | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""

    items: list[T]
    total: int
    page: int
    page_size: int
    has_more: bool
```

#### 1.2 创建 RAG 路由模块

创建 `src/chatbi/api/__init__.py`：

```python
"""API 路由模块"""
```

创建 `src/chatbi/api/rag.py`：

```python
"""RAG API 路由"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbi.rag_llamaindex.index import load_llamaindex
from chatbi.rag_llamaindex.query import LlamaIndexQueryEngine

router = APIRouter(prefix="/rag", tags=["RAG"])

# 全局引擎实例
_engine: LlamaIndexQueryEngine | None = None


def get_engine() -> LlamaIndexQueryEngine:
    global _engine
    if _engine is None:
        index = load_llamaindex()
        _engine = LlamaIndexQueryEngine(index)
    return _engine


class RAGQueryRequest(BaseModel):
    """RAG 查询请求"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="返回文档数量")


class SourceDocument(BaseModel):
    """引用文档"""

    content: str
    score: float
    source: str
    chunk_index: int | None = None


class RAGQueryResponse(BaseModel):
    """RAG 查询响应"""

    answer: str
    sources: list[SourceDocument]
    query_time_ms: int


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """
    RAG 文档问答

    根据用户问题检索相关文档，并生成答案。
    """
    start_time = time.time()

    try:
        engine = get_engine()
        result = engine.query(request.question)

        sources = [
            SourceDocument(
                content=s["content"],
                score=s["score"],
                source=s["metadata"].get("file_name", "未知"),
                chunk_index=s["metadata"].get("chunk_index"),
            )
            for s in result.sources
        ]

        query_time_ms = int((time.time() - start_time) * 1000)

        return RAGQueryResponse(
            answer=result.answer,
            sources=sources,
            query_time_ms=query_time_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """RAG 服务健康检查"""
    try:
        engine = get_engine()
        return {"status": "healthy", "index_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

#### 1.3 注册路由

更新 `src/chatbi/main.py`：

```python
from chatbi.api.rag import router as rag_router

# 注册路由
app.include_router(rag_router)
```

### Step 2: 添加日志系统

#### 2.1 配置日志

创建 `src/chatbi/utils/logging.py`：

```python
"""日志配置"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from chatbi.config import get_settings


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    logger = logging.getLogger("chatbi")
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "chatbi") -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


# 初始化日志
setup_logging(log_file="./logs/chatbi.log")
```

#### 2.2 创建查询日志记录

创建 `src/chatbi/utils/query_logger.py`：

```python
"""查询日志记录"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class QueryLog(BaseModel):
    """查询日志"""

    timestamp: str
    query_type: str  # rag, sql, agent
    question: str
    answer: str | None = None
    sources: list[dict] | None = None
    sql: str | None = None
    latency_ms: int
    success: bool
    error: str | None = None
    metadata: dict[str, Any] | None = None


class QueryLogger:
    """查询日志记录器"""

    def __init__(self, log_dir: str = "./logs/queries"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self) -> Path:
        """获取当天日志文件"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"queries_{date_str}.jsonl"

    def log(self, query_log: QueryLog):
        """记录查询日志"""
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(query_log.model_dump_json() + "\n")

    def log_rag_query(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        latency_ms: int,
        success: bool = True,
        error: str | None = None,
    ):
        """记录 RAG 查询"""
        log = QueryLog(
            timestamp=datetime.now().isoformat(),
            query_type="rag",
            question=question,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        self.log(log)

    def log_sql_query(
        self,
        question: str,
        sql: str,
        answer: str | None,
        latency_ms: int,
        success: bool = True,
        error: str | None = None,
    ):
        """记录 SQL 查询"""
        log = QueryLog(
            timestamp=datetime.now().isoformat(),
            query_type="sql",
            question=question,
            sql=sql,
            answer=answer,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        self.log(log)


# 全局日志记录器
query_logger = QueryLogger()
```

### Step 3: 构建 Web Demo

#### 3.1 安装 Streamlit

```bash
# 更新 pyproject.toml
# dependencies = [
#     ...
#     "streamlit>=1.38.0",
# ]

uv pip install -e ".[dev]"
```

#### 3.2 创建 Streamlit 应用

创建 `src/chatbi/web/__init__.py`：

```python
"""Web 界面模块"""
```

创建 `src/chatbi/web/app.py`：

```python
"""Streamlit Web 应用"""

import requests
import streamlit as st

# 配置
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ChatBI - 智能数据分析",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ChatBI - 智能数据分析助手")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    query_type = st.radio(
        "查询类型",
        ["问文档 (RAG)", "问数据 (SQL)"],
        index=0,
    )
    top_k = st.slider("返回文档数", 1, 10, 3)

    st.divider()
    st.markdown("### 使用说明")
    st.markdown("""
    - **问文档**：查询指标定义、表结构等文档内容
    - **问数据**：通过自然语言查询数据库
    """)

# 主界面
tab1, tab2 = st.tabs(["💬 对话", "📜 历史记录"])

with tab1:
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 引用来源"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['source']}** (相关度: {source['score']:.2f})")
                        st.markdown(f"> {source['content']}")
                        st.divider()

    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 API
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    if "RAG" in query_type:
                        response = requests.post(
                            f"{API_BASE_URL}/rag/query",
                            json={"question": prompt, "top_k": top_k},
                            timeout=60,
                        )
                    else:
                        response = requests.post(
                            f"{API_BASE_URL}/sql/query",
                            json={"question": prompt},
                            timeout=60,
                        )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("📚 引用来源"):
                                for source in sources:
                                    st.markdown(f"**{source['source']}** (相关度: {source['score']:.2f})")
                                    st.markdown(f"> {source['content']}")
                                    st.divider()

                        # 保存到历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    else:
                        st.error(f"请求失败: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("无法连接到后端服务，请确保 API 服务已启动")
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")

with tab2:
    st.markdown("### 查询历史")
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"**Q{i//2 + 1}:** {msg['content']}")
            else:
                st.markdown(f"**A{i//2 + 1}:** {msg['content'][:200]}...")
                st.divider()
    else:
        st.info("暂无查询历史")

    if st.button("清空历史"):
        st.session_state.messages = []
        st.rerun()
```

#### 3.3 启动 Web Demo

```bash
# 先启动后端 API
uvicorn chatbi.main:app --reload &

# 启动 Streamlit
streamlit run src/chatbi/web/app.py
```

访问 http://localhost:8501

---

## Part 2: Text-to-SQL / SQL Agent 雏形

### Step 4: 准备数据库环境

#### 4.1 添加数据库依赖

```bash
# 更新 pyproject.toml
# dependencies = [
#     ...
#     "sqlalchemy>=2.0.0",
#     "psycopg2-binary>=2.9.0",  # PostgreSQL
#     # 或 "pymysql>=1.1.0",     # MySQL
# ]

uv pip install -e ".[dev]"
```

#### 4.2 创建测试数据库（使用 SQLite 简化）

创建 `src/chatbi/sql/__init__.py`：

```python
"""SQL 模块"""
```

创建 `src/chatbi/sql/database.py`：

```python
"""数据库连接管理"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from chatbi.config import get_settings


def get_database_url() -> str:
    """获取数据库连接 URL"""
    settings = get_settings()
    # 默认使用 SQLite
    return getattr(settings, "database_url", "sqlite:///./data/chatbi.db")


engine = create_engine(get_database_url(), echo=False)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def get_db_session():
    """获取数据库会话"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_query(sql: str) -> list[dict]:
    """
    执行只读 SQL 查询

    Args:
        sql: SQL 查询语句

    Returns:
        查询结果列表
    """
    with get_db_session() as session:
        result = session.execute(text(sql))
        columns = result.keys()
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]
```

#### 4.3 创建测试数据

创建 `src/chatbi/sql/init_testdb.py`：

```python
"""初始化测试数据库"""

from sqlalchemy import text
from chatbi.sql.database import engine


def init_test_database():
    """初始化测试数据库和数据"""

    with engine.connect() as conn:
        # 创建用户表
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(200),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active'
            )
        """))

        # 创建订单表
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                amount DECIMAL(10, 2),
                status VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))

        # 创建用户行为表
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_events (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                event_type VARCHAR(50),
                event_date DATE,
                page_url VARCHAR(500),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))

        # 插入测试数据
        conn.execute(text("""
            INSERT OR IGNORE INTO users (id, name, email, status) VALUES
            (1, '张三', 'zhangsan@example.com', 'active'),
            (2, '李四', 'lisi@example.com', 'active'),
            (3, '王五', 'wangwu@example.com', 'inactive'),
            (4, '赵六', 'zhaoliu@example.com', 'active'),
            (5, '钱七', 'qianqi@example.com', 'active')
        """))

        conn.execute(text("""
            INSERT OR IGNORE INTO orders (id, user_id, amount, status, created_at) VALUES
            (1, 1, 100.00, 'completed', '2024-01-15 10:00:00'),
            (2, 1, 200.00, 'completed', '2024-01-16 11:00:00'),
            (3, 2, 150.00, 'completed', '2024-01-15 12:00:00'),
            (4, 2, 300.00, 'cancelled', '2024-01-17 09:00:00'),
            (5, 3, 250.00, 'completed', '2024-01-18 14:00:00'),
            (6, 4, 180.00, 'pending', '2024-01-19 16:00:00'),
            (7, 1, 120.00, 'completed', '2024-01-20 10:00:00'),
            (8, 5, 400.00, 'completed', '2024-01-21 11:00:00')
        """))

        conn.execute(text("""
            INSERT OR IGNORE INTO user_events (id, user_id, event_type, event_date, page_url) VALUES
            (1, 1, 'login', '2024-01-15', '/home'),
            (2, 1, 'page_view', '2024-01-15', '/products'),
            (3, 2, 'login', '2024-01-15', '/home'),
            (4, 2, 'click', '2024-01-15', '/products/1'),
            (5, 3, 'login', '2024-01-16', '/home'),
            (6, 1, 'login', '2024-01-16', '/home'),
            (7, 4, 'login', '2024-01-16', '/home'),
            (8, 5, 'login', '2024-01-17', '/home')
        """))

        conn.commit()

    print("✅ 测试数据库初始化完成")


if __name__ == "__main__":
    init_test_database()
```

运行初始化：

```bash
python -m chatbi.sql.init_testdb
```

### Step 5: Schema 表达与 Prompt 设计

#### 5.1 创建 Schema 管理

创建 `src/chatbi/sql/schema.py`：

```python
"""数据库 Schema 管理"""

from pydantic import BaseModel


class ColumnInfo(BaseModel):
    """列信息"""

    name: str
    type: str
    description: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: str | None = None


class TableInfo(BaseModel):
    """表信息"""

    name: str
    description: str
    columns: list[ColumnInfo]


class DatabaseSchema(BaseModel):
    """数据库 Schema"""

    tables: list[TableInfo]

    def to_prompt_text(self) -> str:
        """转换为 Prompt 文本"""
        lines = []
        for table in self.tables:
            lines.append(f"## 表: {table.name}")
            lines.append(f"描述: {table.description}")
            lines.append("")
            lines.append("| 字段名 | 类型 | 说明 | 备注 |")
            lines.append("|--------|------|------|------|")
            for col in table.columns:
                notes = []
                if col.is_primary_key:
                    notes.append("主键")
                if col.is_foreign_key:
                    notes.append(f"外键->{col.foreign_key_ref}")
                note_str = ", ".join(notes) if notes else "-"
                lines.append(f"| {col.name} | {col.type} | {col.description} | {note_str} |")
            lines.append("")
        return "\n".join(lines)


# 预定义 Schema
CHATBI_SCHEMA = DatabaseSchema(
    tables=[
        TableInfo(
            name="users",
            description="用户信息表",
            columns=[
                ColumnInfo(name="id", type="INTEGER", description="用户ID", is_primary_key=True),
                ColumnInfo(name="name", type="VARCHAR", description="用户名"),
                ColumnInfo(name="email", type="VARCHAR", description="邮箱"),
                ColumnInfo(name="created_at", type="TIMESTAMP", description="注册时间"),
                ColumnInfo(name="status", type="VARCHAR", description="状态: active/inactive"),
            ],
        ),
        TableInfo(
            name="orders",
            description="订单表",
            columns=[
                ColumnInfo(name="id", type="INTEGER", description="订单ID", is_primary_key=True),
                ColumnInfo(
                    name="user_id",
                    type="INTEGER",
                    description="用户ID",
                    is_foreign_key=True,
                    foreign_key_ref="users.id",
                ),
                ColumnInfo(name="amount", type="DECIMAL", description="订单金额"),
                ColumnInfo(name="status", type="VARCHAR", description="状态: pending/completed/cancelled"),
                ColumnInfo(name="created_at", type="TIMESTAMP", description="下单时间"),
            ],
        ),
        TableInfo(
            name="user_events",
            description="用户行为事件表",
            columns=[
                ColumnInfo(name="id", type="INTEGER", description="事件ID", is_primary_key=True),
                ColumnInfo(
                    name="user_id",
                    type="INTEGER",
                    description="用户ID",
                    is_foreign_key=True,
                    foreign_key_ref="users.id",
                ),
                ColumnInfo(name="event_type", type="VARCHAR", description="事件类型: login/page_view/click"),
                ColumnInfo(name="event_date", type="DATE", description="事件日期"),
                ColumnInfo(name="page_url", type="VARCHAR", description="页面URL"),
            ],
        ),
    ]
)
```

#### 5.2 创建 Text-to-SQL Prompt

创建 `src/chatbi/prompts/text_to_sql.py`：

```python
"""Text-to-SQL Prompt"""

from chatbi.prompts.templates import PromptTemplate

TEXT_TO_SQL_SYSTEM = """你是一个专业的 SQL 专家，负责将用户的自然语言问题转换为 SQL 查询。

## 数据库 Schema
{schema}

## 规则
1. 只生成 SELECT 查询，禁止 INSERT/UPDATE/DELETE/DROP 等操作
2. 查询必须带 LIMIT 限制（默认 100）
3. 使用清晰的别名让结果更易读
4. 如果问题不明确，做出合理假设并在解释中说明

## 输出格式
输出必须是 JSON 格式：
```json
{{
    "sql": "SELECT ...",
    "explanation": "这个查询做了什么..."
}}
```
"""

TEXT_TO_SQL_USER = PromptTemplate(
    """请将以下问题转换为 SQL 查询：

问题：${question}

${examples_section}

请直接输出 JSON，不要有其他内容。
"""
)

# Few-shot 示例
TEXT_TO_SQL_EXAMPLES = """
## 示例

问题：查询所有活跃用户
```json
{
    "sql": "SELECT id, name, email FROM users WHERE status = 'active' LIMIT 100",
    "explanation": "查询状态为 active 的用户基本信息"
}
```

问题：统计每个用户的订单总金额
```json
{
    "sql": "SELECT u.name, SUM(o.amount) as total_amount FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' GROUP BY u.id, u.name ORDER BY total_amount DESC LIMIT 100",
    "explanation": "关联用户表和订单表，统计每个用户的已完成订单总金额，按金额降序排列"
}
```

问题：统计每天的活跃用户数
```json
{
    "sql": "SELECT event_date, COUNT(DISTINCT user_id) as dau FROM user_events GROUP BY event_date ORDER BY event_date DESC LIMIT 100",
    "explanation": "按日期分组，统计每天的独立活跃用户数（DAU）"
}
```
"""


def build_text_to_sql_prompt(question: str, schema_text: str, include_examples: bool = True) -> tuple[str, str]:
    """
    构建 Text-to-SQL Prompt

    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = TEXT_TO_SQL_SYSTEM.format(schema=schema_text)

    examples_section = TEXT_TO_SQL_EXAMPLES if include_examples else ""
    user_prompt = TEXT_TO_SQL_USER.format(
        question=question,
        examples_section=examples_section,
    )

    return system_prompt, user_prompt
```

### Step 6: 实现 Text-to-SQL 服务

#### 6.1 创建数据模型

创建 `src/chatbi/models/sql_query.py`：

```python
"""SQL 查询数据模型"""

from pydantic import BaseModel, Field


class GeneratedSQL(BaseModel):
    """生成的 SQL"""

    sql: str = Field(..., description="SQL 查询语句")
    explanation: str = Field(..., description="SQL 解释")


class SQLQueryResult(BaseModel):
    """SQL 查询结果"""

    sql: str
    explanation: str
    rows: list[dict]
    row_count: int
    result_explanation: str | None = None
```

#### 6.2 创建 Text-to-SQL 服务

创建 `src/chatbi/services/text_to_sql.py`：

```python
"""Text-to-SQL 服务"""

import pandas as pd

from chatbi.llm import chat_completion
from chatbi.models.sql_query import GeneratedSQL, SQLQueryResult
from chatbi.prompts.text_to_sql import build_text_to_sql_prompt
from chatbi.sql.database import execute_query
from chatbi.sql.schema import CHATBI_SCHEMA
from chatbi.utils.json_parser import JsonParseError, parse_json_response


class TextToSQLService:
    """Text-to-SQL 服务"""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.schema = CHATBI_SCHEMA

    def generate_sql(self, question: str) -> GeneratedSQL:
        """
        生成 SQL

        Args:
            question: 自然语言问题

        Returns:
            生成的 SQL 和解释
        """
        system_prompt, user_prompt = build_text_to_sql_prompt(
            question=question,
            schema_text=self.schema.to_prompt_text(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = chat_completion(messages, temperature=0.1)
                return parse_json_response(response, GeneratedSQL)
            except JsonParseError as e:
                last_error = e
                if attempt < self.max_retries:
                    messages.append({"role": "assistant", "content": e.raw_content})
                    messages.append({
                        "role": "user",
                        "content": "输出格式不正确，请重新输出有效的 JSON 格式。",
                    })

        raise last_error

    def execute_sql(self, sql: str) -> list[dict]:
        """
        执行 SQL（只读）

        Args:
            sql: SQL 查询语句

        Returns:
            查询结果
        """
        # 安全检查
        sql_upper = sql.upper()
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for word in forbidden:
            if word in sql_upper:
                raise ValueError(f"不允许执行 {word} 操作")

        return execute_query(sql)

    def explain_result(self, question: str, sql: str, df: pd.DataFrame) -> str:
        """
        解释查询结果

        Args:
            question: 原始问题
            sql: 执行的 SQL
            df: 查询结果 DataFrame

        Returns:
            自然语言解释
        """
        # 准备数据摘要
        if len(df) == 0:
            data_summary = "查询结果为空"
        else:
            data_summary = f"查询返回 {len(df)} 条记录\n\n前几行数据：\n{df.head(5).to_markdown(index=False)}"

        prompt = f"""请根据以下信息，用简洁的中文解释查询结果：

## 用户问题
{question}

## 执行的 SQL
```sql
{sql}
```

## 查询结果
{data_summary}

请给出简洁的分析和结论（2-3句话）：
"""

        messages = [{"role": "user", "content": prompt}]
        return chat_completion(messages, temperature=0.3)

    def query(self, question: str) -> SQLQueryResult:
        """
        完整的 Text-to-SQL 查询流程

        Args:
            question: 自然语言问题

        Returns:
            查询结果
        """
        # 1. 生成 SQL
        generated = self.generate_sql(question)

        # 2. 执行 SQL
        rows = self.execute_sql(generated.sql)

        # 3. 解释结果
        df = pd.DataFrame(rows)
        result_explanation = self.explain_result(question, generated.sql, df)

        return SQLQueryResult(
            sql=generated.sql,
            explanation=generated.explanation,
            rows=rows,
            row_count=len(rows),
            result_explanation=result_explanation,
        )


# 默认服务实例
text_to_sql_service = TextToSQLService()
```

### Step 7: 添加 SQL API

创建 `src/chatbi/api/sql.py`：

```python
"""SQL API 路由"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbi.services.text_to_sql import text_to_sql_service
from chatbi.utils.json_parser import JsonParseError
from chatbi.utils.query_logger import query_logger

router = APIRouter(prefix="/sql", tags=["SQL"])


class SQLQueryRequest(BaseModel):
    """SQL 查询请求"""

    question: str = Field(..., min_length=1, max_length=500, description="自然语言问题")


class SQLQueryResponse(BaseModel):
    """SQL 查询响应"""

    sql: str
    explanation: str
    rows: list[dict]
    row_count: int
    result_explanation: str | None
    query_time_ms: int


@router.post("/query", response_model=SQLQueryResponse)
async def query_sql(request: SQLQueryRequest):
    """
    自然语言转 SQL 查询

    将用户的自然语言问题转换为 SQL，执行查询并返回结果。
    """
    start_time = time.time()

    try:
        result = text_to_sql_service.query(request.question)
        latency_ms = int((time.time() - start_time) * 1000)

        # 记录日志
        query_logger.log_sql_query(
            question=request.question,
            sql=result.sql,
            answer=result.result_explanation,
            latency_ms=latency_ms,
        )

        return SQLQueryResponse(
            sql=result.sql,
            explanation=result.explanation,
            rows=result.rows,
            row_count=result.row_count,
            result_explanation=result.result_explanation,
            query_time_ms=latency_ms,
        )

    except JsonParseError as e:
        raise HTTPException(status_code=400, detail=f"SQL 生成失败: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询执行失败: {str(e)}")


class GenerateSQLRequest(BaseModel):
    """生成 SQL 请求（不执行）"""

    question: str


class GenerateSQLResponse(BaseModel):
    """生成 SQL 响应"""

    sql: str
    explanation: str


@router.post("/generate", response_model=GenerateSQLResponse)
async def generate_sql(request: GenerateSQLRequest):
    """仅生成 SQL，不执行"""
    try:
        result = text_to_sql_service.generate_sql(request.question)
        return GenerateSQLResponse(sql=result.sql, explanation=result.explanation)
    except JsonParseError as e:
        raise HTTPException(status_code=400, detail=f"SQL 生成失败: {str(e)}")
```

注册路由到 `main.py`：

```python
from chatbi.api.sql import router as sql_router
app.include_router(sql_router)
```

### Step 8: 更新 Web Demo

更新 `src/chatbi/web/app.py`，添加 SQL 查询 Tab：

```python
# 在现有代码基础上，添加 SQL 查询功能
# 当用户选择 "问数据 (SQL)" 时，调用 /sql/query 接口

# 显示 SQL 结果时，额外展示生成的 SQL 和数据表格
if "SQL" in query_type:
    response = requests.post(
        f"{API_BASE_URL}/sql/query",
        json={"question": prompt},
        timeout=60,
    )
    if response.status_code == 200:
        data = response.json()
        st.markdown(data.get("result_explanation", ""))

        with st.expander("📝 生成的 SQL"):
            st.code(data.get("sql", ""), language="sql")
            st.markdown(f"*{data.get('explanation', '')}*")

        if data.get("rows"):
            st.markdown("### 查询结果")
            st.dataframe(data["rows"])
```

### Step 9: 测试 Text-to-SQL

```bash
# 1. 初始化测试数据库
python -m chatbi.sql.init_testdb

# 2. 启动服务
uvicorn chatbi.main:app --reload

# 3. 测试 SQL 生成
curl -X POST http://localhost:8000/sql/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "查询所有活跃用户"}'

# 4. 测试完整查询
curl -X POST http://localhost:8000/sql/query \
  -H "Content-Type: application/json" \
  -d '{"question": "统计每个用户的订单总金额，按金额降序排列"}'

curl -X POST http://localhost:8000/sql/query \
  -H "Content-Type: application/json" \
  -d '{"question": "查询最近的日活跃用户数"}'
```

---

## 验收检查清单

### Part 1: RAG API 化 + Web Demo
- [ ] `/rag/query` 接口正常工作
- [ ] 响应包含答案、来源和查询耗时
- [ ] 日志记录功能正常
- [ ] Streamlit Web Demo 可以访问
- [ ] 对话历史正确显示

### Part 2: Text-to-SQL
- [ ] 测试数据库初始化成功
- [ ] `/sql/generate` 可以生成 SQL
- [ ] `/sql/query` 可以执行查询并返回结果
- [ ] SQL 安全检查生效（禁止 DDL/DML）
- [ ] 结果解释正确生成
- [ ] Web Demo 可以展示 SQL 和数据表格

---

## 下一步

完成本阶段后，进入 [阶段 4：LangChain Agents + LangGraph + LLMOps](phase-4-langchain-langgraph-llmops.md)
