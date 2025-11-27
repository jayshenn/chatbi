# 阶段 1：Python 实战 + Prompt 工程 + 结构化输出

> 预计时间：2-3 周

## 学习目标

- 强化 Python 数据处理能力（文件读写、pandas、异步）
- 掌握 Prompt 工程基础技巧
- 使用 Pydantic v2 实现 LLM 结构化输出
- 完成小项目：SQL 解释助手

## 前置条件

- 完成 [阶段 0](phase-0-python-engineering-llm-infra.md)
- 项目基础框架已搭建完成

---

## Step 1: Python 数据处理强化

### 1.1 添加依赖

更新 `pyproject.toml`：

```toml
dependencies = [
    # ... 已有依赖
    "pandas>=2.2.0",
    "aiofiles>=24.1.0",
]
```

安装：

```bash
uv pip install -e ".[dev]"
```

### 1.2 创建数据处理工具模块

创建 `src/chatbi/utils/data.py`：

```python
"""数据处理工具"""

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(file_path: str | Path) -> pd.DataFrame:
    """读取 CSV 文件"""
    return pd.read_csv(file_path)


def read_json(file_path: str | Path) -> dict | list:
    """读取 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """写入 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_markdown(file_path: str | Path) -> str:
    """读取 Markdown 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    """DataFrame 转 Markdown 表格"""
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def dataframe_summary(df: pd.DataFrame) -> dict:
    """生成 DataFrame 摘要信息"""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample": df.head(3).to_dict(orient="records"),
    }
```

### 1.3 Pandas 常用操作示例

创建 `src/chatbi/utils/pandas_examples.py`（仅作学习参考）：

```python
"""Pandas 常用操作示例"""

import pandas as pd


def demo_pandas_operations():
    """演示常用 Pandas 操作"""

    # 创建示例数据
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "user_id": [1, 2, 1, 3, 2, 1, 4, 3, 2, 1],
            "amount": [100, 200, 150, 300, 250, 120, 180, 220, 190, 160],
            "category": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
        }
    )

    # 1. 过滤
    filtered = df[df["amount"] > 150]

    # 2. 分组聚合
    grouped = df.groupby("category").agg(
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        count=("amount", "count"),
    )

    # 3. 排序
    sorted_df = df.sort_values("amount", ascending=False)

    # 4. 多条件过滤
    complex_filter = df[(df["amount"] > 100) & (df["category"].isin(["A", "B"]))]

    # 5. 透视表
    pivot = df.pivot_table(
        values="amount", index="category", columns="user_id", aggfunc="sum", fill_value=0
    )

    return {
        "original": df,
        "filtered": filtered,
        "grouped": grouped,
        "sorted": sorted_df,
        "complex_filter": complex_filter,
        "pivot": pivot,
    }
```

### 1.4 异步编程基础

创建 `src/chatbi/utils/async_utils.py`：

```python
"""异步工具"""

import asyncio
from typing import Any, Callable, Coroutine

import aiofiles


async def read_file_async(file_path: str) -> str:
    """异步读取文件"""
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        return await f.read()


async def write_file_async(file_path: str, content: str) -> None:
    """异步写入文件"""
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(content)


async def gather_with_concurrency(
    n: int, *coros: Coroutine[Any, Any, Any]
) -> list[Any]:
    """
    限制并发数量的 gather

    Args:
        n: 最大并发数
        *coros: 协程列表
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: Coroutine[Any, Any, Any]) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
```

---

## Step 2: Prompt 工程基础

### 2.1 创建 Prompt 模板模块

创建 `src/chatbi/prompts/__init__.py`：

```python
"""Prompt 模板模块"""

from chatbi.prompts.templates import PromptTemplate

__all__ = ["PromptTemplate"]
```

创建 `src/chatbi/prompts/templates.py`：

```python
"""Prompt 模板定义"""

from string import Template
from typing import Any


class PromptTemplate:
    """
    Prompt 模板类

    支持使用 $variable 或 ${variable} 语法进行变量替换
    """

    def __init__(self, template: str):
        self.template = Template(template)
        self._raw = template

    def format(self, **kwargs: Any) -> str:
        """格式化模板"""
        return self.template.safe_substitute(**kwargs)

    def __str__(self) -> str:
        return self._raw


# ========== 通用 Prompt 模板 ==========

SYSTEM_ROLE_TEMPLATE = PromptTemplate(
    """你是一个${role}。

## 你的职责
${responsibilities}

## 约束条件
${constraints}
"""
)

JSON_OUTPUT_TEMPLATE = PromptTemplate(
    """请根据以下要求生成 JSON 格式的输出。

## 任务描述
${task}

## 输入内容
${input}

## 输出格式要求
输出必须是有效的 JSON，结构如下：
```json
${schema}
```

## 注意事项
- 只输出 JSON，不要有其他内容
- 确保所有字段都有值
- 字符串值使用双引号
"""
)

FEW_SHOT_TEMPLATE = PromptTemplate(
    """${task_description}

## 示例

${examples}

## 现在请处理以下输入

输入：${input}
输出："""
)
```

### 2.2 SQL 解释助手 Prompt

创建 `src/chatbi/prompts/sql_explainer.py`：

```python
"""SQL 解释助手 Prompt"""

from chatbi.prompts.templates import PromptTemplate

SQL_EXPLAINER_SYSTEM = """你是一个资深的数据分析专家，擅长解读复杂的 SQL 查询。

## 你的职责
- 分析 SQL 查询的结构和逻辑
- 提取查询的关键信息（指标、过滤条件、分组、关联等）
- 用业务语言解释 SQL 的含义

## 输出要求
- 输出必须是严格的 JSON 格式
- 所有字段必须填写，如果没有相关内容则使用空数组 []
- business_explanation 必须用中文，简洁明了
"""

SQL_EXPLAINER_USER = PromptTemplate(
    """请分析以下 SQL 查询，并以 JSON 格式输出分析结果。

## SQL 查询
```sql
${sql}
```

${sample_data_section}

## 输出 JSON 结构
```json
{
    "target_metrics": ["查询要获取的指标或字段，如 SUM(amount), COUNT(*)"],
    "filters": ["WHERE 条件中的过滤逻辑"],
    "group_by": ["GROUP BY 的字段"],
    "joins": ["表关联信息，格式：表A JOIN 表B ON 条件"],
    "order_by": ["排序信息"],
    "business_explanation": "用业务语言解释这个查询在做什么"
}
```

请直接输出 JSON，不要有其他内容。
"""
)


def build_sql_explainer_prompt(sql: str, sample_data: str | None = None) -> str:
    """
    构建 SQL 解释 Prompt

    Args:
        sql: SQL 查询语句
        sample_data: 可选的示例数据（CSV 格式）
    """
    sample_section = ""
    if sample_data:
        sample_section = f"""
## 参考数据样例
```csv
{sample_data}
```
"""

    return SQL_EXPLAINER_USER.format(sql=sql, sample_data_section=sample_section)
```

---

## Step 3: Pydantic v2 结构化输出

### 3.1 创建 SQL 解释模型

创建 `src/chatbi/models/__init__.py`：

```python
"""数据模型"""

from chatbi.models.sql import SqlExplanation

__all__ = ["SqlExplanation"]
```

创建 `src/chatbi/models/sql.py`：

```python
"""SQL 相关数据模型"""

from pydantic import BaseModel, Field


class SqlExplanation(BaseModel):
    """SQL 解释结果"""

    target_metrics: list[str] = Field(
        default_factory=list,
        description="查询要获取的指标或字段",
    )
    filters: list[str] = Field(
        default_factory=list,
        description="WHERE 条件中的过滤逻辑",
    )
    group_by: list[str] = Field(
        default_factory=list,
        description="GROUP BY 的字段",
    )
    joins: list[str] = Field(
        default_factory=list,
        description="表关联信息",
    )
    order_by: list[str] = Field(
        default_factory=list,
        description="排序信息",
    )
    business_explanation: str = Field(
        default="",
        description="业务语言解释",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_metrics": ["SUM(order_amount) as total_gmv", "COUNT(DISTINCT user_id) as user_count"],
                "filters": ["order_date >= '2024-01-01'", "status = 'completed'"],
                "group_by": ["DATE(order_date)", "category"],
                "joins": ["orders JOIN users ON orders.user_id = users.id"],
                "order_by": ["total_gmv DESC"],
                "business_explanation": "统计2024年以来各品类每天的GMV和下单用户数，按GMV降序排列",
            }
        }
    }
```

### 3.2 创建 JSON 解析工具

创建 `src/chatbi/utils/json_parser.py`：

```python
"""JSON 解析工具"""

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class JsonParseError(Exception):
    """JSON 解析错误"""

    def __init__(self, message: str, raw_content: str):
        super().__init__(message)
        self.raw_content = raw_content


def extract_json_from_text(text: str) -> str:
    """
    从文本中提取 JSON 内容

    处理以下情况：
    1. 纯 JSON 文本
    2. Markdown 代码块中的 JSON
    3. 带有前后说明文字的 JSON
    """
    # 尝试匹配 ```json ... ``` 代码块
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_block_pattern, text)
    if matches:
        return matches[0].strip()

    # 尝试匹配 { ... } 或 [ ... ]
    # 找到第一个 { 或 [ 和最后一个 } 或 ]
    text = text.strip()

    # 查找 JSON 对象
    obj_start = text.find("{")
    obj_end = text.rfind("}")

    # 查找 JSON 数组
    arr_start = text.find("[")
    arr_end = text.rfind("]")

    # 选择更早出现的起始符号
    if obj_start >= 0 and (arr_start < 0 or obj_start < arr_start):
        if obj_end > obj_start:
            return text[obj_start : obj_end + 1]
    elif arr_start >= 0:
        if arr_end > arr_start:
            return text[arr_start : arr_end + 1]

    return text


def parse_json_response(text: str, model: Type[T]) -> T:
    """
    解析 LLM 返回的 JSON 并校验

    Args:
        text: LLM 返回的原始文本
        model: Pydantic 模型类

    Returns:
        解析并校验后的模型实例

    Raises:
        JsonParseError: JSON 解析或校验失败
    """
    try:
        # 提取 JSON
        json_str = extract_json_from_text(text)

        # 解析 JSON
        data = json.loads(json_str)

        # Pydantic 校验
        return model.model_validate(data)

    except json.JSONDecodeError as e:
        raise JsonParseError(f"JSON 解析失败: {e}", text)
    except ValidationError as e:
        raise JsonParseError(f"数据校验失败: {e}", text)
```

---

## Step 4: 实现 SQL 解释助手

### 4.1 创建核心服务

创建 `src/chatbi/services/__init__.py`：

```python
"""业务服务模块"""
```

创建 `src/chatbi/services/sql_explainer.py`：

```python
"""SQL 解释服务"""

from chatbi.llm import chat_completion
from chatbi.models.sql import SqlExplanation
from chatbi.prompts.sql_explainer import SQL_EXPLAINER_SYSTEM, build_sql_explainer_prompt
from chatbi.utils.json_parser import JsonParseError, parse_json_response


class SqlExplainerService:
    """SQL 解释服务"""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def explain(self, sql: str, sample_data: str | None = None) -> SqlExplanation:
        """
        解释 SQL 查询

        Args:
            sql: SQL 查询语句
            sample_data: 可选的示例数据（CSV 格式）

        Returns:
            SQL 解释结果
        """
        user_prompt = build_sql_explainer_prompt(sql, sample_data)

        messages = [
            {"role": "system", "content": SQL_EXPLAINER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = chat_completion(messages, temperature=0.3)
                return parse_json_response(response, SqlExplanation)

            except JsonParseError as e:
                last_error = e
                if attempt < self.max_retries:
                    # 添加错误反馈，要求重新生成
                    messages.append({"role": "assistant", "content": e.raw_content})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"上面的输出不是有效的 JSON 格式，错误：{str(e)}。请重新输出，确保是严格的 JSON 格式。",
                        }
                    )

        # 所有重试都失败
        raise last_error


# 创建默认实例
sql_explainer = SqlExplainerService()
```

### 4.2 添加 API 接口

更新 `src/chatbi/main.py`，添加 SQL 解释接口：

```python
from chatbi.models.sql import SqlExplanation
from chatbi.services.sql_explainer import sql_explainer
from chatbi.utils.json_parser import JsonParseError


class ExplainSqlRequest(BaseModel):
    """SQL 解释请求"""

    sql: str
    sample_data: str | None = None


class ExplainSqlResponse(BaseModel):
    """SQL 解释响应"""

    success: bool
    data: SqlExplanation | None = None
    error: str | None = None


@app.post("/explain_sql", response_model=ExplainSqlResponse)
async def explain_sql(request: ExplainSqlRequest):
    """解释 SQL 查询"""
    try:
        result = sql_explainer.explain(request.sql, request.sample_data)
        return ExplainSqlResponse(success=True, data=result)
    except JsonParseError as e:
        return ExplainSqlResponse(success=False, error=str(e))
    except Exception as e:
        return ExplainSqlResponse(success=False, error=f"服务错误: {str(e)}")
```

### 4.3 创建 CLI 工具

创建 `src/chatbi/cli/__init__.py`：

```python
"""CLI 工具"""
```

创建 `src/chatbi/cli/sql_explainer.py`：

```python
"""SQL 解释器 CLI"""

import argparse
import json
import sys

from chatbi.services.sql_explainer import sql_explainer
from chatbi.utils.json_parser import JsonParseError


def main():
    parser = argparse.ArgumentParser(description="SQL 解释助手")
    parser.add_argument("--sql", type=str, help="SQL 查询语句")
    parser.add_argument("--file", type=str, help="包含 SQL 的文件路径")
    parser.add_argument("--sample", type=str, help="示例数据文件路径（CSV）")
    parser.add_argument("--output", type=str, choices=["json", "text"], default="text", help="输出格式")

    args = parser.parse_args()

    # 获取 SQL
    sql = None
    if args.sql:
        sql = args.sql
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sql = f.read()
    else:
        print("请通过 --sql 或 --file 提供 SQL 查询", file=sys.stderr)
        sys.exit(1)

    # 获取示例数据
    sample_data = None
    if args.sample:
        with open(args.sample, "r", encoding="utf-8") as f:
            sample_data = f.read()

    # 调用服务
    try:
        result = sql_explainer.explain(sql, sample_data)

        if args.output == "json":
            print(result.model_dump_json(indent=2))
        else:
            print("\n=== SQL 解释结果 ===\n")
            print(f"📊 目标指标: {', '.join(result.target_metrics) or '无'}")
            print(f"🔍 过滤条件: {', '.join(result.filters) or '无'}")
            print(f"📁 分组字段: {', '.join(result.group_by) or '无'}")
            print(f"🔗 表关联: {', '.join(result.joins) or '无'}")
            print(f"📈 排序: {', '.join(result.order_by) or '无'}")
            print(f"\n💡 业务解释:\n{result.business_explanation}")

    except JsonParseError as e:
        print(f"解析错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

在 `pyproject.toml` 中添加入口点：

```toml
[project.scripts]
sql-explainer = "chatbi.cli.sql_explainer:main"
```

重新安装：

```bash
uv pip install -e ".[dev]"
```

---

## Step 5: 测试 SQL 解释助手

### 5.1 API 测试

```bash
# 启动服务
uvicorn chatbi.main:app --reload

# 测试简单 SQL
curl -X POST http://localhost:8000/explain_sql \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT category, SUM(amount) as total FROM orders WHERE status = '\''completed'\'' GROUP BY category ORDER BY total DESC LIMIT 10"
  }'

# 测试复杂 SQL
curl -X POST http://localhost:8000/explain_sql \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total_amount FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.created_at >= '\''2024-01-01'\'' AND o.status IN ('\''completed'\'', '\''shipped'\'') GROUP BY u.id, u.name HAVING COUNT(o.id) > 5 ORDER BY total_amount DESC"
  }'
```

### 5.2 CLI 测试

```bash
# 直接传入 SQL
sql-explainer --sql "SELECT * FROM users WHERE age > 18"

# JSON 输出
sql-explainer --sql "SELECT category, COUNT(*) FROM products GROUP BY category" --output json

# 从文件读取
echo "SELECT * FROM orders WHERE date > '2024-01-01'" > /tmp/test.sql
sql-explainer --file /tmp/test.sql
```

### 5.3 编写单元测试

创建 `tests/test_sql_explainer.py`：

```python
"""SQL 解释器测试"""

import pytest
from chatbi.models.sql import SqlExplanation
from chatbi.utils.json_parser import extract_json_from_text, parse_json_response


def test_extract_json_from_markdown():
    """测试从 Markdown 提取 JSON"""
    text = """
这是一些说明文字。

```json
{"key": "value"}
```

结束。
"""
    result = extract_json_from_text(text)
    assert result == '{"key": "value"}'


def test_extract_json_direct():
    """测试直接 JSON"""
    text = '{"target_metrics": ["COUNT(*)"], "filters": []}'
    result = extract_json_from_text(text)
    assert "target_metrics" in result


def test_parse_sql_explanation():
    """测试解析 SQL 解释结果"""
    json_text = """
{
    "target_metrics": ["SUM(amount)"],
    "filters": ["status = 'completed'"],
    "group_by": ["category"],
    "joins": [],
    "order_by": [],
    "business_explanation": "按类别统计已完成订单金额"
}
"""
    result = parse_json_response(json_text, SqlExplanation)
    assert result.target_metrics == ["SUM(amount)"]
    assert result.business_explanation == "按类别统计已完成订单金额"


def test_sql_explanation_model():
    """测试 SqlExplanation 模型"""
    data = {
        "target_metrics": ["COUNT(*)"],
        "filters": [],
        "group_by": [],
        "joins": [],
        "order_by": [],
        "business_explanation": "统计总数",
    }
    model = SqlExplanation.model_validate(data)
    assert model.target_metrics == ["COUNT(*)"]
```

运行测试：

```bash
pytest tests/test_sql_explainer.py -v
```

---

## 项目结构检查

完成本阶段后，项目结构应该如下：

```
chatbi/
├── src/chatbi/
│   ├── __init__.py
│   ├── main.py           # FastAPI 主应用
│   ├── config.py         # 配置管理
│   ├── llm.py            # LLM 客户端
│   ├── cli/
│   │   ├── __init__.py
│   │   └── sql_explainer.py  # CLI 工具
│   ├── models/
│   │   ├── __init__.py
│   │   └── sql.py        # SQL 相关模型
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py   # 通用模板
│   │   └── sql_explainer.py  # SQL 解释 Prompt
│   ├── services/
│   │   ├── __init__.py
│   │   └── sql_explainer.py  # SQL 解释服务
│   └── utils/
│       ├── __init__.py
│       ├── data.py       # 数据处理工具
│       ├── async_utils.py
│       └── json_parser.py # JSON 解析
├── tests/
│   ├── test_api.py
│   ├── test_llm.py
│   └── test_sql_explainer.py
├── docs/
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## 验收检查清单

- [ ] Pandas 基础操作理解（过滤、分组、排序、join）
- [ ] 异步文件读写可用
- [ ] Prompt 模板系统建立
- [ ] Pydantic v2 模型定义正确
- [ ] JSON 解析和校验逻辑完善
- [ ] `POST /explain_sql` 接口正常工作
- [ ] `sql-explainer` CLI 命令可用
- [ ] 单元测试通过

---

## 下一步

完成本阶段后，进入 [阶段 2：RAG from scratch + LlamaIndex + 向量数据库](phase-2-rag-llamaindex-vectordb.md)
