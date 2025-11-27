# 阶段 0：现代 Python 工程 + LLM 基础设施

> 预计时间：1-2 周

## 学习目标

- 掌握现代 Python 项目工程化实践
- 搭建 FastAPI + Docker 开发环境
- 完成第一次 LLM API 调用
- 配置 LiteLLM 网关实现模型统一管理

## 前置条件

- Python 3.11+ 已安装
- Docker Desktop 已安装
- 任意 OpenAI 兼容的 API Key（OpenAI、Claude、国产模型等）

---

## Step 1: 初始化项目结构

### 1.1 创建项目目录

```bash
cd /Users/shenjiuyang/workspace/chatbi

# 创建核心目录
mkdir -p src/chatbi
mkdir -p tests
touch src/chatbi/__init__.py
touch tests/__init__.py
```

### 1.2 安装 uv（现代 Python 包管理器）

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证安装
uv --version
```

### 1.3 初始化 pyproject.toml

创建 `pyproject.toml` 文件：

```toml
[project]
name = "chatbi"
version = "0.1.0"
description = "Natural language interface for data analytics"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-dotenv>=1.0.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "pydantic>=2.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.7.0",
    "black>=24.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.black]
line-length = 120
target-version = ["py311"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### 1.4 安装依赖

```bash
# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv pip install -e ".[dev]"
```

### 1.5 验证工具链

```bash
# 代码检查
ruff check src/

# 代码格式化
black --check src/

# 运行测试
pytest
```

---

## Step 2: 环境变量管理

### 2.1 创建 .env.example 模板

```bash
touch .env.example
```

内容：

```env
# LLM Configuration
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

# LiteLLM (可选)
LITELLM_API_KEY=sk-xxx
LITELLM_BASE_URL=http://localhost:4000

# Server
HOST=0.0.0.0
PORT=8000
```

### 2.2 创建实际的 .env 文件

```bash
cp .env.example .env
# 编辑 .env，填入你的真实 API Key
```

### 2.3 创建配置模块

创建 `src/chatbi/config.py`：

```python
"""应用配置管理"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """应用配置"""

    # LLM 配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # LiteLLM 配置
    litellm_api_key: str = os.getenv("LITELLM_API_KEY", "")
    litellm_base_url: str = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
    use_litellm: bool = os.getenv("USE_LITELLM", "false").lower() == "true"

    # 服务配置
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
```

---

## Step 3: FastAPI 基础服务

### 3.1 创建主应用

创建 `src/chatbi/main.py`：

```python
"""FastAPI 主应用"""

from fastapi import FastAPI
from pydantic import BaseModel

from chatbi.config import get_settings

app = FastAPI(
    title="ChatBI",
    description="Natural language interface for data analytics",
    version="0.1.0",
)


class EchoRequest(BaseModel):
    """Echo 请求体"""

    message: str


class EchoResponse(BaseModel):
    """Echo 响应体"""

    message: str


@app.get("/ping")
async def ping():
    """健康检查"""
    return {"status": "pong"}


@app.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest):
    """原样返回输入"""
    return EchoResponse(message=request.message)


@app.get("/config")
async def config():
    """查看当前配置（脱敏）"""
    settings = get_settings()
    return {
        "model_name": settings.model_name,
        "base_url": settings.openai_base_url,
        "use_litellm": settings.use_litellm,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
```

### 3.2 启动并测试

```bash
# 启动服务
python -m chatbi.main

# 或使用 uvicorn（支持热重载）
uvicorn chatbi.main:app --reload --host 0.0.0.0 --port 8000
```

### 3.3 测试接口

```bash
# 健康检查
curl http://localhost:8000/ping

# Echo 测试
curl -X POST http://localhost:8000/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello ChatBI"}'

# 查看配置
curl http://localhost:8000/config
```

访问 http://localhost:8000/docs 查看 Swagger UI 文档。

---

## Step 4: LLM 调用基本模式

### 4.1 创建 LLM 客户端模块

创建 `src/chatbi/llm.py`：

```python
"""LLM 客户端封装"""

from openai import OpenAI

from chatbi.config import get_settings


def get_llm_client() -> OpenAI:
    """获取 LLM 客户端"""
    settings = get_settings()

    if settings.use_litellm:
        return OpenAI(
            api_key=settings.litellm_api_key,
            base_url=settings.litellm_base_url,
        )

    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    调用 LLM Chat Completion

    Args:
        messages: 消息列表，格式 [{"role": "user", "content": "..."}]
        model: 模型名称，默认使用配置中的模型
        temperature: 温度参数
        max_tokens: 最大生成 token 数

    Returns:
        模型回复内容
    """
    settings = get_settings()
    client = get_llm_client()

    response = client.chat.completions.create(
        model=model or settings.model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""


def simple_chat(user_message: str, system_prompt: str | None = None) -> str:
    """
    简单的单轮对话

    Args:
        user_message: 用户消息
        system_prompt: 系统提示词（可选）

    Returns:
        模型回复
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_message})

    return chat_completion(messages)
```

### 4.2 添加 LLM 测试接口

更新 `src/chatbi/main.py`，添加：

```python
from chatbi.llm import simple_chat


class LLMTestRequest(BaseModel):
    """LLM 测试请求"""

    message: str
    system_prompt: str | None = None


class LLMTestResponse(BaseModel):
    """LLM 测试响应"""

    reply: str
    model: str


@app.post("/llm-test", response_model=LLMTestResponse)
async def llm_test(request: LLMTestRequest):
    """测试 LLM 调用"""
    settings = get_settings()
    reply = simple_chat(request.message, request.system_prompt)
    return LLMTestResponse(reply=reply, model=settings.model_name)
```

### 4.3 测试 LLM 接口

```bash
# 简单测试
curl -X POST http://localhost:8000/llm-test \
  -H "Content-Type: application/json" \
  -d '{"message": "请用一句话介绍你自己"}'

# 带系统提示词
curl -X POST http://localhost:8000/llm-test \
  -H "Content-Type: application/json" \
  -d '{
    "message": "什么是 GMV？",
    "system_prompt": "你是一个数据分析专家，请用简洁的语言回答问题"
  }'
```

---

## Step 5: Docker 容器化

### 5.1 创建 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装 uv
RUN pip install uv

# 复制依赖文件
COPY pyproject.toml .

# 安装依赖
RUN uv pip install --system -e .

# 复制源代码
COPY src/ src/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "chatbi.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 创建 .dockerignore

```
.git
.gitignore
.env
.venv
__pycache__
*.pyc
.pytest_cache
.ruff_cache
.idea
.vscode
*.md
docs/
tests/
```

### 5.3 构建并运行

```bash
# 构建镜像
docker build -t chatbi:latest .

# 运行容器
docker run -d \
  --name chatbi \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OPENAI_BASE_URL=$OPENAI_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  chatbi:latest

# 查看日志
docker logs -f chatbi

# 测试
curl http://localhost:8000/ping
```

### 5.4 创建 docker-compose.yml

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - HOST=0.0.0.0
      - PORT=8000
    restart: unless-stopped

  # LiteLLM 网关（可选）
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    volumes:
      - ./litellm_config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    profiles:
      - litellm
```

---

## Step 6: LiteLLM 网关配置（可选）

### 6.1 创建 LiteLLM 配置

创建 `litellm_config.yaml`：

```yaml
model_list:
  # OpenAI 模型
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  # 可添加其他模型...
  # - model_name: claude-3-sonnet
  #   litellm_params:
  #     model: anthropic/claude-3-sonnet-20240229
  #     api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: sk-litellm-master-key  # 生产环境请修改
```

### 6.2 启动 LiteLLM

```bash
# 单独启动 LiteLLM
docker-compose --profile litellm up -d litellm

# 测试 LiteLLM
curl http://localhost:4000/health
```

### 6.3 切换使用 LiteLLM

修改 `.env`：

```env
USE_LITELLM=true
LITELLM_API_KEY=sk-litellm-master-key
LITELLM_BASE_URL=http://localhost:4000
```

重启服务后测试，验证功能不受影响。

---

## Step 7: 编写基础测试

### 7.1 创建测试文件

创建 `tests/test_api.py`：

```python
"""API 测试"""

import pytest
from fastapi.testclient import TestClient

from chatbi.main import app

client = TestClient(app)


def test_ping():
    """测试健康检查"""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}


def test_echo():
    """测试 echo 接口"""
    response = client.post("/echo", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json()["message"] == "Hello"


def test_config():
    """测试配置接口"""
    response = client.get("/config")
    assert response.status_code == 200
    assert "model_name" in response.json()
```

创建 `tests/test_llm.py`：

```python
"""LLM 模块测试"""

import pytest

from chatbi.llm import simple_chat


@pytest.mark.skipif(
    not pytest.importorskip("openai"),
    reason="需要有效的 API Key",
)
def test_simple_chat():
    """测试简单对话（需要真实 API）"""
    # 这个测试需要真实的 API Key
    # 可以用 pytest -m "not integration" 跳过
    pass
```

### 7.2 运行测试

```bash
# 运行所有测试
pytest -v

# 运行单个测试文件
pytest tests/test_api.py -v

# 查看覆盖率
pytest --cov=chatbi --cov-report=html
```

---

## 验收检查清单

完成本阶段后，确认以下内容：

- [ ] 项目结构完整：`src/chatbi/`、`tests/`、`pyproject.toml`
- [ ] `uv install` 可以正常安装依赖
- [ ] `ruff check .` 和 `black --check .` 通过
- [ ] `pytest` 测试通过
- [ ] `GET /ping` 返回 `{"status": "pong"}`
- [ ] `POST /echo` 正常工作
- [ ] `POST /llm-test` 能成功调用 LLM 并返回结果
- [ ] Docker 镜像可以成功构建和运行
- [ ] `.env` 文件包含所有必要配置，且不提交到 Git

---

## 下一步

完成本阶段后，进入 [阶段 1：Python 实战 + Prompt 工程 + 结构化输出](phase-1-python-prompt-structured-output.md)
