# 阶段 2：RAG from scratch + LlamaIndex + 向量数据库

> 预计时间：3 周

## 学习目标

- 从零实现 RAG（检索增强生成）流程
- 掌握文档解析、切分、向量化的核心概念
- 使用向量数据库（Chroma/Qdrant）存储和检索
- 使用 LlamaIndex 框架重构 RAG

## 前置条件

- 完成 [阶段 1](phase-1-python-prompt-structured-output.md)
- 准备一些业务文档（指标口径、表结构说明等）

---

## Part 1: 手写 RAG（不用框架）

### Step 1: 准备文档数据

#### 1.1 创建示例文档目录

```bash
mkdir -p data/docs
```

#### 1.2 创建示例文档

创建 `data/docs/metrics.md`：

```markdown
# 核心业务指标说明

## GMV（Gross Merchandise Volume）

**定义**：成交总额，指一定时间内的成交金额总和。

**计算公式**：
```
GMV = SUM(订单金额)
WHERE 订单状态 IN ('已支付', '已发货', '已完成')
```

**注意事项**：
- 不包含取消和退款订单
- 包含运费和优惠前的原价

## DAU（Daily Active Users）

**定义**：日活跃用户数，当天有登录或访问行为的独立用户数。

**计算公式**：
```
DAU = COUNT(DISTINCT user_id)
WHERE event_date = 目标日期
AND event_type IN ('login', 'page_view', 'click')
```

## 用户留存率

**定义**：在某个时间点新增的用户中，经过一段时间后仍然活跃的用户比例。

**次日留存**：
```
次日留存率 = 次日活跃的新用户数 / 当日新增用户数 * 100%
```

**7日留存**：
```
7日留存率 = 第7天活跃的新用户数 / 当日新增用户数 * 100%
```
```

创建 `data/docs/tables.md`：

```markdown
# 数据表结构说明

## 用户表 (users)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | 用户ID，主键 |
| name | VARCHAR(100) | 用户名 |
| email | VARCHAR(200) | 邮箱 |
| phone | VARCHAR(20) | 手机号 |
| created_at | TIMESTAMP | 注册时间 |
| status | VARCHAR(20) | 状态：active/inactive/banned |

## 订单表 (orders)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | 订单ID，主键 |
| user_id | BIGINT | 用户ID，外键关联 users.id |
| amount | DECIMAL(10,2) | 订单金额 |
| status | VARCHAR(20) | 订单状态：pending/paid/shipped/completed/cancelled |
| created_at | TIMESTAMP | 下单时间 |
| paid_at | TIMESTAMP | 支付时间 |

## 用户行为表 (user_events)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | 事件ID |
| user_id | BIGINT | 用户ID |
| event_type | VARCHAR(50) | 事件类型：login/page_view/click/purchase |
| event_date | DATE | 事件日期 |
| event_time | TIMESTAMP | 事件时间 |
| page_url | VARCHAR(500) | 页面URL |
```

### Step 2: 文档解析与切分

#### 2.1 添加依赖

更新 `pyproject.toml`：

```toml
dependencies = [
    # ... 已有依赖
    "tiktoken>=0.7.0",      # Token 计数
    "numpy>=1.26.0",        # 向量计算
    "chromadb>=0.5.0",      # 向量数据库
]
```

```bash
uv pip install -e ".[dev]"
```

#### 2.2 创建文档处理模块

创建 `src/chatbi/rag/__init__.py`：

```python
"""RAG 模块"""
```

创建 `src/chatbi/rag/document.py`：

```python
"""文档数据结构"""

from dataclasses import dataclass, field
from typing import Any
import hashlib


@dataclass
class Document:
    """文档"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """生成文档唯一 ID"""
        source = self.metadata.get("source", "")
        return hashlib.md5(f"{source}:{self.content[:100]}".encode()).hexdigest()


@dataclass
class Chunk:
    """文档片段"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        """生成片段唯一 ID"""
        source = self.metadata.get("source", "")
        index = self.metadata.get("chunk_index", 0)
        return hashlib.md5(f"{source}:{index}:{self.content[:50]}".encode()).hexdigest()
```

#### 2.3 创建文档加载器

创建 `src/chatbi/rag/loader.py`：

```python
"""文档加载器"""

from pathlib import Path

from chatbi.rag.document import Document


def load_markdown(file_path: str | Path) -> Document:
    """加载 Markdown 文件"""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return Document(
        content=content,
        metadata={
            "source": str(path),
            "filename": path.name,
            "filetype": "markdown",
        },
    )


def load_text(file_path: str | Path) -> Document:
    """加载纯文本文件"""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return Document(
        content=content,
        metadata={
            "source": str(path),
            "filename": path.name,
            "filetype": "text",
        },
    )


def load_directory(dir_path: str | Path, extensions: list[str] | None = None) -> list[Document]:
    """
    加载目录下所有文档

    Args:
        dir_path: 目录路径
        extensions: 文件扩展名列表，默认 [".md", ".txt"]
    """
    if extensions is None:
        extensions = [".md", ".txt"]

    docs = []
    path = Path(dir_path)

    for ext in extensions:
        for file_path in path.glob(f"**/*{ext}"):
            if ext == ".md":
                docs.append(load_markdown(file_path))
            else:
                docs.append(load_text(file_path))

    return docs
```

#### 2.4 创建文本切分器

创建 `src/chatbi/rag/splitter.py`：

```python
"""文本切分器"""

import re
from typing import Callable

import tiktoken

from chatbi.rag.document import Chunk, Document


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """计算文本的 token 数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class TextSplitter:
    """文本切分器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: Callable[[str], int] = count_tokens,
    ):
        """
        Args:
            chunk_size: 每块最大长度（token 数或字符数）
            chunk_overlap: 块之间的重叠长度
            length_function: 计算文本长度的函数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        """将文本切分为多个片段"""
        # 按段落分割
        paragraphs = re.split(r"\n\n+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = self.length_function(para)

            # 如果单个段落超过 chunk_size，需要进一步切分
            if para_length > self.chunk_size:
                # 先保存当前块
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 按句子切分大段落
                sentences = re.split(r"(?<=[。！？.!?])\s*", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sent_length = self.length_function(sentence)
                    if current_length + sent_length <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_length += sent_length
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sent_length
            else:
                if current_length + para_length <= self.chunk_size:
                    current_chunk.append(para)
                    current_length += para_length
                else:
                    # 保存当前块，开始新块
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))

                    # 添加重叠
                    if self.chunk_overlap > 0 and chunks:
                        overlap_text = self._get_overlap(chunks[-1])
                        current_chunk = [overlap_text, para] if overlap_text else [para]
                        current_length = self.length_function("\n\n".join(current_chunk))
                    else:
                        current_chunk = [para]
                        current_length = para_length

        # 处理最后一块
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _get_overlap(self, text: str) -> str:
        """获取重叠部分"""
        if self.chunk_overlap <= 0:
            return ""

        # 从末尾截取大约 overlap 长度的文本
        words = text.split()
        overlap_words = []
        current_length = 0

        for word in reversed(words):
            word_length = self.length_function(word)
            if current_length + word_length <= self.chunk_overlap:
                overlap_words.insert(0, word)
                current_length += word_length
            else:
                break

        return " ".join(overlap_words)

    def split_document(self, doc: Document) -> list[Chunk]:
        """将文档切分为多个片段"""
        texts = self.split_text(doc.content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)

        return chunks


class MarkdownSplitter(TextSplitter):
    """Markdown 感知的切分器"""

    def split_text(self, text: str) -> list[str]:
        """按标题切分 Markdown"""
        # 按一级和二级标题分割
        sections = re.split(r"\n(?=##?\s)", text)

        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_length = self.length_function(section)
            if section_length <= self.chunk_size:
                chunks.append(section)
            else:
                # 章节太长，进一步切分
                sub_chunks = super().split_text(section)
                chunks.extend(sub_chunks)

        return chunks
```

### Step 3: Embedding + 向量检索

#### 3.1 创建 Embedding 模块

创建 `src/chatbi/rag/embedding.py`：

```python
"""Embedding 向量化"""

import numpy as np
from openai import OpenAI

from chatbi.config import get_settings


def get_embedding_client() -> OpenAI:
    """获取 Embedding 客户端"""
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    获取文本的 embedding 向量

    Args:
        text: 输入文本
        model: embedding 模型名称
    """
    client = get_embedding_client()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    批量获取文本的 embedding 向量

    Args:
        texts: 输入文本列表
        model: embedding 模型名称
    """
    client = get_embedding_client()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算余弦相似度"""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SimpleVectorStore:
    """简单的内存向量存储（仅用于学习理解原理）"""

    def __init__(self):
        self.chunks: list[tuple[str, list[float], dict]] = []  # (content, embedding, metadata)

    def add(self, content: str, embedding: list[float], metadata: dict | None = None):
        """添加向量"""
        self.chunks.append((content, embedding, metadata or {}))

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        相似度检索

        Returns:
            [{"content": str, "score": float, "metadata": dict}, ...]
        """
        results = []
        for content, embedding, metadata in self.chunks:
            score = cosine_similarity(query_embedding, embedding)
            results.append({
                "content": content,
                "score": score,
                "metadata": metadata,
            })

        # 按相似度降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
```

#### 3.2 创建 Chroma 向量存储

创建 `src/chatbi/rag/vectorstore.py`：

```python
"""向量数据库封装"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from chatbi.rag.document import Chunk
from chatbi.rag.embedding import get_embedding, get_embeddings


class ChromaVectorStore:
    """Chroma 向量数据库封装"""

    def __init__(
        self,
        collection_name: str = "chatbi",
        persist_directory: str | None = None,
    ):
        """
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录，None 则使用内存模式
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100):
        """
        添加文档片段到向量库

        Args:
            chunks: 文档片段列表
            batch_size: 批处理大小
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]

            # 批量获取 embeddings
            embeddings = get_embeddings(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        相似度检索

        Args:
            query: 查询文本
            top_k: 返回数量
            where: 元数据过滤条件

        Returns:
            [{"content": str, "score": float, "metadata": dict}, ...]
        """
        query_embedding = get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                items.append({
                    "content": doc,
                    "score": 1 - results["distances"][0][i],  # 转换为相似度
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return items

    def delete_collection(self):
        """删除集合"""
        self.client.delete_collection(self.collection.name)

    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count()
```

### Step 4: RAG 查询流程

#### 4.1 创建 RAG 查询引擎

创建 `src/chatbi/rag/query.py`：

```python
"""RAG 查询引擎"""

from dataclasses import dataclass

from chatbi.llm import chat_completion
from chatbi.rag.vectorstore import ChromaVectorStore


@dataclass
class RAGResponse:
    """RAG 响应"""

    answer: str
    sources: list[dict]  # [{"content": str, "score": float, "metadata": dict}]


RAG_SYSTEM_PROMPT = """你是一个专业的数据分析助手，负责回答用户关于数据指标、表结构、数仓设计等问题。

## 回答要求
1. 只基于提供的参考文档回答问题
2. 如果文档中没有相关信息，明确告知用户
3. 回答要准确、简洁
4. 如果引用了文档内容，请标注来源

## 参考文档
{context}
"""

RAG_USER_PROMPT = """请根据参考文档回答以下问题：

问题：{question}
"""


class RAGQueryEngine:
    """RAG 查询引擎"""

    def __init__(
        self,
        vectorstore: ChromaVectorStore,
        top_k: int = 3,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def query(self, question: str) -> RAGResponse:
        """
        执行 RAG 查询

        Args:
            question: 用户问题

        Returns:
            RAG 响应，包含答案和引用来源
        """
        # 1. 检索相关文档
        retrieved = self.vectorstore.search(question, top_k=self.top_k)

        # 2. 构建上下文
        context_parts = []
        for i, item in enumerate(retrieved, 1):
            source = item["metadata"].get("filename", "未知来源")
            context_parts.append(f"[文档{i}] 来源: {source}\n{item['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # 3. 构建 Prompt
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        user_prompt = RAG_USER_PROMPT.format(question=question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 4. 调用 LLM
        answer = chat_completion(messages, temperature=0.3)

        return RAGResponse(answer=answer, sources=retrieved)
```

#### 4.2 创建索引构建脚本

创建 `src/chatbi/rag/build_index.py`：

```python
"""构建 RAG 索引"""

import argparse
from pathlib import Path

from chatbi.rag.loader import load_directory
from chatbi.rag.splitter import MarkdownSplitter
from chatbi.rag.vectorstore import ChromaVectorStore


def build_index(
    docs_dir: str,
    persist_dir: str = "./data/vectordb",
    collection_name: str = "chatbi",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    构建 RAG 索引

    Args:
        docs_dir: 文档目录
        persist_dir: 向量库持久化目录
        collection_name: 集合名称
        chunk_size: 切分块大小
        chunk_overlap: 块重叠大小
    """
    print(f"📂 加载文档: {docs_dir}")
    docs = load_directory(docs_dir)
    print(f"   找到 {len(docs)} 个文档")

    print("✂️  切分文档...")
    splitter = MarkdownSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_document(doc)
        all_chunks.extend(chunks)
        print(f"   {doc.metadata['filename']}: {len(chunks)} 个片段")

    print(f"📊 总计 {len(all_chunks)} 个文档片段")

    print(f"🔄 向量化并存入数据库: {persist_dir}")
    vectorstore = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    vectorstore.add_chunks(all_chunks)

    print(f"✅ 索引构建完成，共 {vectorstore.count()} 条记录")


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 索引")
    parser.add_argument("--docs", type=str, default="./data/docs", help="文档目录")
    parser.add_argument("--persist", type=str, default="./data/vectordb", help="向量库目录")
    parser.add_argument("--collection", type=str, default="chatbi", help="集合名称")
    parser.add_argument("--chunk-size", type=int, default=500, help="切分块大小")
    parser.add_argument("--overlap", type=int, default=50, help="块重叠大小")

    args = parser.parse_args()

    build_index(
        docs_dir=args.docs,
        persist_dir=args.persist,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
```

#### 4.3 添加 API 接口

更新 `src/chatbi/main.py`，添加 RAG 接口：

```python
from chatbi.rag.vectorstore import ChromaVectorStore
from chatbi.rag.query import RAGQueryEngine, RAGResponse


# 初始化 RAG 引擎（延迟加载）
_rag_engine: RAGQueryEngine | None = None


def get_rag_engine() -> RAGQueryEngine:
    global _rag_engine
    if _rag_engine is None:
        vectorstore = ChromaVectorStore(
            collection_name="chatbi",
            persist_directory="./data/vectordb",
        )
        _rag_engine = RAGQueryEngine(vectorstore)
    return _rag_engine


class AskRAGRequest(BaseModel):
    """RAG 查询请求"""
    question: str


class SourceItem(BaseModel):
    """引用来源"""
    content: str
    score: float
    source: str


class AskRAGResponse(BaseModel):
    """RAG 查询响应"""
    answer: str
    sources: list[SourceItem]


@app.post("/ask_rag_raw", response_model=AskRAGResponse)
async def ask_rag_raw(request: AskRAGRequest):
    """手写 RAG 查询接口"""
    engine = get_rag_engine()
    result = engine.query(request.question)

    sources = [
        SourceItem(
            content=s["content"][:200] + "..." if len(s["content"]) > 200 else s["content"],
            score=s["score"],
            source=s["metadata"].get("filename", "未知"),
        )
        for s in result.sources
    ]

    return AskRAGResponse(answer=result.answer, sources=sources)
```

### Step 5: 测试手写 RAG

```bash
# 1. 构建索引
python -m chatbi.rag.build_index --docs ./data/docs

# 2. 启动服务
uvicorn chatbi.main:app --reload

# 3. 测试查询
curl -X POST http://localhost:8000/ask_rag_raw \
  -H "Content-Type: application/json" \
  -d '{"question": "GMV 的计算公式是什么？"}'

curl -X POST http://localhost:8000/ask_rag_raw \
  -H "Content-Type: application/json" \
  -d '{"question": "订单表有哪些字段？"}'

curl -X POST http://localhost:8000/ask_rag_raw \
  -H "Content-Type: application/json" \
  -d '{"question": "如何计算用户次日留存率？"}'
```

---

## Part 2: 使用 LlamaIndex 重写 RAG

### Step 6: 安装 LlamaIndex

```bash
# 更新 pyproject.toml
# dependencies = [
#     ...
#     "llama-index>=0.11.0",
#     "llama-index-vector-stores-chroma>=0.2.0",
#     "llama-index-embeddings-openai>=0.2.0",
# ]

uv pip install -e ".[dev]"
```

### Step 7: LlamaIndex RAG 实现

创建 `src/chatbi/rag_llamaindex/__init__.py`：

```python
"""LlamaIndex RAG 模块"""
```

创建 `src/chatbi/rag_llamaindex/index.py`：

```python
"""LlamaIndex 索引管理"""

from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from chatbi.config import get_settings


def setup_llama_index():
    """配置 LlamaIndex 全局设置"""
    settings = get_settings()

    Settings.llm = OpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        api_base=settings.openai_base_url,
        temperature=0.3,
    )

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
        api_base=settings.openai_base_url,
    )


def build_llamaindex(
    docs_dir: str,
    persist_dir: str = "./data/llamaindex_db",
    collection_name: str = "chatbi_llamaindex",
) -> VectorStoreIndex:
    """
    使用 LlamaIndex 构建索引

    Args:
        docs_dir: 文档目录
        persist_dir: 持久化目录
        collection_name: 集合名称
    """
    setup_llama_index()

    # 加载文档
    documents = SimpleDirectoryReader(docs_dir).load_data()
    print(f"📂 加载了 {len(documents)} 个文档")

    # 配置向量存储
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 配置节点解析器
    node_parser = MarkdownNodeParser()

    # 构建索引
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[node_parser],
        show_progress=True,
    )

    print("✅ LlamaIndex 索引构建完成")
    return index


def load_llamaindex(
    persist_dir: str = "./data/llamaindex_db",
    collection_name: str = "chatbi_llamaindex",
) -> VectorStoreIndex:
    """加载已有的 LlamaIndex 索引"""
    setup_llama_index()

    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(vector_store)
    return index
```

创建 `src/chatbi/rag_llamaindex/query.py`：

```python
"""LlamaIndex 查询引擎"""

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor


@dataclass
class LlamaIndexResponse:
    """LlamaIndex 响应"""

    answer: str
    sources: list[dict]


QUERY_SYSTEM_PROMPT = """你是一个专业的数据分析助手。
请根据提供的上下文信息回答用户的问题。
如果上下文中没有相关信息，请明确告知。
回答要准确、简洁，并标注信息来源。
"""


class LlamaIndexQueryEngine:
    """LlamaIndex 查询引擎封装"""

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 3,
        similarity_cutoff: float = 0.5,
    ):
        self.index = index
        self.top_k = top_k

        # 配置检索器
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

        # 配置后处理器
        postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)

        # 创建查询引擎
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[postprocessor],
        )

    def query(self, question: str) -> LlamaIndexResponse:
        """执行查询"""
        response = self.query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "score": node.score or 0.0,
                "metadata": node.metadata,
            })

        return LlamaIndexResponse(
            answer=str(response),
            sources=sources,
        )
```

### Step 8: 添加 LlamaIndex API

更新 `src/chatbi/main.py`：

```python
from chatbi.rag_llamaindex.index import load_llamaindex
from chatbi.rag_llamaindex.query import LlamaIndexQueryEngine

# LlamaIndex 引擎（延迟加载）
_llamaindex_engine: LlamaIndexQueryEngine | None = None


def get_llamaindex_engine() -> LlamaIndexQueryEngine:
    global _llamaindex_engine
    if _llamaindex_engine is None:
        index = load_llamaindex()
        _llamaindex_engine = LlamaIndexQueryEngine(index)
    return _llamaindex_engine


@app.post("/ask_rag", response_model=AskRAGResponse)
async def ask_rag(request: AskRAGRequest):
    """LlamaIndex RAG 查询接口"""
    engine = get_llamaindex_engine()
    result = engine.query(request.question)

    sources = [
        SourceItem(
            content=s["content"],
            score=s["score"],
            source=s["metadata"].get("file_name", "未知"),
        )
        for s in result.sources
    ]

    return AskRAGResponse(answer=result.answer, sources=sources)
```

### Step 9: 测试 LlamaIndex RAG

```bash
# 1. 构建 LlamaIndex 索引
python -c "
from chatbi.rag_llamaindex.index import build_llamaindex
build_llamaindex('./data/docs')
"

# 2. 测试查询
curl -X POST http://localhost:8000/ask_rag \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 DAU？如何计算？"}'
```

---

## 验收检查清单

### Part 1: 手写 RAG
- [ ] 文档加载器可以读取 Markdown/文本文件
- [ ] 文本切分器能按 token 限制切块
- [ ] Embedding 向量化正常工作
- [ ] Chroma 向量存储可以添加和检索
- [ ] `POST /ask_rag_raw` 接口返回正确答案和来源

### Part 2: LlamaIndex RAG
- [ ] LlamaIndex 索引构建成功
- [ ] 索引可以持久化和加载
- [ ] `POST /ask_rag` 接口正常工作
- [ ] 检索结果包含相关性分数和来源信息

---

## 下一步

完成本阶段后，进入 [阶段 3：服务化、Web Demo、Text-to-SQL](phase-3-service-webdemo-text2sql.md)
