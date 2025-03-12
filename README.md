# 手搓版本

python >= 3.11

模型通过 API 调用

embedding 模型使用"text-embedding-v3"

llm 使用"deepseek-r1"

chunk 算法目前直接按固定长度分块（块与块间有一定重叠），可以处理 txt、md 文件。

知识库没用向量数据库，直接把`chunks_vector`字典存为 json，对小规模文件没有压力。

## 安装下列依赖

```python
numpy
openai
```

```bash
pip install -r requirements.txt
```

## 配置

打开 config.py

```python
# 模型名称
LLM_NAME = "deepseek-reasoner"
EMBEDDING_NAME = "text-embedding-v3"

# 知识文件存放路径
FILES_PATH = 'data'

# 知识库存放路径
DATABASE_PATH = 'database'

# 知识库名称
DATABASE_NAME = 'database_test'


# 分块长度
CHUNK_LENGTH = 50

# 分块重叠长度
OVERLAP = 0

# top_k的数量
K = 5

# 接受top_k的基线
BASELINE = 0.5
```

## 建立知识库

```bash
python database.py
```

将`FILES_PATH`下的源文件转换为向量存放在`DATABASE_PATH`，可以增删文件，动态建立知识库。

## 运行

```bash
python ai.py
```

在命令行中进入和 AI 的问答循环。
