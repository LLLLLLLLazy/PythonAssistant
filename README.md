# 手搓版本
python >= 3.11

embedding模型使用Huggingface"sentence-transformers/all-MiniLM-L6-v2"

llm使用"Qwen/Qwen2.5-1.5B"

chunk算法目前直接按固定长度分块，可以处理.txt,.md文件。

知识库管理没用外部库，直接把`chunks_vector`字典存为json，对小规模文件没有压力。


## 安装下列依赖
```python
numpy
transfomers
torch
```
```bash
pip install -r requirements.txt
```

## 安装embedding及llm
首先将config.py里的`EMBEDDING_MODEL_PATH`,`LLM_PATH`更改为存放的文件夹路径。
再运行download_model.py,会自动从Huggingface下载对应的模型。
```bash
python download_model.py
```

## 更改配置
打开config.py
```python
# 模型存放的路径
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen2/qwen2-1.5B"

EMBEDDING_MODEL_PATH = "F:\\embedding_model"
LLM_PATH = "F:\\llm"


# 知识文件
FILES = ['test0.txt', 'test1.md', 'maogai.txt']
DATABASE_NAME = 'database_test'


# 分块长度
CHUNK_LENGTH = 50 

# top_k的数量
K = 3
```
`FILES`更改为data中的文件，`DATABASE_NAME`改为知识库名称，将在database文件夹下建立.json文件
## 运行

```bash
python main.py
```

运行将建立知识库，并在命令行中进入和AI的问答循环。