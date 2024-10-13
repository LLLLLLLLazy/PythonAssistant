# 手搓版本
embedding模型使用transformers"sentence-transformers/all-MiniLM-L6-v2"
分词器算法目前直接按固定长度分块，可以处理.txt,.md文件。
知识库没用外部库，直接把chunks_vector字典存为json，目前看来对小规模文件没有压力。
目前llm部分返回处理好的提示词，还没接大模型，可以考虑API，应该比GPU服务器便宜而且效果更好。

## 安装下列依赖
```python
numpy
transfomers
torch
```
```bash
pip install -r requirements.txt
```
## 运行
```bash
python main.py
```

如果环境没问题的话(主要是transformers和torch，用来embedding，第一次运行应该会下载embedding模型，比较久)。
运行main.py将会在文件夹下建立名为DATABASE_NAME的知识库，然后AI在命令行里循环回答问题。

后续看后端怎么处理响应，再改AskAI的过程，主要是和main函数对接，以及优化Chunk和添加命令行管理知识库的功能。。。
