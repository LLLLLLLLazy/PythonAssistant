import os
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np


from Chunk import chunker
from Embedding import embedding



K = 3


class Database():
    def __init__(self, name: str, tokenizer, embedding_model)->None:
        self.files = []
        self.file_nums = 0
        
        self.db = os.path.join('database', (name + '.json'))
        
        with open(self.db, 'w', encoding = 'utf-8') as f:
            json.dump({}, f, ensure_ascii=False)

        self.tokenizer = tokenizer
        self.model = embedding_model


    def add(self, file_name: str)->None:
        '''
        添加文件到知识库
        输入：文件名
        输出：无
        '''
        if file_name in self.files:
            print(f"{file_name} already in {self.db}.")
            return
        
        file = os.path.join('data', file_name)
        
        # 读取文件为str
        with open(file, 'r', encoding = 'utf-8') as f:
            text = f.read()

        # 分块
        chunks = chunker(text)
        
        # 生成chunk的embedding
        chunks_vector = {}
        for chunk in chunks:
            chunks_vector[chunk] = embedding(self.model, self.tokenizer, chunk)
        
        # 保存到知识库
        with open(self.db, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        
        data.update(chunks_vector)

        with open(self.db, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        self.files.append(file_name)
        self.file_nums += 1
        print(f"Add file {file_name} to {self.db}.")


    def cosine_similarity(self, vector1: list[float], vector2: list[float]) -> float:
        '''
        计算两个向量的余弦相似度
        输入：两个向量
        输出：余弦相似度
        '''
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    def top_k_chunks(self, query: str, k: int=K) -> list[str]:
        '''
        返回与查询最相似的k个chunk
        输入：查询字符串，k
        输出：最相似的k个chunk
        '''
        similarities = {}

        # 生成查询的embedding
        query_vector = np.array(embedding(self.model, self.tokenizer, query))
        
        # 读取知识库
        with open(self.db, 'r', encoding = 'utf-8') as f:
            data = json.load(f)

        # 计算余弦相似度
        for chunk, vector in data.items():
            vector = np.array(vector)
            similarity = self.cosine_similarity(query_vector, vector)
            similarities[chunk] = similarity

        # 获取最相似的k个chunk
        top_k = sorted(similarities, key=similarities.get, reverse=True)[:k]
        
        return top_k



# 测试
#tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
#model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
#db = Database('dbtest', tokenizer, model)
#db.add(FILE)
#print(db.top_k_chunks('条件满足的时候执行语句'))




