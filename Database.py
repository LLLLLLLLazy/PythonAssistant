import os
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np


from config import K, DATABASE_PATH, FILES_PATH
from Chunk import chunker
from Embedding import embedding




class Database():
    def __init__(self, name: str, tokenizer, embedding_model)->None:
        self.db = os.path.join(DATABASE_PATH, (name + '.json'))
        
        if not os.path.exists(self.db):
            self.files = []
            initial_data = {"files": self.files, "vectors": {}}
            with open(self.db, 'w', encoding = 'utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False)
        else:
            with open(self.db, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
            self.files = data["files"]
        
        self.tokenizer = tokenizer
        self.model = embedding_model


    def add(self, file_name: str)->None:
        '''
        添加文件到知识库
        输入：文件名
        输出：无
        '''
        
        file = os.path.join(FILES_PATH, file_name)
        
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
        
        data["vectors"][file_name] = chunks_vector

        self.files.append(file_name)
        data["files"] = self.files
        with open(self.db, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    
    def delete(self, file_name: str)->None:
        '''
        从知识库删除文件
        输入：文件名
        输出：无
        '''
        with open(self.db, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        
        data["vectors"].pop(file_name)
        self.files.remove(file_name)
        data["files"] = self.files
        with open(self.db, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii=False)


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
        for _file, chunk_vectors in data["vectors"].items():
            for chunk, vector in chunk_vectors.items():
                vector = np.array(vector)
                similarity = self.cosine_similarity(query_vector, vector)
                similarities[chunk] = similarity

        # 获取最相似的k个chunk
        top_k = sorted(similarities, key=similarities.get, reverse=True)[:k]
        
        return top_k
