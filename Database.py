import os
import json
from openai import OpenAI
import numpy as np


from config import FILES_PATH, DATABASE_PATH, K, BASELINE, DATABASE_NAME
from chunker import chunking
from embedding import embedding




class Database():
    def __init__(self, name: str)->None:
        self.db = os.path.join(DATABASE_PATH, (name + '.json'))
        
        if not os.path.exists(self.db):
            print(f"{self.db} not found.")
            raise FileNotFoundError
        
        with open(self.db, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        self.files = data["files"]
        
        self.embedding_client = OpenAI(
            api_key="sk-2345ecb2afae4811abbe33775a3e8e87",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 填写百炼服务的base_url
        )

    def cosine_similarity(self, vector1: list[float], vector2: list[float]) -> float:
        '''
        计算两个向量的余弦相似度
        输入：两个向量
        输出：余弦相似度
        '''
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def top_k_chunks(self, query: str, k: int=K, baseline: float=BASELINE) -> list[str]:
        '''
        返回与查询最相似的k个chunk
        输入：查询字符串，k
        输出：最相似的k个chunk
        '''
        similarities = {}

        # 生成查询的embedding
        query_vector = np.array(embedding(self.embedding_client, query))
        
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

        # 移除相似度低于BASELINE的chunk
        for chunk in top_k:
            if similarities[chunk] < baseline:
                top_k.remove(chunk)
        
        return top_k


class DynamicDatabase(Database):
    def __init__(self, name: str)->None:
        super().__init__(name)
        if not os.path.exists(self.db):
            with open(self.db, 'w', encoding = 'utf-8') as f:
                json.dump({"files": [], "vectors": {}}, f, ensure_ascii=False)

    def add(self, file_name: str)->None:
        '''
        添加文件到知识库
        输入：文件名
        输出：无
        '''
        
        file = os.path.join(FILES_PATH  , file_name)
        
        # 读取文件为str
        with open(file, 'r', encoding = 'utf-8') as f:
            text = f.read()

        # 分块
        chunks = chunking(text)
        
        # 生成chunk的embedding
        chunks_vector = {}
        for chunk in chunks:
            if chunk:
                chunks_vector[chunk] = embedding(self.embedding_client, chunk)
        
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

def init_database():
    database = DynamicDatabase(DATABASE_NAME)
    
    FILES = os.listdir(FILES_PATH)
    files = FILES.copy()

    for file in database.files:
        if file not in files:
            files.append(file)
    
    for file in files:
        if file in database.files and file in FILES:
            print(f"{file} already in {database.db}.")
            continue
        elif file in database.files:
            database.delete(file)
            print(f"{file} deleted from {database.db}.")
        else:
            database.add(file)
            print(f"{file} added to {database.db}.")
    
    print(f"{database.db} initialized.")

if __name__ == "__main__":
    init_database()
