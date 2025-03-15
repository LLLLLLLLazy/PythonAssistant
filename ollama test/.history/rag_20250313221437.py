from ollama import chat, Message, embeddings
import numpy as np

# messages = [
#     Message(role='system', content='将用户的输入翻译为英文'),
#     Message(role='user', content='不要温顺地走进那个凉夜')
# ]

# response = chat(model='deepseek-r1:8b', messages=messages)
# print(response)
#创建知识库
class Kb:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        self.content = content
        self.docs = self.split_content(content)
        self.embeds = self.encode(self.docs)
        
    @staticmethod
    def split_content(content, max_length = 256):
        chunks = []
        for i in range(0, len(content), max_length):
            chunks.append(content[i:i + max_length])
        return chunks
    
    def encode(self, texts):
        embeds = []
        for text in texts:
            response = embeddings(model='nomic-embed-text', prompt='hello')
            embeds.append(response['embedding'])
        return np.array(embeds)
    
    #余弦相似度计算
    def similarity(self, A, B):
        dot_product = 

kb = Kb('python_test1.txt')
for e in kb.embeds:
    print(e)

