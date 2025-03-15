from ollama import chat, Message, embeddings
import numpy as np

class Kb:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        self.content = content
        self.docs = self.split_content(content)
        self.embeds = self.encode(self.docs)
        
    @staticmethod
    def split_content(content, max_length=256):
        return [content[i:i+max_length] for i in range(0, len(content), max_length)]
    
    def encode(self, texts):
        return np.array([embeddings(model='nomic-embed-text', prompt=text)['embedding'] for text in texts])
    
    def search(self, text, top_k=3):
        query_embed = self.encode([text])[0]
        # 向量化计算余弦相似度
        dot_products = np.dot(self.embeds, query_embed)
        norms = np.linalg.norm(self.embeds, axis=1)
        norm_q = np.linalg.norm(query_embed)
        similarities = dot_products / (norms * norm_q)
        
        # 使用argpartition优化获取top_k索引
        partitioned_indices = np.argpartition(similarities, -top_k)[-top_k:]
        sorted_indices = partitioned_indices[np.argsort(-similarities[partitioned_indices])]
        
        return [self.docs[i] for i in sorted_indices]

class Rag:
    def __init__(self, model, kb: Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """你是一个python编程专家，根据三条相关内容回答问题，若给出相关内容没有需要知识，ze：
        相关内容：
        %s
        
        问题：%s
        答案："""
        
    def chat(self, text):
        contents = self.kb.search(text)
        context = "\n".join([f"[相关段落{i+1}] {c}" for i, c in enumerate(contents)])
        prompt = self.prompt_template % (context, text)
        response = chat(self.model, [Message(role='system', content=prompt)])
        return response['message']

rag = Rag('deepseek-r1:8b', Kb('test.txt'))

while True:
    q = input('Human:')
    r = rag.chat(q)
    print('Assistant:', r['content'])