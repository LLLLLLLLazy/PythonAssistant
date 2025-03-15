from ollama import chat, Message, embeddings
import numpy as np
import heapq

class Kb:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        self.docs = self.split_content(content)
        self.embeds = self.encode(self.docs)
        # 预计算范数提升效率
        self.norms = np.linalg.norm(self.embeds, axis=1, keepdims=True)
        self.normalized_embeds = self.embeds / self.norms

    @staticmethod
    def split_content(content, max_length=256):
        return [content[i:i+max_length] for i in range(0, len(content), max_length)]

    def encode(self, texts):
        # 批量处理提升效率
        batch_embeds = []
        for text in texts:
            response = embeddings(model='nomic-embed-text:v1.5', prompt=text)
            batch_embeds.append(response['embedding'])
        return np.array(batch_embeds)

    def search(self, text, top_k=3):
        query_embed = self.encode([text])
        # 归一化处理
        query_norm = np.linalg.norm(query_embed)
        normalized_query = query_embed / query_norm
        
        # 矩阵运算代替循环（效率提升关键）
        similarities = np.dot(self.normalized_embeds, normalized_query)
        
        # 使用堆结构获取top_k
        top_indices = heapq.nlargest(
            top_k, 
            range(len(similarities)), 
            key=lambda i: similarities[i]
        )
        
        return [self.docs[i] for i in top_indices]

class Rag:
    def __init__(self, model, kb: Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """结合以下相关资料（按相关度排序）：
%s

请回答这个问题：%s
若资料不相关请说明"""
        
    def format_context(self, contexts):
        return "\n\n".join([f"[相关段落 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

    def chat(self, text):
        contexts = self.kb.search(text, top_k=3)
        prompt = self.prompt_template % (
            self.format_context(contexts), 
            text
        )
        response = chat(self.model, [Message(role='system', content=prompt)])
        return response['message']

# 使用示例
rag = Rag('deepseek-r1:8b', Kb('test.txt'))
while True:
    q = input('Human: ')
    r = rag.chat(q)
    print('Assistant:', r['content'])
