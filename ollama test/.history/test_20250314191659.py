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
        dot_products = np.dot(self.embeds, query_embed)
        norms = np.linalg.norm(self.embeds, axis=1)
        norm_q = np.linalg.norm(query_embed)
        similarities = dot_products / (norms * norm_q)
        partitioned_indices = np.argpartition(similarities, -top_k)[-top_k:]
        sorted_indices = partitioned_indices[np.argsort(-similarities[partitioned_indices])]
        return [self.docs[i] for i in sorted_indices]

class Rag:
    def __init__(self, model, kb: Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """你是一个Python专家，帮我结合给定的资料，先简述资料内容再回答问题。如果问题无法从资料中获得，请说明。
        资料：
        %s
        
        问题：%s
        """
        
    def chat_stream(self, text):
        contents = self.kb.search(text)
        context = "\n".join([f"[相关段落{i+1}] {c}" for i, c in enumerate(contents)])
        prompt = self.prompt_template % (context, text)
        
        # 创建消息流并启用stream参数
        stream = chat(
            model=self.model,
            messages=[Message(role='system', content=prompt)],
            stream=True
        )
        
        # 逐块生成响应内容
        for chunk in stream:
            yield chunk['message']['content']

# 测试示例
rag = Rag('deepseek-r1:8b', Kb('python_test1.txt'))
while True:
    q = input('Human: ')
    if
    print('Assistant:', end='', flush=True)  # 不换行且立即输出
    
    full_response = []
    for chunk in rag.chat_stream(q):
        print(chunk, end='', flush=True)  # 逐块输出不换行
        full_response.append(chunk)
    
    print()  # 最终换行