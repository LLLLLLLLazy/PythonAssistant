from ollama import chat, Message, embeddings
import numpy as np

class Kb:
    # 保持原有Kb类实现不变
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
    
    def generate_prompt(self, text):
        """生成完整prompt并返回"""
        contents = self.kb.search(text)
        context = "\n".join([f"[相关段落{i+1}] {c}" for i, c in enumerate(contents)])
        return self.prompt_template % (context, text)
    
    def chat_stream(self, prompt):
        """接收预生成的prompt进行流式对话"""
        stream = chat(
            model=self.model,
            messages=[Message(role='system', content=prompt)],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']

# 测试示例
rag = Rag('deepseek-r1:8b', Kb('python_test1.txt'))
while True:
    q = input('Human: ')
    
    if q == "#Exit":
        break
    
    # 生成并显示prompt
    prompt = rag.generate_prompt(q)
    print("\n" + "="*40 + " DEBUG PROMPT " + "="*40)
    print(prompt)
    print("="*93 + "\n")
    
    # 流式输出回答
    print('Assistant:', end='', flush=True)
    full_response = []
    for chunk in rag.chat_stream(prompt):
        print(chunk, end='', flush=True)
        full_response.append(chunk)
    
    print()  # 最终换行