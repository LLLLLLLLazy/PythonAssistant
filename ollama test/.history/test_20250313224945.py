from ollama import chat, Message, embeddings
import numpy as np
import heapq

class Kb:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        self.docs = self.split_content(content)
        self.embeds = self.encode(self.docs)
        # Ԥ���㷶������Ч��
        self.norms = np.linalg.norm(self.embeds, axis=1, keepdims=True)
        self.normalized_embeds = self.embeds / self.norms

    @staticmethod
    def split_content(content, max_length=256):
        return [content[i:i+max_length] for i in range(0, len(content), max_length)]

    def encode(self, texts):
        # ������������Ч��
        batch_embeds = []
        for text in texts:
            response = embeddings(model='nomic-embed-text:v1.5', prompt=text)
            batch_embeds.append(response['embedding'])
        return np.array(batch_embeds)

    def search(self, text, top_k=3):
        query_embed = self.encode([text])
        # ��һ������
        query_norm = np.linalg.norm(query_embed)
        normalized_query = query_embed / query_norm
        
        # �����������ѭ����Ч�������ؼ���
        similarities = np.dot(self.normalized_embeds, normalized_query)
        
        # ʹ�öѽṹ��ȡtop_k
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
        self.prompt_template = """�������������ϣ�����ض����򣩣�
%s

��ش�������⣺%s
�����ϲ������˵��"""
        
    def format_context(self, contexts):
        return "\n\n".join([f"[��ض��� {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

    def chat(self, text):
        contexts = self.kb.search(text, top_k=3)
        prompt = self.prompt_template % (
            self.format_context(contexts), 
            text
        )
        response = chat(self.model, [Message(role='system', content=prompt)])
        return response['message']

# ʹ��ʾ��
rag = Rag('deepseek-r1:8b', Kb('test.txt'))
while True:
    q = input('Human: ')
    r = rag.chat(q)
    print('Assistant:', r['content'])
