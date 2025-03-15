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
            response = embeddings(model='nomic-embed-text', prompt=text)
            embeds.append(response['embedding'])
        return np.array(embeds)
    
    #余弦相似度计算
    @staticmethod
    def similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim
    
    def search(self, text):
        max_similarity = 0
        max_similarity_index = 0
        e = self.encode([text])[0]
        for idx, te in enumerate(self.embeds):
            similarity = self.similarity(e, te)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = idx
                
        return self.docs[max_similarity_index]
        
class Rag:
    def __init__(self, model, kb:Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """
        基于:%s
        回答:%s
        """
    def chat(self, text):
        #在知识库中查找
        content = self.kb.search(text)
        prompt = self.prompt_template % (content, text)
        response = chat(self.model, [Message(role='system', content=prompt)])
        return response['message']
        

rag = Rag('deepseek-r1:8b', Kb('test.txt'))
while True:
    q = input('Human:')
    r = rag.chat(q)
    print('Assistant:', r)
