# ��װ��Ҫ�⣨����ǰ��ִ�У�
# pip install ollama numpy rank_bm25 jieba langchain

from ollama import chat, Message, embeddings
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Kb:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # �Ľ�1�������ı��ֿ�
        self.docs = self.split_content(content)
        
        # �Ľ�2����ϼ�����ʼ��
        self.embeds = self.encode(self.docs)
        self.bm25_index = self.build_bm25_index(self.docs)
    
    @staticmethod
    def split_content(content):
        """�Ľ������ֿܷ����"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,    # Ŀ����С
            chunk_overlap=50,   # ����ص���ֹ�Ͼ�
            separators=["\n\n", "\n", "��", "��", "��", "��"],  # �����Ѻ÷ָ���
            length_function=len
        )
        return splitter.split_text(content)
    
    def build_bm25_index(self, docs):
        """����BM25�ؼ��ʼ�������"""
        # ���ķִʴ���
        tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
        return BM25Okapi(tokenized_docs)
    
    def encode(self, texts):
        """Ƕ�����"""
        return np.array([embeddings(model='nomic-embed-text', prompt=text)['embedding'] for text in texts])
    
    def search(self, text, top_k=3):
        """�Ľ�2����ϼ�������"""
        # ����ִ�����ּ���
        semantic_results = self.semantic_search(text, top_k*2)
        keyword_results = self.keyword_search(text, top_k*2)
        
        # ����ں���ȥ��
        combined = list({doc: None for doc in semantic_results + keyword_results}.keys())
        
        # ��������
        return self.rerank(combined, text)[:top_k]
    
    def semantic_search(self, text, top_k):
        """�������"""
        query_embed = self.encode([text])[0]
        similarities = np.dot(self.embeds, query_embed)
        indices = np.argpartition(similarities, -top_k)[-top_k:]
        return [self.docs[i] for i in indices]
    
    def keyword_search(self, text, top_k):
        """�ؼ��ʼ���"""
        tokenized_query = list(jieba.cut(text))
        scores = self.bm25_index.get_scores(tokenized_query)
        indices = np.argpartition(scores, -top_k)[-top_k:]
        return [self.docs[i] for i in indices]
    
    def rerank(self, docs, query):
        """��������㷨"""
        # �����������ƶ�
        semantic_scores = [np.dot(self.encode([doc])[0], self.encode([query])[0]) for doc in docs]
        
        # ����ؼ��ʵ÷֣��������֣�
        tokenized_query = list(jieba.cut(query))
        all_bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # �����ĵ���ԭʼ������ӳ��
        doc_to_index = {doc: idx for idx, doc in enumerate(self.docs)}
        
        # ��ȡ��ǰdocs��Ӧ��BM25����
        keyword_scores = [all_bm25_scores[doc_to_index[doc]] for doc in docs]
        
        # ��Ȩ�ۺϵ÷֣��ɵ���Ȩ�أ�
        combined_scores = 0.7 * np.array(semantic_scores) + 0.3 * np.array(keyword_scores)
        
        # ���ۺϵ÷�����
        sorted_indices = np.argsort(combined_scores)[::-1]
        return [docs[i] for i in sorted_indices]

class Rag:
    def __init__(self, model, kb: Kb):
        self.model = model
        self.kb = kb
        
        # �Ż����promptģ��
        self.prompt_template = """����һ��Pythonר�ң�����������Ͻ��ش����⡣�����²���ִ�У�
1. ����ÿ�����������������ԣ�������ض����ţ�
2. ������ض��������
3. ���������������ȷ˵��

������ϣ�
%s

���⣺%s
"""

    def generate_prompt(self, text):
        """���ɲ�����prompt������������"""
        contents = self.kb.search(text)
        context = "\n".join([f"[��ض���{i+1}] {c}" for i, c in enumerate(contents)])
        return self.prompt_template % (context, text)
    
    def chat_stream(self, prompt):
        """ʹ��Ԥ���ɵ�prompt������ʽ�Ի�"""
        stream = chat(
            model=self.model,
            messages=[Message(role='system', content=prompt)],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']

# ��������
if __name__ == "__main__":
    # ��ʼ��֪ʶ���RAG
    print("��ʼ��֪ʶ���RAG��......")
    kb = Kb("python_test1.txt")
    rag = Rag('deepseek-r1:8b', kb)
    print("֪ʶ�⹹�����!!!")
    print()
    
    while True:
        q = input('Human: ')
        if q == "#Exit":
            break
        
        # ���ɲ���ʾprompt
        prompt = rag.generate_prompt(q)  # ֻ���ɲ���������
        print("\n" + "="*40 + " DEBUG PROMPT " + "="*40)
        print(prompt)
        print("="*93 + "\n")
        
        # ��ʽ����ش�
        print('Assistant:', end='', flush=True)
        full_response = []
        for chunk in rag.chat_stream(prompt):  # ʹ��ͬһprompt
            print(chunk, end='', flush=True)
            full_response.append(chunk)
        print()
