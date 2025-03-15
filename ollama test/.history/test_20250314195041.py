# 安装必要库（运行前先执行）
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
        
        # 改进1：智能文本分块
        self.docs = self.split_content(content)
        
        # 改进2：混合检索初始化
        self.embeds = self.encode(self.docs)
        self.bm25_index = self.build_bm25_index(self.docs)
    
    @staticmethod
    def split_content(content):
        """改进的智能分块策略"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,    # 目标块大小
            chunk_overlap=50,   # 块间重叠防止断句
            separators=["\n\n", "\n", "。", "！", "？", "；"],  # 中文友好分隔符
            length_function=len
        )
        return splitter.split_text(content)
    
    def build_bm25_index(self, docs):
        """构建BM25关键词检索索引"""
        # 中文分词处理
        tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
        return BM25Okapi(tokenized_docs)
    
    def encode(self, texts):
        """嵌入编码"""
        return np.array([embeddings(model='nomic-embed-text', prompt=text)['embedding'] for text in texts])
    
    def search(self, text, top_k=3):
        """改进2：混合检索策略"""
        # 并行执行两种检索
        semantic_results = self.semantic_search(text, top_k*2)
        keyword_results = self.keyword_search(text, top_k*2)
        
        # 结果融合与去重
        combined = list({doc: None for doc in semantic_results + keyword_results}.keys())
        
        # 重新排序
        return self.rerank(combined, text)[:top_k]
    
    def semantic_search(self, text, top_k):
        """语义检索"""
        query_embed = self.encode([text])[0]
        similarities = np.dot(self.embeds, query_embed)
        indices = np.argpartition(similarities, -top_k)[-top_k:]
        return [self.docs[i] for i in indices]
    
    def keyword_search(self, text, top_k):
        """关键词检索"""
        tokenized_query = list(jieba.cut(text))
        scores = self.bm25_index.get_scores(tokenized_query)
        indices = np.argpartition(scores, -top_k)[-top_k:]
        return [self.docs[i] for i in indices]
    
    def rerank(self, docs, query):
        """混合排序算法"""
        # 计算语义相似度
        semantic_scores = [np.dot(self.encode([doc])[0], self.encode([query])[0]) for doc in docs]
        
        # 计算关键词得分（修正部分）
        tokenized_query = list(jieba.cut(query))
        all_bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 建立文档到原始索引的映射
        doc_to_index = {doc: idx for idx, doc in enumerate(self.docs)}
        
        # 获取当前docs对应的BM25分数
        keyword_scores = [all_bm25_scores[doc_to_index[doc]] for doc in docs]
        
        # 加权综合得分（可调整权重）
        combined_scores = 0.7 * np.array(semantic_scores) + 0.3 * np.array(keyword_scores)
        
        # 按综合得分排序
        sorted_indices = np.argsort(combined_scores)[::-1]
        return [docs[i] for i in sorted_indices]

class Rag:
    def __init__(self, model, kb: Kb):
        self.model = model
        self.kb = kb
        
        # 改进3：优化后的prompt模板
        self.prompt_template = """你是一个Python专家，请严格根据以下资料回答问题。按以下步骤执行：
1. 分析每个相关段落与问题的真实相关性，过滤不相关内容（保留段落编号）
3. 如果无相关内容，回答"资料中未找到相关信息"

相关资料：
%s

问题：%s

请先给出思考过程（包含相关性分析），再用###回答###标记最终答案。"""
    
    def generate_response(self, text):
        """生成完整响应"""
        contents = self.kb.search(text)
        context = "\n".join([f"[相关段落{i+1}] {c}" for i, c in enumerate(contents)])
        prompt = self.prompt_template % (context, text)
        
        response = chat(
            model=self.model,
            messages=[Message(role='system', content=prompt)],
            stream=False
        )
        return response['message']['content']
    
    def chat_stream(self, prompt):
        """接收预生成的prompt进行流式对话"""
        stream = chat(
            model=self.model,
            messages=[Message(role='system', content=prompt)],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']

# 测试用例
if __name__ == "__main__":
    # 初始化知识库和RAG
    print("初始化知识库和RAG中......")
    kb = Kb("python_test1.txt")
    rag = Rag('deepseek-r1:8b', kb)
    print("知识库构建完成!!!")
    print()
    

    while True:
        q = input('Human: ')
        
        if q == "#Exit":
            break
        
        # 生成并显示prompt
        prompt = rag.generate_response(q)
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