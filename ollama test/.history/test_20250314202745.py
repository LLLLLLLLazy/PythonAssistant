import os
import json
import hashlib
from ollama import chat, Message, embeddings
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Kb:
    def __init__(self, filepath, cache_dir="kb_cache"):
        self.filepath = filepath
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成唯一缓存标识
        self.file_hash = self._get_file_hash()
        
        # 尝试加载缓存
        if not self._load_cache():
            # 缓存不存在则重新处理
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.docs = self.split_content(content)
            self.tokenized_docs = [list(jieba.cut(doc)) for doc in self.docs]
            self.embeds = self.encode(self.docs)
            self.bm25_index = BM25Okapi(self.tokenized_docs)
            self._save_cache()

    def _get_file_hash(self):
        """生成文件内容哈希值作为缓存标识"""
        with open(self.filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _cache_path(self, filename):
        return os.path.join(self.cache_dir, f"{self.file_hash}_{filename}")

    def _save_cache(self):
        """保存所有缓存数据"""
        # 保存文档列表
        with open(self._cache_path("docs.json"), 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        
        # 保存分词结果
        with open(self._cache_path("tokenized_docs.json"), 'w', encoding='utf-8') as f:
            json.dump(self.tokenized_docs, f, ensure_ascii=False)
        
        # 保存Embedding向量
        np.save(self._cache_path("embeds.npy"), self.embeds)
        
        # 保存BM25参数（由于BM25Okapi不可直接序列化，保存必要参数）
        bm25_params = {
            'doc_freqs': self.bm25_index.doc_freqs,
            'doc_len': self.bm25_index.doc_len,
            'avgdl': self.bm25_index.avgdl,
            'k1': self.bm25_index.k1,
            'b': self.bm25_index.b
        }
        with open(self._cache_path("bm25_params.json"), 'w') as f:
            json.dump(bm25_params, f)

    def _load_cache(self):
        """加载缓存数据"""
        try:
            # 加载文档列表
            with open(self._cache_path("docs.json"), 'r', encoding='utf-8') as f:
                self.docs = json.load(f)
            
            # 加载分词结果
            with open(self._cache_path("tokenized_docs.json"), 'r', encoding='utf-8') as f:
                self.tokenized_docs = json.load(f)
            
            # 加载Embedding
            self.embeds = np.load(self._cache_path("embeds.npy"))
            
            # 重建BM25索引
            with open(self._cache_path("bm25_params.json"), 'r') as f:
                bm25_params = json.load(f)
            
            self.bm25_index = BM25Okapi(
                corpus=self.tokenized_docs,
                k1=bm25_params['k1'],
                b=bm25_params['b']
            )
            # 手动设置内部参数
            self.bm25_index.doc_freqs = bm25_params['doc_freqs']
            self.bm25_index.doc_len = bm25_params['doc_len']
            self.bm25_index.avgdl = bm25_params['avgdl']
            
            return True
        except FileNotFoundError:
            return False
    
    @staticmethod
    def split_content(content):
        """改进的智能分块策略"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,    # 目标块大小
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
        
        # 优化后的prompt模板
        self.prompt_template = """你是一个Python专家，请根据资料严谨回答问题，若无相关内容则按照原有知识库回答问题：

相关资料：
%s

问题：%s
"""

    def generate_prompt(self, text):
        """生成并返回prompt（不发送请求）"""
        contents = self.kb.search(text)
        context = "\n".join([f"[相关段落{i+1}] {c}" for i, c in enumerate(contents)])
        return self.prompt_template % (context, text)
    
    def chat_stream(self, prompt):
        """使用预生成的prompt进行流式对话"""
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
        # prompt = rag.generate_prompt(q)  # 只生成不发送请求
        # print("\n" + "="*40 + " DEBUG PROMPT " + "="*40)
        # print(prompt)
        # print("="*93 + "\n")
        
        # 流式输出回答
        print('Assistant:', end='', flush=True)
        full_response = []
        for chunk in rag.chat_stream(prompt):  # 使用同一prompt
            print(chunk, end='', flush=True)
            full_response.append(chunk)
        print()