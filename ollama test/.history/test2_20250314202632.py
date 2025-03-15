import os
import json
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import hashlib

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

    # 以下方法保持不变...
    @staticmethod
    def split_content(content):
        # 原有分块逻辑保持不变...
    
    def encode(self, texts):
        # 原有编码逻辑保持不变...

    # 其余方法保持不变...