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
        
        # ����Ψһ�����ʶ
        self.file_hash = self._get_file_hash()
        
        # ���Լ��ػ���
        if not self._load_cache():
            # ���治���������´���
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.docs = self.split_content(content)
            self.tokenized_docs = [list(jieba.cut(doc)) for doc in self.docs]
            self.embeds = self.encode(self.docs)
            self.bm25_index = BM25Okapi(self.tokenized_docs)
            self._save_cache()

    def _get_file_hash(self):
        """�����ļ����ݹ�ϣֵ��Ϊ�����ʶ"""
        with open(self.filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _cache_path(self, filename):
        return os.path.join(self.cache_dir, f"{self.file_hash}_{filename}")

    def _save_cache(self):
        """�������л�������"""
        # �����ĵ��б�
        with open(self._cache_path("docs.json"), 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        
        # ����ִʽ��
        with open(self._cache_path("tokenized_docs.json"), 'w', encoding='utf-8') as f:
            json.dump(self.tokenized_docs, f, ensure_ascii=False)
        
        # ����Embedding����
        np.save(self._cache_path("embeds.npy"), self.embeds)
        
        # ����BM25����������BM25Okapi����ֱ�����л��������Ҫ������
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
        """���ػ�������"""
        try:
            # �����ĵ��б�
            with open(self._cache_path("docs.json"), 'r', encoding='utf-8') as f:
                self.docs = json.load(f)
            
            # ���طִʽ��
            with open(self._cache_path("tokenized_docs.json"), 'r', encoding='utf-8') as f:
                self.tokenized_docs = json.load(f)
            
            # ����Embedding
            self.embeds = np.load(self._cache_path("embeds.npy"))
            
            # �ؽ�BM25����
            with open(self._cache_path("bm25_params.json"), 'r') as f:
                bm25_params = json.load(f)
            
            self.bm25_index = BM25Okapi(
                corpus=self.tokenized_docs,
                k1=bm25_params['k1'],
                b=bm25_params['b']
            )
            # �ֶ������ڲ�����
            self.bm25_index.doc_freqs = bm25_params['doc_freqs']
            self.bm25_index.doc_len = bm25_params['doc_len']
            self.bm25_index.avgdl = bm25_params['avgdl']
            
            return True
        except FileNotFoundError:
            return False

    # ���·������ֲ���...
    @staticmethod
    def split_content(content):
        # ԭ�зֿ��߼����ֲ���...
    
    def encode(self, texts):
        # ԭ�б����߼����ֲ���...

    # ���෽�����ֲ���...