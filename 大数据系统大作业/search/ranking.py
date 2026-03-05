"""
相关性排序算法
"""
from typing import Dict, List
from collections import Counter
import math


class TFIDF:
    """
    TF-IDF算法
    """
    
    def __init__(self, documents: List[Dict]):
        """
        初始化TF-IDF
        
        Args:
            documents: 文档列表，每个文档包含'tokens'字段
        """
        self.documents = documents
        self.doc_count = len(documents)
        self.idf_cache = {}
        self._calculate_idf()
    
    def _calculate_idf(self):
        """
        计算IDF值
        """
        # 统计每个词出现在多少文档中
        doc_freq = {}
        for doc in self.documents:
            unique_tokens = set(doc.get('tokens', []))
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # 计算IDF
        for token, freq in doc_freq.items():
            self.idf_cache[token] = math.log(self.doc_count / (freq + 1))
    
    def calculate_tfidf(self, doc_tokens: List[str], query_tokens: List[str]) -> float:
        """
        计算文档与查询的TF-IDF相似度
        
        Args:
            doc_tokens: 文档的分词结果
            query_tokens: 查询的分词结果
            
        Returns:
            TF-IDF分数
        """
        if not doc_tokens or not query_tokens:
            return 0.0
        
        # 计算文档中每个词的TF
        doc_token_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)
        
        score = 0.0
        for query_token in query_tokens:
            if query_token in doc_token_freq:
                # TF
                tf = doc_token_freq[query_token] / doc_length
                # IDF
                idf = self.idf_cache.get(query_token, 0)
                # TF-IDF
                score += tf * idf
        
        return score


class BM25:
    """
    BM25算法（改进的TF-IDF）
    """
    
    def __init__(self, documents: List[Dict], k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25
        
        Args:
            documents: 文档列表
            k1: 词频饱和度参数
            b: 长度归一化参数
        """
        self.documents = documents
        self.doc_count = len(documents)
        self.k1 = k1
        self.b = b
        
        # 计算平均文档长度
        self.avg_doc_length = sum(len(doc.get('tokens', [])) for doc in documents) / max(self.doc_count, 1)
        
        # 计算IDF
        self.idf_cache = {}
        self._calculate_idf()
    
    def _calculate_idf(self):
        """
        计算IDF值
        """
        doc_freq = {}
        for doc in self.documents:
            unique_tokens = set(doc.get('tokens', []))
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        for token, freq in doc_freq.items():
            # BM25的IDF公式
            self.idf_cache[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def calculate_bm25(self, doc_tokens: List[str], query_tokens: List[str], doc_id: int = None) -> float:
        """
        计算BM25分数
        
        Args:
            doc_tokens: 文档的分词结果
            query_tokens: 查询的分词结果
            doc_id: 文档ID（用于获取文档长度）
            
        Returns:
            BM25分数
        """
        if not doc_tokens or not query_tokens:
            return 0.0
        
        doc_length = len(doc_tokens)
        doc_token_freq = Counter(doc_tokens)
        
        score = 0.0
        for query_token in query_tokens:
            if query_token in doc_token_freq:
                # 词频
                tf = doc_token_freq[query_token]
                
                # BM25公式
                numerator = self.idf_cache.get(query_token, 0) * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1)))
                
                score += numerator / max(denominator, 1)
        
        return score


def calculate_title_weight(title_tokens: List[str], query_tokens: List[str]) -> float:
    """
    计算标题权重（标题匹配的文档应该获得更高分数）
    
    Args:
        title_tokens: 标题分词结果
        query_tokens: 查询分词结果
        
    Returns:
        权重倍数
    """
    if not title_tokens or not query_tokens:
        return 1.0
    
    # 计算标题中匹配的查询词比例
    matched = sum(1 for token in query_tokens if token in title_tokens)
    match_ratio = matched / len(query_tokens)
    
    # 如果标题完全匹配，给予较高权重
    if match_ratio >= 0.8:
        return 2.0
    elif match_ratio >= 0.5:
        return 1.5
    else:
        return 1.0


