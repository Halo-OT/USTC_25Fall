"""
倒排索引构建器
"""
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.hbase_client import HBaseClient
from storage.data_model import Document
from search.tokenizer import Tokenizer


class Indexer:
    """
    倒排索引构建器
    """
    
    def __init__(self, hbase_client: HBaseClient = None):
        """
        初始化索引器
        
        Args:
            hbase_client: HBase客户端
        """
        self.hbase_client = hbase_client or HBaseClient()
        self.tokenizer = Tokenizer()
        
        # 倒排索引：词 -> {文档ID: 词频}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # 文档信息：文档ID -> 文档
        self.documents: Dict[str, Document] = {}
        
        # 文档分词结果：文档ID -> 分词列表
        self.doc_tokens: Dict[str, List[str]] = {}
    
    def build_index(self, limit: int = None):
        """
        构建倒排索引
        
        Args:
            limit: 限制处理的文档数量
        """
        print("Loading documents from storage...")
        documents = self.hbase_client.get_all_documents(limit=limit)
        print(f"Loaded {len(documents)} documents")
        
        print("Building inverted index...")
        for idx, doc in enumerate(documents):
            if idx % 100 == 0:
                print(f"Processing document {idx + 1}/{len(documents)}")
            
            # 生成文档ID
            doc_id = self._generate_doc_id(doc.url)
            self.documents[doc_id] = doc
            
            # 分词
            title_tokens = self.tokenizer.tokenize_title(doc.title)
            content_tokens = self.tokenizer.tokenize_content(doc.content)
            all_tokens = title_tokens + content_tokens
            
            # 存储分词结果
            self.doc_tokens[doc_id] = all_tokens
            
            # 构建倒排索引
            token_freq = {}
            for token in all_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            for token, freq in token_freq.items():
                self.inverted_index[token][doc_id] = freq
        
        print(f"Index built with {len(self.inverted_index)} unique terms")
        
        # 保存索引到HBase
        print("Saving index to storage...")
        self._save_index()
        print("Index saved successfully")
    
    def _generate_doc_id(self, url: str) -> str:
        """
        生成文档ID
        """
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def _save_index(self):
        """
        保存倒排索引到HBase
        """
        for term, doc_freq in self.inverted_index.items():
            doc_ids = list(doc_freq.keys())
            self.hbase_client.save_index(term, doc_ids, doc_freq)
    
    def get_documents(self) -> Dict[str, Document]:
        """
        获取所有文档
        """
        return self.documents
    
    def get_doc_tokens(self) -> Dict[str, List[str]]:
        """
        获取所有文档的分词结果
        """
        return self.doc_tokens
    
    def get_inverted_index(self) -> Dict[str, Dict[str, int]]:
        """
        获取倒排索引
        """
        return self.inverted_index


