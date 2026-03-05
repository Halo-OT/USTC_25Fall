"""
搜索引擎
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.hbase_client import HBaseClient
from storage.data_model import Document
from search.tokenizer import Tokenizer
from search.ranking import TFIDF, BM25, calculate_title_weight


class Searcher:
    """
    搜索引擎
    """
    
    def __init__(self, hbase_client: HBaseClient = None):
        """
        初始化搜索引擎
        
        Args:
            hbase_client: HBase客户端
        """
        self.hbase_client = hbase_client or HBaseClient()
        self.tokenizer = Tokenizer()
        
        # 加载配置
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        search_config = config.get('search', {})
        self.ranking_algorithm = search_config.get('ranking_algorithm', 'bm25')
        self.max_results = search_config.get('max_results', 50)
        
        # 加载索引数据
        self.documents: Dict[str, Document] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        
        self._load_index()
    
    def _load_index(self):
        """
        从HBase加载索引数据
        """
        print("Loading index from storage...")
        
        # 加载所有文档，带重试机制
        import time
        max_retries = 5
        expected_min_docs = 20000  # 期望至少加载这么多文档
        
        for attempt in range(max_retries):
            try:
                # 确保连接正常
                if not self.hbase_client.use_hbase:
                    print("HBase not available, using empty index")
                    return
                
                if not self.hbase_client.connection:
                    print("Reconnecting to HBase...")
                    self.hbase_client.close()
                    self.hbase_client = HBaseClient()
                    time.sleep(2)
                
                print(f"Loading documents (attempt {attempt + 1}/{max_retries})...")
                # 确保连接稳定
                if not self.hbase_client.connection:
                    print("Connection lost, reconnecting...")
                    self.hbase_client._init_connection()
                    time.sleep(1)
                
                all_docs = self.hbase_client.get_all_documents(limit=None)
                print(f"get_all_documents returned {len(all_docs)} documents")
                
                if len(all_docs) < expected_min_docs:
                    print(f"Warning: Only loaded {len(all_docs)} documents (expected at least {expected_min_docs}), retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        # 重新初始化连接
                        self.hbase_client.close()
                        self.hbase_client = HBaseClient()
                        continue
                    else:
                        print(f"Proceeding with {len(all_docs)} documents (less than expected)")
                
                # 成功加载，处理文档
                print(f"Processing {len(all_docs)} documents into index...")
                processed_count = 0
                skipped_count = 0
                for doc in all_docs:
                    try:
                        doc_id = self._generate_doc_id(doc.url)
                        self.documents[doc_id] = doc
                        processed_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to process document {doc.url}: {e}")
                        skipped_count += 1
                        continue

                print(f"Document processing complete: {processed_count} processed, {skipped_count} skipped")
                
                print(f"Successfully loaded {len(self.documents)} documents into search index")
                break  # 成功则退出
            except Exception as e:
                print(f"Error loading documents (attempt {attempt + 1}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(3)
                    try:
                        self.hbase_client.close()
                    except:
                        pass
                    self.hbase_client = HBaseClient()
                else:
                    print("Failed to load documents after all retries")
                    self.documents = {}  # 初始化为空，避免后续错误
    
    def _generate_doc_id(self, url: str) -> str:
        """
        生成文档ID
        """
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def _build_doc_tokens(self):
        """
        构建文档分词结果（如果还没有）
        """
        if not self.doc_tokens:
            print("Building document tokens...")
            # 使用 list() 创建副本，避免在遍历时修改字典
            doc_items = list(self.documents.items())
            for doc_id, doc in doc_items:
                title_tokens = self.tokenizer.tokenize_title(doc.title)
                content_tokens = self.tokenizer.tokenize_content(doc.content)
                self.doc_tokens[doc_id] = title_tokens + content_tokens
            print("Document tokens built")
    
    def search(self, query: str, max_results: int = None) -> List[Tuple[Document, float]]:
        """
        搜索文档
        
        Args:
            query: 查询字符串
            max_results: 最大结果数量
            
        Returns:
            (文档, 分数) 列表，按分数降序排列
        """
        if not query or not query.strip():
            return []
        
        max_results = max_results or self.max_results
        
        # 分词
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # 构建文档分词结果
        self._build_doc_tokens()
        
        # 找到包含查询词的文档
        candidate_docs = set()
        for token in query_tokens:
            found = False
            # 从倒排索引查找（如果已加载）
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token].keys())
                found = True

            # 即使有倒排索引，也要检查复合词匹配
            # 遍历所有文档，检查是否包含查询词（作为子串）
            # 使用 list() 创建副本，避免在遍历时修改字典
            doc_tokens_items = list(self.doc_tokens.items())
            for doc_id, tokens in doc_tokens_items:
                doc = self.documents.get(doc_id)
                if not doc:
                    continue
                # 检查标题和内容是否包含查询词
                if (token in doc.title or token in doc.content or
                    any(token in doc_token for doc_token in tokens)):
                    candidate_docs.add(doc_id)
        
        if not candidate_docs:
            return []
        
        # 计算相关性分数
        scores = []
        
        # 准备文档数据用于排序算法
        doc_list = []
        for doc_id in candidate_docs:
            doc = self.documents[doc_id]
            tokens = self.doc_tokens[doc_id]
            doc_list.append({
                'doc_id': doc_id,
                'doc': doc,
                'tokens': tokens
            })
        
        # 使用TF-IDF或BM25计算分数
        if self.ranking_algorithm == 'bm25':
            ranker = BM25(doc_list)
            for item in doc_list:
                doc_id = item['doc_id']
                doc = item['doc']
                tokens = item['tokens']
                
                # 计算BM25分数
                score = ranker.calculate_bm25(tokens, query_tokens)
                
                # 标题权重
                title_tokens = self.tokenizer.tokenize_title(doc.title)
                title_weight = calculate_title_weight(title_tokens, query_tokens)
                score *= title_weight
                
                scores.append((doc, score))
        else:
            # 使用TF-IDF
            ranker = TFIDF(doc_list)
            for item in doc_list:
                doc_id = item['doc_id']
                doc = item['doc']
                tokens = item['tokens']
                
                # 计算TF-IDF分数
                score = ranker.calculate_tfidf(tokens, query_tokens)
                
                # 标题权重
                title_tokens = self.tokenizer.tokenize_title(doc.title)
                title_weight = calculate_title_weight(title_tokens, query_tokens)
                score *= title_weight
                
                scores.append((doc, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前max_results个结果
        return scores[:max_results]
    
    def search_by_source(self, query: str, source: str, max_results: int = None) -> List[Tuple[Document, float]]:
        """
        按来源搜索
        
        Args:
            query: 查询字符串
            source: 来源网站
            max_results: 最大结果数量
            
        Returns:
            (文档, 分数) 列表
        """
        results = self.search(query, max_results=None)
        filtered = [(doc, score) for doc, score in results if source in doc.source]
        max_results = max_results or self.max_results
        return filtered[:max_results]


