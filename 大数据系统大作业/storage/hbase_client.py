"""
HBase客户端封装
"""
import yaml
import os
from typing import Optional, List, Dict
from pathlib import Path
from storage.data_model import Document


class HBaseClient:
    """
    HBase客户端，用于连接和操作HBase
    如果HBase不可用，使用本地文件作为fallback
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化HBase客户端
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent.parent, 'config', 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.hbase_config = config.get('hbase', {})
        self.storage_config = config.get('storage', {})
        self.host = self.hbase_config.get('host', 'localhost')
        self.port = self.hbase_config.get('port', 9090)
        self.table_name = self.hbase_config.get('table_name', 'ustc_documents')
        self.index_table_name = self.hbase_config.get('index_table_name', 'ustc_index')
        
        # 尝试连接HBase
        self.connection = None
        self.use_hbase = False
        self._init_connection()
        
        # 如果HBase不可用，使用本地存储
        if not self.use_hbase:
            self.local_storage_path = os.path.join(
                Path(__file__).parent.parent,
                self.storage_config.get('file_storage_path', './data/storage')
            )
            os.makedirs(self.local_storage_path, exist_ok=True)
            print(f"Warning: HBase not available, using local storage at {self.local_storage_path}")
    
    def _init_connection(self):
        """
        初始化HBase连接
        """
        try:
            import happybase
            self.connection = happybase.Connection(
                host=self.host,
                port=self.port,
                timeout=5000
            )
            # 测试连接
            self.connection.tables()
            self.use_hbase = True
            print("HBase connection established")
            
            # 创建表（如果不存在）
            self._create_tables()
        except Exception as e:
            print(f"HBase connection failed: {e}")
            print("Will use local file storage instead")
            self.use_hbase = False
    
    def _create_tables(self):
        """
        创建HBase表（如果不存在）
        """
        if not self.use_hbase:
            return
        
        try:
            tables = self.connection.tables()
            
            # 创建文档表
            if self.table_name.encode() not in tables:
                self.connection.create_table(
                    self.table_name,
                    {
                        'info': dict(),  # 元数据列族
                        'file': dict(),  # 文件数据列族（可选）
                    }
                )
                print(f"Created table: {self.table_name}")
            
            # 创建索引表
            if self.index_table_name.encode() not in tables:
                self.connection.create_table(
                    self.index_table_name,
                    {
                        'index': dict(),  # 倒排索引列族
                    }
                )
                print(f"Created table: {self.index_table_name}")
        except Exception as e:
            print(f"Error creating tables: {e}")
    
    def _generate_row_key(self, url: str) -> str:
        """
        生成Row Key
        """
        import hashlib
        from datetime import datetime
        
        # 使用URL的hash + 时间戳
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{url_hash}_{timestamp}"
    
    def save_document(self, doc: Document) -> str:
        """
        保存文档到HBase或本地存储
        
        Returns:
            row_key: 文档的row key
        """
        row_key = self._generate_row_key(doc.url)
        
        if self.use_hbase:
            return self._save_to_hbase(doc, row_key)
        else:
            return self._save_to_local(doc, row_key)
    
    def _save_to_hbase(self, doc: Document, row_key: str) -> str:
        """
        保存到HBase，带重试机制
        """
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if not self.connection:
                    self._init_connection()
                
                table = self.connection.table(self.table_name)
                
                data = doc.to_dict()
                
                # 准备HBase数据
                hbase_data = {}
                for key, value in data.items():
                    if value:
                        hbase_data[f'info:{key}'] = str(value).encode('utf-8')
                
                # 如果文件路径存在，可以存储文件信息
                if doc.file_path:
                    hbase_data['info:file_path'] = doc.file_path.encode('utf-8')
                
                table.put(row_key.encode(), hbase_data)
                return row_key
            except Exception as e:
                print(f"Error saving to HBase (attempt {attempt + 1}/{max_retries}): {e}")
                
                # 尝试重连
                print("Attempting to reconnect to HBase...")
                try:
                    self.close()
                    time.sleep(1) # 等待一秒再重连
                    self._init_connection()
                except Exception as reconnect_error:
                    print(f"Reconnection failed: {reconnect_error}")
                
                # 如果是最后一次尝试，则抛出异常
                if attempt == max_retries - 1:
                    raise

    def _save_to_local(self, doc: Document, row_key: str) -> str:
        """
        保存到本地文件（JSON格式）
        """
        import json
        
        doc_file = os.path.join(self.local_storage_path, f"{row_key}.json")
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
        
        return row_key
    
    def get_document(self, row_key: str) -> Optional[Document]:
        """
        从HBase或本地存储获取文档
        """
        if self.use_hbase:
            return self._get_from_hbase(row_key)
        else:
            return self._get_from_local(row_key)
    
    def _get_from_hbase(self, row_key: str) -> Optional[Document]:
        """
        从HBase获取文档
        """
        try:
            table = self.connection.table(self.table_name)
            row = table.row(row_key.encode())
            
            if not row:
                return None
            
            # 解析数据
            data = {}
            for key, value in row.items():
                col_family, col_name = key.decode().split(':')
                if col_family == 'info':
                    data[col_name] = value.decode('utf-8')
            
            return Document.from_dict(data)
        except Exception as e:
            print(f"Error getting from HBase: {e}")
            return None
    
    def _get_from_local(self, row_key: str) -> Optional[Document]:
        """
        从本地文件获取文档
        """
        import json
        
        doc_file = os.path.join(self.local_storage_path, f"{row_key}.json")
        if not os.path.exists(doc_file):
            return None
        
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Document.from_dict(data)
        except Exception as e:
            print(f"Error getting from local: {e}")
            return None
    
    def get_all_documents(self, limit: Optional[int] = None) -> List[Document]:
        """
        获取所有文档
        """
        documents = []
        
        if self.use_hbase:
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if not self.connection:
                        self._init_connection()
                    
                    table = self.connection.table(self.table_name)
                    count = 0
                    batch_size = 0
                    last_key = None
                    
                    try:
                        for key, data in table.scan():
                            if limit and count >= limit:
                                break
                            
                            try:
                                row_data = {}
                                for col_key, col_value in data.items():
                                    col_family, col_name = col_key.decode().split(':')
                                    if col_family == 'info':
                                        row_data[col_name] = col_value.decode('utf-8')
                                
                                doc = Document.from_dict(row_data)
                                documents.append(doc)
                                count += 1
                                batch_size += 1
                                last_key = key
                                
                                # 每处理 1000 条打印一次进度
                                if batch_size % 1000 == 0:
                                    print(f"Loaded {batch_size} documents...")
                            except Exception as row_error:
                                print(f"Error processing row {key}: {row_error}")
                                continue
                        
                        print(f"Successfully loaded {len(documents)} documents from HBase")
                        # 如果加载的文档数量太少，可能是扫描提前中断了
                        if len(documents) < 100 and attempt < max_retries - 1:
                            print(f"Warning: Only {len(documents)} documents loaded, may be incomplete. Retrying...")
                            documents = []  # 清空，准备重试
                            raise Exception("Incomplete scan detected")
                        
                        break  # 成功则退出重试循环
                    except (BrokenPipeError, ConnectionError, OSError) as scan_error:
                        print(f"Connection error during scan (loaded {len(documents)} so far): {scan_error}")
                        if attempt < max_retries - 1:
                            # 如果已经加载了一些文档，先保存，然后重试
                            if len(documents) > 0:
                                print(f"Partial results: {len(documents)} documents, will retry for complete scan")
                            documents = []  # 清空，准备完整重试
                            raise  # 重新抛出异常，触发重连逻辑
                        else:
                            # 最后一次尝试，返回已加载的部分结果
                            print(f"Returning partial results: {len(documents)} documents")
                            break
                except Exception as e:
                    print(f"Error scanning HBase (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print("Attempting to reconnect...")
                        try:
                            self.close()
                            time.sleep(1)
                            self._init_connection()
                        except:
                            pass
                    else:
                        print(f"Failed to load documents after {max_retries} attempts")
        else:
            # 从本地文件读取
            import json
            count = 0
            for filename in os.listdir(self.local_storage_path):
                if filename.endswith('.json'):
                    if limit and count >= limit:
                        break
                    
                    doc_file = os.path.join(self.local_storage_path, filename)
                    try:
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        doc = Document.from_dict(data)
                        documents.append(doc)
                        count += 1
                    except:
                        continue
        
        return documents
    
    def save_index(self, term: str, doc_ids: List[str], term_freq: Dict[str, int]):
        """
        保存倒排索引到HBase
        """
        if self.use_hbase:
            import time
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if not self.connection:
                        self._init_connection()
                        
                    table = self.connection.table(self.index_table_name)
                    
                    # 存储文档ID列表和词频
                    data = {
                        b'index:doc_ids': ','.join(doc_ids).encode('utf-8'),
                        b'index:term_freq': str(term_freq).encode('utf-8'),
                        b'index:doc_count': str(len(doc_ids)).encode('utf-8')
                    }
                    
                    table.put(term.encode(), data)
                    break # 成功则退出循环
                except Exception as e:
                    print(f"Error saving index to HBase (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    # 尝试重连
                    try:
                        self.close()
                        time.sleep(1)
                        self._init_connection()
                    except:
                        pass
                        
                    if attempt == max_retries - 1:
                        print(f"Failed to save index for term: {term}")
        else:
            # 保存到本地文件
            import json
            index_file = os.path.join(self.local_storage_path, 'index', f"{term}.json")
            os.makedirs(os.path.dirname(index_file), exist_ok=True)
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'term': term,
                    'doc_ids': doc_ids,
                    'term_freq': term_freq
                }, f, ensure_ascii=False)
    
    def get_index(self, term: str) -> Optional[Dict]:
        """
        获取倒排索引
        """
        if self.use_hbase:
            try:
                table = self.connection.table(self.index_table_name)
                row = table.row(term.encode())
                
                if not row:
                    return None
                
                doc_ids_str = row.get(b'index:doc_ids', b'').decode('utf-8')
                doc_ids = doc_ids_str.split(',') if doc_ids_str else []
                
                return {
                    'term': term,
                    'doc_ids': doc_ids,
                    'doc_count': len(doc_ids)
                }
            except Exception as e:
                print(f"Error getting index from HBase: {e}")
                return None
        else:
            # 从本地文件读取
            import json
            index_file = os.path.join(self.local_storage_path, 'index', f"{term}.json")
            if not os.path.exists(index_file):
                return None
            
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
    
    def close(self):
        """
        关闭连接
        """
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
