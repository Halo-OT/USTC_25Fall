#!/usr/bin/env python
"""
构建倒排索引脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from storage.hbase_client import HBaseClient
from search.indexer import Indexer

if __name__ == '__main__':
    print("=" * 50)
    print("开始构建倒排索引")
    print("=" * 50)
    
    # 初始化
    hbase_client = HBaseClient()
    indexer = Indexer(hbase_client)
    
    # 构建索引
    indexer.build_index()
    
    print("=" * 50)
    print("索引构建完成")
    print("=" * 50)
    
    hbase_client.close()


