#!/usr/bin/env python
"""
测试搜索功能
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from storage.hbase_client import HBaseClient
from search.searcher import Searcher

if __name__ == '__main__':
    print("=" * 50)
    print("测试搜索功能")
    print("=" * 50)
    
    # 初始化搜索器
    searcher = Searcher()
    
    # 测试搜索
    test_queries = [
        "下载",
        "财务",
        "招生",
        "教务处"
    ]
    
    for query in test_queries:
        print(f"\n搜索: {query}")
        print("-" * 50)
        results = searcher.search(query, max_results=5)
        
        if results:
            for i, (doc, score) in enumerate(results, 1):
                print(f"{i}. [{score:.4f}] {doc.title}")
                print(f"   来源: {doc.source}")
                print(f"   URL: {doc.url[:80]}...")
        else:
            print("未找到结果")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


