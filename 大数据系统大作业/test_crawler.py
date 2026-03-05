#!/usr/bin/env python
"""
测试爬虫功能
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_basic_crawl():
    """
    测试基础爬取功能
    """
    import requests
    from bs4 import BeautifulSoup
    from utils.text_processor import extract_text_from_html

    # 测试科大主页
    test_url = "https://www.ustc.edu.cn/"

    try:
        print(f"测试访问: {test_url}")
        response = requests.get(test_url, timeout=10, verify=False)  # 禁用SSL验证

        if response.status_code == 200:
            print("✓ 成功访问科大主页")

            # 解析HTML
            soup = BeautifulSoup(response.text, 'lxml')
            title = soup.find('title')
            if title:
                print(f"✓ 页面标题: {title.get_text().strip()}")

            # 提取文本
            text = extract_text_from_html(response.text)
            print(f"✓ 提取文本长度: {len(text)} 字符")
            print(f"✓ 文本预览: {text[:100]}...")

            # 保存测试文档
            from storage.hbase_client import HBaseClient
            from storage.data_model import Document

            client = HBaseClient()
            doc = Document(
                url=test_url,
                title=title.get_text().strip() if title else "中国科学技术大学",
                content=text[:5000],  # 限制内容长度
                file_type="html",
                source="www.ustc.edu.cn"
            )

            row_key = client.save_document(doc)
            print(f"✓ 文档保存成功，Row Key: {row_key}")
            client.close()

            return True
        else:
            print(f"✗ HTTP错误: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ 网络错误: {e}")
        return False

def test_search():
    """
    测试搜索功能
    """
    from search.searcher import Searcher

    print("\n" + "=" * 50)
    print("测试搜索功能")
    print("=" * 50)

    searcher = Searcher()

    test_queries = ["科大", "中国科学技术大学", "USTC"]

    for query in test_queries:
        print(f"\n搜索: '{query}'")
        results = searcher.search(query, max_results=5)

        if results:
            print(f"✓ 找到 {len(results)} 个结果")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.4f}] {doc.title[:30]}...")
        else:
            print("✗ 未找到结果")

if __name__ == '__main__':
    print("=" * 50)
    print("测试爬虫和搜索功能")
    print("=" * 50)

    success = test_basic_crawl()
    if success:
        test_search()
    else:
        print("爬取失败，跳过搜索测试")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
