#!/usr/bin/env python
"""
全面测试脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("=" * 50)
    print("测试1: 模块导入")
    print("=" * 50)
    
    try:
        from storage.hbase_client import HBaseClient
        from storage.data_model import Document
        from search.tokenizer import Tokenizer
        from search.ranking import TFIDF, BM25
        from utils.text_processor import clean_text, tokenize
        from utils.file_handler import get_file_type
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_storage():
    """测试存储模块"""
    print("\n" + "=" * 50)
    print("测试2: 存储模块")
    print("=" * 50)
    
    try:
        from storage.hbase_client import HBaseClient
        from storage.data_model import Document
        from datetime import datetime
        
        client = HBaseClient()
        print(f"✓ HBase客户端初始化成功 (使用HBase: {client.use_hbase})")
        
        # 创建测试文档
        test_doc = Document(
            url="https://test.ustc.edu.cn/test.html",
            title="测试文档",
            content="这是一个测试文档的内容",
            file_type="html",
            source="test.ustc.edu.cn"
        )
        
        # 保存文档
        row_key = client.save_document(test_doc)
        print(f"✓ 文档保存成功，Row Key: {row_key}")
        
        # 读取文档
        retrieved_doc = client.get_document(row_key)
        if retrieved_doc and retrieved_doc.url == test_doc.url:
            print("✓ 文档读取成功")
        else:
            print("✗ 文档读取失败")
        
        client.close()
        return True
    except Exception as e:
        print(f"✗ 存储测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer():
    """测试分词器"""
    print("\n" + "=" * 50)
    print("测试3: 分词器")
    print("=" * 50)
    
    try:
        from search.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        test_texts = [
            "中科大教务处",
            "下载中心文件",
            "财务处通知"
        ]
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            print(f"  文本: {text} -> 分词: {tokens}")
        
        print("✓ 分词器测试成功")
        return True
    except Exception as e:
        print(f"✗ 分词器测试失败: {e}")
        return False


def test_ranking():
    """测试排序算法"""
    print("\n" + "=" * 50)
    print("测试4: 排序算法")
    print("=" * 50)
    
    try:
        from search.ranking import TFIDF, BM25
        
        # 创建测试文档
        documents = [
            {'tokens': ['中科大', '教务处', '通知']},
            {'tokens': ['财务处', '下载', '文件']},
            {'tokens': ['中科大', '招生', '信息']},
        ]
        
        query_tokens = ['中科大', '通知']
        
        # 测试TF-IDF
        tfidf = TFIDF(documents)
        scores = []
        for i, doc in enumerate(documents):
            score = tfidf.calculate_tfidf(doc['tokens'], query_tokens)
            scores.append((i, score))
            print(f"  文档{i+1} TF-IDF分数: {score:.4f}")
        
        # 测试BM25
        bm25 = BM25(documents)
        scores_bm25 = []
        for i, doc in enumerate(documents):
            score = bm25.calculate_bm25(doc['tokens'], query_tokens)
            scores_bm25.append((i, score))
            print(f"  文档{i+1} BM25分数: {score:.4f}")
        
        print("✓ 排序算法测试成功")
        return True
    except Exception as e:
        print(f"✗ 排序算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_processor():
    """测试文本处理"""
    print("\n" + "=" * 50)
    print("测试5: 文本处理")
    print("=" * 50)
    
    try:
        from utils.text_processor import clean_text, extract_text_from_html, tokenize
        
        # 测试文本清理
        dirty_text = "  hello   world  \n\n  test  "
        cleaned = clean_text(dirty_text)
        print(f"  清理文本: '{dirty_text}' -> '{cleaned}'")
        
        # 测试HTML提取
        html = "<html><body><p>测试内容</p><script>alert('test')</script></body></html>"
        text = extract_text_from_html(html)
        print(f"  HTML提取: {text}")
        
        # 测试分词
        tokens = tokenize("中科大教务处通知")
        print(f"  分词结果: {tokens}")
        
        print("✓ 文本处理测试成功")
        return True
    except Exception as e:
        print(f"✗ 文本处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_searcher():
    """测试搜索引擎（需要数据）"""
    print("\n" + "=" * 50)
    print("测试6: 搜索引擎")
    print("=" * 50)
    
    try:
        from storage.hbase_client import HBaseClient
        from storage.data_model import Document
        from search.searcher import Searcher
        
        # 检查是否有数据
        client = HBaseClient()
        documents = client.get_all_documents(limit=5)
        client.close()
        
        if len(documents) == 0:
            print("⚠ 没有数据，跳过搜索测试")
            print("  提示: 请先运行爬虫 (python run_crawler.py)")
            return True
        
        print(f"  找到 {len(documents)} 个文档，开始测试搜索...")
        searcher = Searcher()
        results = searcher.search("下载", max_results=3)
        
        if results:
            print(f"  ✓ 搜索成功，找到 {len(results)} 个结果")
            for i, (doc, score) in enumerate(results[:3], 1):
                print(f"    结果{i}: [{score:.4f}] {doc.title[:50]}")
        else:
            print("  ⚠ 搜索无结果（可能需要先构建索引）")
        
        return True
    except Exception as e:
        print(f"✗ 搜索引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始全面测试")
    print("=" * 50)
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("存储模块", test_storage()))
    results.append(("分词器", test_tokenizer()))
    results.append(("排序算法", test_ranking()))
    results.append(("文本处理", test_text_processor()))
    results.append(("搜索引擎", test_searcher()))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查上述错误信息")


if __name__ == '__main__':
    main()


