"""
分词器
"""
import jieba
from typing import List, Set
from utils.text_processor import tokenize


class Tokenizer:
    """
    中文分词器
    """
    
    def __init__(self, custom_dict_path: str = None):
        """
        初始化分词器
        
        Args:
            custom_dict_path: 自定义词典路径
        """
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
        
        # 添加一些科大相关的词汇
        self._add_ustc_words()
    
    def _add_ustc_words(self):
        """
        添加科大相关词汇到词典
        """
        ustc_words = [
            '中科大', '科大', 'USTC', '中国科学技术大学',
            '教务处', '财务处', '学工处', '招生办',
            '计算机学院', '数学学院', '物理学院',
            '下载中心', '文件下载'
        ]
        
        for word in ustc_words:
            jieba.add_word(word)
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        return tokenize(text)
    
    def tokenize_title(self, title: str) -> List[str]:
        """
        对标题进行分词（可能需要不同的处理）
        """
        return self.tokenize(title)
    
    def tokenize_content(self, content: str) -> List[str]:
        """
        对内容进行分词
        """
        return self.tokenize(content)


