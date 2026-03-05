"""
文本处理工具函数
"""
import re
import jieba
from typing import List, Set


def clean_text(text: str) -> str:
    """
    清理文本：去除多余空白、特殊字符等
    """
    if not text:
        return ""
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    
    return text


def extract_text_from_html(html: str) -> str:
    """
    从HTML中提取纯文本
    """
    from bs4 import BeautifulSoup
    import re

    # 尝试检测编码，如果是bytes，先解码
    if isinstance(html, bytes):
        # 尝试多种编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'iso-8859-1']:
            try:
                html = html.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # 如果都失败，使用errors='ignore'
            html = html.decode('utf-8', errors='ignore')

    soup = BeautifulSoup(html, 'lxml')
    # 移除script和style标签
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text()
    return clean_text(text)


def tokenize(text: str, cut_all: bool = False) -> List[str]:
    """
    中文分词
    """
    if not text:
        return []
    
    # 使用jieba分词
    words = jieba.cut(text, cut_all=cut_all)
    # 过滤停用词和单字符
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    
    tokens = [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in stop_words]
    return tokens


def calculate_simhash(text: str, hash_bits: int = 64) -> int:
    """
    计算文本的SimHash值，用于去重
    """
    import hashlib
    
    tokens = tokenize(text)
    if not tokens:
        return 0
    
    # 计算每个词的hash值
    v = [0] * hash_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    
    # 生成simhash
    simhash = 0
    for i in range(hash_bits):
        if v[i] > 0:
            simhash |= (1 << i)
    
    return simhash


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    计算两个SimHash的汉明距离
    """
    x = hash1 ^ hash2
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance


