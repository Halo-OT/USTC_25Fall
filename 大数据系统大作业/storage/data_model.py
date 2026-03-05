"""
数据模型定义
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Document:
    """
    文档数据模型
    """
    url: str
    title: str
    content: str
    file_type: str
    file_size: int = 0
    source: str = ""
    crawl_time: Optional[datetime] = None
    file_path: Optional[str] = None  # 文件本地路径（如果存储在本地）
    
    def __post_init__(self):
        if self.crawl_time is None:
            self.crawl_time = datetime.now()
    
    def to_dict(self) -> dict:
        """
        转换为字典格式，用于存储到HBase
        """
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'file_type': self.file_type,
            'file_size': str(self.file_size),
            'source': self.source,
            'crawl_time': self.crawl_time.isoformat() if self.crawl_time else '',
            'file_path': self.file_path or ''
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Document':
        """
        从字典创建Document对象
        """
        crawl_time = None
        if data.get('crawl_time'):
            try:
                crawl_time = datetime.fromisoformat(data['crawl_time'])
            except:
                pass
        
        return cls(
            url=data.get('url', ''),
            title=data.get('title', ''),
            content=data.get('content', ''),
            file_type=data.get('file_type', ''),
            file_size=int(data.get('file_size', 0)),
            source=data.get('source', ''),
            crawl_time=crawl_time,
            file_path=data.get('file_path')
        )


