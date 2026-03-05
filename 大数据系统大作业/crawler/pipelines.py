"""
Scrapy Pipelines
"""
import os
import hashlib
from urllib.parse import urlparse
from scrapy.exceptions import DropItem
from scrapy.pipelines.files import FilesPipeline
from scrapy.http import Request
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.hbase_client import HBaseClient
from storage.data_model import Document
from utils.text_processor import extract_text_from_html, clean_text
from utils.file_handler import get_file_type, extract_text_from_file, get_file_size


class DuplicatesPipeline:
    """
    去重Pipeline：基于URL去重
    """
    
    def __init__(self):
        self.urls_seen = set()
    
    def process_item(self, item, spider):
        if item['url'] in self.urls_seen:
            raise DropItem(f"Duplicate item found: {item['url']}")
        else:
            self.urls_seen.add(item['url'])
            return item


class FileDownloadPipeline(FilesPipeline):
    """
    文件下载Pipeline
    """
    
    def get_media_requests(self, item, info):
        """
        如果item是文件，则下载
        """
        if item.get('is_file') and item.get('url'):
            yield Request(item['url'], meta={'item': item})
    
    def file_path(self, request, response=None, info=None, *, item=None):
        """
        生成文件保存路径
        """
        if item is None:
            item = request.meta.get('item')
        
        url = item['url']
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or '.' not in filename:
            # 如果没有文件名，使用URL的hash
            filename = hashlib.md5(url.encode()).hexdigest()
            # 根据Content-Type添加扩展名
            if response:
                content_type = response.headers.get('Content-Type', b'').decode()
                if 'pdf' in content_type.lower():
                    filename += '.pdf'
                elif 'doc' in content_type.lower():
                    filename += '.doc'
                elif 'docx' in content_type.lower():
                    filename += '.docx'
                elif 'xls' in content_type.lower() or 'excel' in content_type.lower():
                    filename += '.xls'
        
        # 按文件类型组织目录
        file_type = get_file_type(filename)
        return f"{file_type}/{filename}"
    
    def item_completed(self, results, item, info):
        """
        文件下载完成后的处理
        """
        if results:
            ok, result = results[0]
            if ok:
                file_path = result['path']
                item['file_path'] = os.path.join(
                    info.spider.settings.get('FILES_STORE'),
                    file_path
                )
                item['file_size'] = get_file_size(item['file_path'])
                
                # 尝试提取文件文本内容
                text_content = extract_text_from_file(item['file_path'])
                if text_content:
                    item['content'] = clean_text(text_content)
        
        return item


class StoragePipeline:
    """
    存储Pipeline：将数据存储到HBase
    """
    
    def __init__(self):
        self.hbase_client = None
    
    def open_spider(self, spider):
        """
        Spider启动时初始化HBase客户端
        """
        self.hbase_client = HBaseClient()
    
    def close_spider(self, spider):
        """
        Spider关闭时关闭HBase连接
        """
        if self.hbase_client:
            self.hbase_client.close()
    
    def process_item(self, item, spider):
        """
        处理item，存储到HBase
        """
        try:
            # 创建Document对象
            doc = Document(
                url=item.get('url', ''),
                title=item.get('title', ''),
                content=item.get('content', ''),
                file_type=item.get('file_type', 'html'),
                file_size=item.get('file_size', 0),
                source=item.get('source', ''),
                file_path=item.get('file_path')
            )
            
            # 保存到HBase
            row_key = self.hbase_client.save_document(doc)
            spider.logger.info(f"Saved document: {doc.url} -> {row_key}")
            
        except Exception as e:
            spider.logger.error(f"Error storing item: {e}")
        
        return item


