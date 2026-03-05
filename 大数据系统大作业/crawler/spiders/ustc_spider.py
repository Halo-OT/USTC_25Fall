"""
中科大网站爬虫
"""
import scrapy
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crawler.items import DocumentItem
from utils.text_processor import extract_text_from_html, clean_text
from utils.file_handler import get_file_type


class UstcSpider(scrapy.Spider):
    """
    中科大网站爬虫
    """
    name = 'ustc_spider'
    
    # 允许的域名
    allowed_domains = [
        'ustc.edu.cn',
        'saids.ustc.edu.cn',
        'zsb.ustc.edu.cn',
        'job.ustc.edu.cn',
        'teach.ustc.edu.cn',
        'finance.ustc.edu.cn',
        'stuhome.ustc.edu.cn',
        'yz.ustc.edu.cn',
        'bwc.ustc.edu.cn',
        'press.ustc.edu.cn',
        'ispc.ustc.edu.cn',
        'zhb.ustc.edu.cn',
        'young.ustc.edu.cn',
        'vista.ustc.edu.cn',
        'ustcnet.ustc.edu.cn',
        'zhc.ustc.edu.cn',
        'cs.ustc.edu.cn',
        'cybersec.ustc.edu.cn',
        'sgy.ustc.edu.cn',
        'math.ustc.edu.cn',
        'sist.ustc.edu.cn',
        'sz.ustc.edu.cn',
        'sse.ustc.edu.cn',
        'iat.ustc.edu.cn',
    ]
    
    # 起始URL列表
    start_urls = [
        'https://saids.ustc.edu.cn/main.htm',
        'https://zsb.ustc.edu.cn/main.htm',
        'https://www.job.ustc.edu.cn/',
        'https://www.teach.ustc.edu.cn/',
        'https://finance.ustc.edu.cn/main.htm',
        'https://stuhome.ustc.edu.cn/main.htm',
        'https://yz.ustc.edu.cn/',
        'https://bwc.ustc.edu.cn/main.htm',
        'https://press.ustc.edu.cn/main.htm',
        'https://ispc.ustc.edu.cn/_web/main.psp',
        'https://zhb.ustc.edu.cn/main.htm',
        'https://young.ustc.edu.cn/xtwryxx/list.htm',
        'https://vista.ustc.edu.cn/',
        'https://ustcnet.ustc.edu.cn/33490/list.htm',
        'https://zhc.ustc.edu.cn/main.htm',
        'https://cs.ustc.edu.cn/main.htm',
        'https://cybersec.ustc.edu.cn/main.htm',
        'https://sgy.ustc.edu.cn/main.htm',
        'https://math.ustc.edu.cn/main.htm',
        'https://sist.ustc.edu.cn/main.htm',
        'https://sz.ustc.edu.cn/index.html',
        'https://sse.ustc.edu.cn/?pageid=210/main.htm',
        'https://iat.ustc.edu.cn/iat/index.html',
    ]
    
    # 文件扩展名
    file_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.zip', '.rar']
    
    # 关键词，用于识别下载中心等页面
    download_keywords = ['下载', '下载中心', '文件下载', '资料下载', '文档下载', 'download']
    
    def parse(self, response):
        """
        解析响应
        """
        url = response.url
        parsed = urlparse(url)
        source = parsed.netloc
        
        # 检查是否是文件
        is_file = any(url.lower().endswith(ext) for ext in self.file_extensions)
        
        if is_file:
            # 如果是文件，创建文件item
            yield self.create_file_item(response, source)
        else:
            # 解析HTML页面
            yield self.create_page_item(response, source)
            
            # 查找下载链接和文件链接
            for link in self.extract_links(response):
                yield scrapy.Request(link, callback=self.parse, dont_filter=False)
    
    def create_file_item(self, response, source):
        """
        创建文件Item
        """
        item = DocumentItem()
        item['url'] = response.url
        item['title'] = self.extract_title(response)
        item['content'] = ''  # 文件内容将在pipeline中提取
        item['file_type'] = get_file_type(response.url)
        item['file_size'] = len(response.body)
        item['source'] = source
        item['is_file'] = True
        
        return item
    
    def create_page_item(self, response, source):
        """
        创建页面Item
        """
        item = DocumentItem()
        item['url'] = response.url
        item['title'] = self.extract_title(response)
        item['content'] = extract_text_from_html(response.text)
        item['file_type'] = 'html'
        item['file_size'] = len(response.body)
        item['source'] = source
        item['is_file'] = False
        
        return item
    
    def extract_title(self, response):
        """
        提取页面标题
        """
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            title_tag = soup.find('title')
            if title_tag:
                return clean_text(title_tag.get_text())
        except:
            pass
        
        # 如果没有title标签，使用URL
        parsed = urlparse(response.url)
        return parsed.path.split('/')[-1] or response.url
    
    def extract_links(self, response):
        """
        提取页面中的链接
        """
        links = set()
        
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 查找所有a标签
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                link_text = a_tag.get_text().strip()
                
                # 构建绝对URL
                absolute_url = urljoin(response.url, href)
                parsed = urlparse(absolute_url)
                
                # 检查域名
                if parsed.netloc and any(domain in parsed.netloc for domain in self.allowed_domains):
                    # 检查是否是文件链接
                    is_file_link = any(absolute_url.lower().endswith(ext) for ext in self.file_extensions)
                    
                    # 检查是否包含下载关键词
                    is_download_page = any(keyword in link_text for keyword in self.download_keywords) or \
                                     any(keyword in absolute_url.lower() for keyword in self.download_keywords)
                    
                    # 优先爬取文件链接和下载页面
                    if is_file_link or is_download_page:
                        links.add(absolute_url)
                    # 限制深度，只爬取前几层
                    elif response.meta.get('depth', 0) < 3:
                        links.add(absolute_url)
        
        except Exception as e:
            self.logger.error(f"Error extracting links from {response.url}: {e}")
        
        return links


