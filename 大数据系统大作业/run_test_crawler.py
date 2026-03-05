import os
import sys
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from crawler.spiders.ustc_spider import UstcSpider

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_spider():
    # 获取默认设置
    os.environ.setdefault('SCRAPY_SETTINGS_MODULE', 'crawler.settings')
    settings = get_project_settings()
    
    # 覆盖设置以进行测试
    settings.set('CLOSESPIDER_PAGECOUNT', 30)  # 限制爬取30个页面
    settings.set('LOG_LEVEL', 'INFO')
    settings.set('DOWNLOAD_DELAY', 1)
    
    # 显式设置 Pipeline，防止加载失败
    settings.set('ITEM_PIPELINES', {
        'crawler.pipelines.StoragePipeline': 500,
    })
    
    process = CrawlerProcess(settings)
    
    # 修改 spider 的 start_urls，只测试财务处和主页
    class TestSpider(UstcSpider):
        name = 'test_spider'
        start_urls = [
            'https://www.ustc.edu.cn/',
            'https://finance.ustc.edu.cn/'
        ]
        
    process.crawl(TestSpider)
    process.start()

if __name__ == '__main__':
    run_spider()
