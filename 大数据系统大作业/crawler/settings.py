"""
Scrapy设置
"""
import yaml
import os
from pathlib import Path

# 加载配置
config_path = os.path.join(Path(__file__).parent.parent, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

crawler_config = config.get('crawler', {})

# Scrapy基本设置
BOT_NAME = 'ustc_crawler'

SPIDER_MODULES = ['crawler.spiders']
NEWSPIDER_MODULE = 'crawler.spiders'

# 遵守robots.txt
ROBOTSTXT_OBEY = crawler_config.get('respect_robots_txt', True)

# 下载延迟
DOWNLOAD_DELAY = crawler_config.get('download_delay', 1.0)
RANDOMIZE_DOWNLOAD_DELAY = True

# 并发请求数
CONCURRENT_REQUESTS = crawler_config.get('concurrent_requests', 16)
CONCURRENT_REQUESTS_PER_DOMAIN = 8

# User-Agent
USER_AGENT = crawler_config.get('user_agent', 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

# 中间件
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
}

# 管道
ITEM_PIPELINES = {
    'crawler.pipelines.DuplicatesPipeline': 300,
    'crawler.pipelines.FileDownloadPipeline': 400,
    'crawler.pipelines.StoragePipeline': 500,
}

# 日志级别
LOG_LEVEL = 'INFO'

# 文件下载设置
FILES_STORE = os.path.join(Path(__file__).parent.parent, 'data', 'files')
os.makedirs(FILES_STORE, exist_ok=True)


