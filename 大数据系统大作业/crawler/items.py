"""
Scrapy Items定义
"""
import scrapy


class DocumentItem(scrapy.Item):
    """
    文档Item
    """
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    file_type = scrapy.Field()
    file_size = scrapy.Field()
    source = scrapy.Field()
    file_path = scrapy.Field()  # 下载文件的本地路径
    is_file = scrapy.Field()  # 是否是文件（PDF、DOC等）


