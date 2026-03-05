# 中科大校内文件搜索引擎

## 项目简介

本项目是一个基于HBase的分布式文件搜索引擎，用于搜索中科大各学院和管理部门网站上的文件和文档。

## 技术栈

- **爬虫框架**：Scrapy
- **分布式数据库**：HBase
- **搜索引擎**：自建倒排索引 + TF-IDF/BM25算法
- **Web框架**：Flask
- **中文分词**：jieba

## 项目结构

```
DataSystem/
├── crawler/              # 爬虫模块
│   ├── spiders/         # Scrapy爬虫
│   ├── pipelines.py     # 数据处理管道
│   ├── items.py         # 数据项定义
│   └── settings.py      # 爬虫配置
├── storage/             # 存储模块
│   ├── hbase_client.py  # HBase客户端
│   └── data_model.py    # 数据模型
├── search/              # 搜索引擎模块
│   ├── indexer.py       # 索引构建
│   ├── searcher.py      # 搜索实现
│   ├── tokenizer.py     # 分词器
│   └── ranking.py       # 排序算法
├── web/                 # Web界面
│   ├── app.py           # Flask应用
│   ├── templates/       # HTML模板
│   └── static/          # 静态文件
├── utils/               # 工具函数
│   ├── text_processor.py
│   └── file_handler.py
├── config/              # 配置文件
│   └── config.yaml
├── requirements.txt     # Python依赖
├── README.md
└── 作业要求.md
```

## 环境搭建

### 1. 创建conda环境

```bash
conda create -n homework python=3.10
conda activate homework
```

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 安装Hadoop和HBase

请参考课程实验平台的安装指南，或使用Docker部署Hadoop和HBase。

### 4. 配置HBase

确保HBase服务已启动，并配置连接信息。

## 快速开始

### 1. 运行爬虫

爬取中科大各网站的文件和文档：

```bash
python run_crawler.py
```

或者：

```bash
cd crawler
scrapy crawl ustc_spider
```

### 2. 构建索引

在爬取完数据后，构建倒排索引：

```bash
python build_index.py
```

### 3. 启动Web服务

启动搜索Web界面：

```bash
python run_web.py
```

然后在浏览器中访问：http://localhost:5000

## 功能特性

- ✅ **智能爬虫**：自动识别"下载中心"等文件专栏，支持多种文件格式
- ✅ **分布式存储**：使用HBase存储数据（支持本地文件fallback）
- ✅ **中文搜索**：基于jieba分词和倒排索引
- ✅ **相关性排序**：使用BM25算法计算相关性分数
- ✅ **Web界面**：友好的搜索界面，支持按来源筛选

## 详细文档

更多使用说明请参考：[使用说明.md](使用说明.md)

## 项目状态

- [x] 爬虫开发
- [x] HBase数据存储
- [x] 搜索引擎实现
- [x] Web界面开发
- [x] 基础功能测试

## 注意事项

1. 爬取时请遵守robots.txt协议
2. 设置合理的爬取延迟，避免对服务器造成压力
3. 定期备份HBase数据

