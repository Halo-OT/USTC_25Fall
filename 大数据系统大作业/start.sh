#!/bin/bash
# 项目启动脚本

echo "=========================================="
echo "中科大校内文件搜索引擎"
echo "=========================================="
echo ""

# 检查conda环境
if ! conda env list | grep -q "homework"; then
    echo "错误: 未找到conda环境 'homework'"
    echo "请先运行: conda create -n homework python=3.10"
    exit 1
fi

# 激活环境
echo "激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate homework

echo ""
echo "请选择操作："
echo "1. 运行爬虫"
echo "2. 构建索引"
echo "3. 启动Web服务"
echo "4. 测试搜索"
echo "5. 全部执行（爬虫 -> 索引 -> Web服务）"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "启动爬虫..."
        python run_crawler.py
        ;;
    2)
        echo "构建索引..."
        python build_index.py
        ;;
    3)
        echo "启动Web服务..."
        echo "访问 http://localhost:5000"
        python run_web.py
        ;;
    4)
        echo "测试搜索..."
        python test_search.py
        ;;
    5)
        echo "执行完整流程..."
        echo "1. 运行爬虫..."
        python run_crawler.py
        echo ""
        echo "2. 构建索引..."
        python build_index.py
        echo ""
        echo "3. 启动Web服务..."
        echo "访问 http://localhost:5000"
        python run_web.py
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac


