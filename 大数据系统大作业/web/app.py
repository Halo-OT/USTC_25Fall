"""
Flask Web应用
"""
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.hbase_client import HBaseClient
from search.searcher import Searcher

app = Flask(__name__)

# 加载配置
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

web_config = config.get('web', {})
app.config['DEBUG'] = web_config.get('debug', True)

# 初始化搜索器
searcher = Searcher()


@app.route('/')
def index():
    """
    首页
    """
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    搜索接口
    """
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '').strip()
        source = data.get('source', '')
        max_results = data.get('max_results', 50)
    else:
        query = request.args.get('q', '').strip()
        source = request.args.get('source', '')
        max_results = int(request.args.get('limit', 50))
    
    if not query:
        return jsonify({
            'success': False,
            'message': '查询字符串不能为空',
            'results': []
        })
    
    try:
        # 执行搜索
        if source:
            results = searcher.search_by_source(query, source, max_results)
        else:
            results = searcher.search(query, max_results)
        
        # 格式化结果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'url': doc.url,
                'title': doc.title,
                'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                'source': doc.source,
                'file_type': doc.file_type,
                'file_size': doc.file_size,
                'score': round(score, 4),
                'file_path': doc.file_path
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(formatted_results),
            'results': formatted_results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'搜索出错: {str(e)}',
            'results': []
        })


@app.route('/sources')
def get_sources():
    """
    获取所有来源网站列表
    """
    try:
        hbase_client = HBaseClient()
        documents = hbase_client.get_all_documents()
        sources = list(set(doc.source for doc in documents if doc.source))
        sources.sort()
        return jsonify({
            'success': True,
            'sources': sources
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'sources': []
        })


if __name__ == '__main__':
    host = web_config.get('host', '0.0.0.0')
    port = web_config.get('port', 5000)
    app.run(host=host, port=port, debug=app.config['DEBUG'])


