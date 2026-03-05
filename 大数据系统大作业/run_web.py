#!/usr/bin/env python
"""
启动Web服务脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from web.app import app

if __name__ == '__main__':
    print("=" * 50)
    print("启动Web服务")
    print("访问 http://localhost:5000")
    print("=" * 50)
    app.run(debug=True)


