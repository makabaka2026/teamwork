"""诗词自动生成系统 - 一键启动"""
import sys
from pathlib import Path

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 50)
    print("  诗词自动生成系统  Poetry Generator")
    print("=" * 50)
    print(f"  访问地址: http://localhost:5000")
    print(f"  API 文档: http://localhost:5000/api/health")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
