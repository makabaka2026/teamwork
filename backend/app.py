"""Flask 应用工厂"""
from flask import Flask
from flask_cors import CORS

from backend.config import Config


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    app.config.from_object(Config)
    CORS(app)

    # 注册路由
    from backend.routes.poetry_routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # 主页
    from flask import render_template

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
