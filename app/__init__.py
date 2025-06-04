import logging
from flask import Flask
from .core.config import Config
from .api import api_bp  # Blueprint 임포트


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
        format=Config.LOG_FORMAT,
    )
    app.logger.info("Flask app created and configured.")

    # 블루프린트 등록
    app.register_blueprint(api_bp, url_prefix="/")

    @app.route("/")
    def home():
        app.logger.info("Root path requested.")
        return {
            "service": "AIssue Repository RAG 분석 서비스",
            "version": Config.API_VERSION,
            "status": "running",
            "swagger_docs": "/docs/",
        }

    return app
