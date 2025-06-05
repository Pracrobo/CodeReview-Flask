import logging
from flask import Flask
from .core.config import Config
from .api import api_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 로깅 설정
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    stream_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))

    app.logger.handlers.clear()
    app.logger.addHandler(stream_handler)
    app.logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    app.logger.info("Flask 애플리케이션이 생성되고 구성되었습니다.")

    # API Blueprint 등록
    app.register_blueprint(api_bp, url_prefix="/")

    @app.route("/")
    def home():
        app.logger.info("루트 경로 요청됨")
        return {
            "service": "AIssue Repository RAG 분석 서비스",
            "version": Config.API_VERSION,
            "status": "running",
            "swagger_docs": "/docs/",
        }

    return app
