import logging
from flask import Flask
from .core.config import Config
from .api import api_bp # Blueprint 임포트 (추후 생성)
# from .api import register_blueprints # 블루프린트 등록 함수 (추후 생성)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
        format=Config.LOG_FORMAT
    )
    app.logger.info("Flask app created and configured.")

    # 블루프린트 등록
    app.register_blueprint(api_bp, url_prefix='/') 

    @app.route("/")
    def home():
        app.logger.info("Root path requested.")
        return {
            "service": "AIssue Repository RAG 분석 서비스",
            "version": Config.API_VERSION,
            "status": "running",
            "swagger_docs": "/docs/",  # Swagger UI 링크는 Blueprint 경로에 따라 변경
        }

    # 추가적인 앱 설정 (예: 데이터베이스 초기화, CORS 설정 등)
    # db.init_app(app)
    # from flask_cors import CORS
    # CORS(app) # 예시

    return app 