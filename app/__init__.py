import logging
from flask import Flask
from .core.config import Config
from .api import api_bp # Blueprint 임포트 (추후 생성)
# from .api import register_blueprints # 블루프린트 등록 함수 (추후 생성)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 로깅 설정
    # logging.basicConfig 제거

    # Flask 앱 로거 가져오기
    # gunicorn_logger = logging.getLogger('gunicorn.error') # gunicorn 로거 사용하지 않음
    # app.logger.handlers = gunicorn_logger.handlers
    # app.logger.setLevel(gunicorn_logger.level)

    # 콘솔 출력을 위한 StreamHandler 설정
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    stream_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))

    # 기존 핸들러 제거 후 새 핸들러 추가
    app.logger.handlers.clear()
    app.logger.addHandler(stream_handler)

    # 앱 로거 레벨 설정
    app.logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    app.logger.info("Flask app created and configured.")
    app.logger.info("플라스크 로거가 설정되었습니다.") # 한국어 로그 메시지

    # 블루프린트 등록
    app.register_blueprint(api_bp, url_prefix='/') 

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