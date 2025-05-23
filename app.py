from flask import Flask
import logging

from api.repository import repository_bp
from config import Config


def create_app():
    """Flask 애플리케이션 팩토리 함수"""
    app = Flask(__name__)

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT
    )

    # Blueprint 등록
    app.register_blueprint(repository_bp, url_prefix="/api")

    @app.route("/")
    def home():
        return {
            "service": "AIssue Repository RAG 분석 서비스",
            "version": Config.API_VERSION,
            "status": "running",
        }

    @app.route("/health")
    def health_check():
        return {"status": "healthy"}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=3002)
