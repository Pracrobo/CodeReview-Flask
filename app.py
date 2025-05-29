from flask import Flask
import logging

from api.swagger_config import api
from api.repository import repository_ns
from config import Config


def create_app():
    """Flask 앱 팩토리"""
    app = Flask(__name__)

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT
    )

    # Flask-RESTX API 초기화
    api.init_app(app)

    # Namespace 등록
    api.add_namespace(repository_ns, path="/api/repository")

    @app.route("/")
    def home():
        return {
            "service": "AIssue Repository RAG 분석 서비스",
            "version": Config.API_VERSION,
            "status": "running",
            "swagger_docs": "/docs/",  # Swagger UI 링크 추가
        }

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=3002)
