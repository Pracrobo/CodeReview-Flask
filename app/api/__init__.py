# This file makes the 'api' directory a Python package.

from flask import Blueprint
from flask_restx import Api
from .repository_api import repository_ns  # repository_api.py 에서 정의된 네임스페이스

# Blueprint 생성
api_bp = Blueprint("api", __name__)

# Flask-RESTX Api 객체 생성 및 Blueprint에 연결
# doc='/docs/'를 설정하여 Swagger UI 경로를 /docs/ 로 만듭니다.
api = Api(
    api_bp,
    version="1.0",
    title="AIssue Repository RAG 분석 API",
    description="GitHub 저장소의 코드와 문서를 분석하여 RAG 기반 검색을 제공하는 API",
    doc="/docs/",
)

# 네임스페이스 임포트 및 등록
api.add_namespace(repository_ns, path="/repository")
