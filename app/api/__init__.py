# This file makes the 'api' directory a Python package.

from flask import Blueprint
from flask_restx import Api


# Blueprint 생성
api_bp = Blueprint("api", __name__)

# Flask-RESTX Api 객체 생성
api = Api(
    api_bp,
    version="1.0",
    title="AIssue Repository RAG 분석 API",
    description="GitHub 저장소의 코드와 문서를 분석하여 RAG 기반 검색을 제공하는 API",
    doc="/docs/",
)

# 네임스페이스 등록
from .repository_api import repository_ns
from .analyze_issue_api import issue_ns
from .chatbot_api import chatbot_ns  # 새로운 네임스페이스 임포트

api.add_namespace(repository_ns, path="/repository")
api.add_namespace(issue_ns, path="/issue")
api.add_namespace(chatbot_ns, path="/chatbot")  # 새로운 네임스페이스 등록
