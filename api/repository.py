from flask import Blueprint, request, jsonify
import logging

from repo_rag_analyzer.service import RepositoryService
from common.exceptions import ServiceError

# Blueprint 생성
repository_bp = Blueprint("repository", __name__)

# 로거 설정
logger = logging.getLogger(__name__)

# Repository 서비스 인스턴스 생성
repo_service = RepositoryService()


@repository_bp.route("/repository/index", methods=["POST"])
def index_repository():
    """저장소 인덱싱 API"""
    try:
        # Content-Type 검증
        if not request.is_json:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Content-Type이 application/json이어야 합니다.",
                    }
                ),
                400,
            )

        # JSON 데이터 파싱
        try:
            data = request.get_json(force=True)
        except Exception as json_error:
            return (
                jsonify(
                    {"status": "error", "message": f"JSON 파싱 오류: {str(json_error)}"}
                ),
                400,
            )

        if not data or "repo_url" not in data:
            return (
                jsonify({"status": "error", "message": "repo_url이 필요합니다."}),
                400,
            )

        repo_url = str(data["repo_url"]).strip()
        if not repo_url:
            return (
                jsonify(
                    {"status": "error", "message": "유효한 repo_url을 입력해주세요."}
                ),
                400,
            )

        # 저장소 인덱싱 실행
        result = repo_service.index_repository(repo_url)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "저장소 인덱싱이 완료되었습니다.",
                    "data": result,
                }
            ),
            200,
        )

    except ServiceError as e:
        logger.error(f"서비스 오류: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        return (
            jsonify({"status": "error", "message": "서버 내부 오류가 발생했습니다."}),
            500,
        )


@repository_bp.route("/repository/search", methods=["POST"])
def search_repository():
    """저장소 검색 API"""
    try:
        # Content-Type 검증
        if not request.is_json:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Content-Type이 application/json이어야 합니다.",
                    }
                ),
                400,
            )

        # JSON 데이터 파싱
        try:
            data = request.get_json(force=True)
        except Exception as json_error:
            return (
                jsonify(
                    {"status": "error", "message": f"JSON 파싱 오류: {str(json_error)}"}
                ),
                400,
            )

        if not data or "repo_url" not in data or "query" not in data:
            return (
                jsonify(
                    {"status": "error", "message": "repo_url과 query가 필요합니다."}
                ),
                400,
            )

        repo_url = str(data["repo_url"]).strip()
        query = str(data["query"]).strip()
        search_type = str(data.get("search_type", "code")).strip()  # 기본값: code

        if not repo_url or not query:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "유효한 repo_url과 query를 입력해주세요.",
                    }
                ),
                400,
            )

        # 검색 실행
        result = repo_service.search_repository(repo_url, query, search_type)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "검색이 완료되었습니다.",
                    "data": result,
                }
            ),
            200,
        )

    except ServiceError as e:
        logger.error(f"서비스 오류: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        return (
            jsonify({"status": "error", "message": "서버 내부 오류가 발생했습니다."}),
            500,
        )


@repository_bp.route("/repository/status/<repo_name>", methods=["GET"])
def get_repository_status(repo_name):
    """저장소 인덱싱 상태 확인 API"""
    try:
        status = repo_service.get_repository_status(repo_name)
        return jsonify({"status": "success", "data": status}), 200
    except Exception as e:
        logger.error(f"상태 조회 오류: {e}")
        return (
            jsonify(
                {"status": "error", "message": "상태 조회 중 오류가 발생했습니다."}
            ),
            500,
        )
