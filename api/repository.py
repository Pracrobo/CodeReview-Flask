from flask import Blueprint, request
import logging
import threading  # 스레딩 임포트

from repo_rag_analyzer.service import RepositoryService
from common.exceptions import ServiceError, ValidationError
from common.response_utils import success_response, error_response  # 새로운 import
from common.validators import (
    validate_repo_url,
    validate_search_request,
)

# Blueprint 생성
repository_bp = Blueprint("repository", __name__)

# 로거 설정
logger = logging.getLogger(__name__)

# Repository 서비스 인스턴스 생성
repo_service = RepositoryService()


@repository_bp.route("/repository/index", methods=["POST"])
def index_repository():
    """저장소 인덱싱 API. 요청 즉시 응답 후 백그라운드에서 인덱싱 진행."""
    try:
        if not request.is_json:
            return error_response(
                message="Content-Type이 application/json이어야 합니다.",
                error_code="INVALID_CONTENT_TYPE",
                status_code=400,
            )
        data = request.get_json(force=True)
        repo_url = validate_repo_url(data)

        # 인덱싱 준비 및 초기 상태 확인
        initial_status_result = repo_service.prepare_indexing(repo_url)

        current_status = initial_status_result.get("status")
        repo_name_from_result = initial_status_result.get(
            "repo_name", "알 수 없는 저장소"
        )

        if current_status == "indexing" or current_status == "pending":
            # 새 요청인 경우 백그라운드 스레드에서 실제 인덱싱 작업 실행
            if initial_status_result.get("is_new_request", False):
                thread = threading.Thread(
                    target=repo_service.perform_indexing, args=(repo_url,)
                )
                thread.daemon = True
                thread.start()

            return success_response(
                data=initial_status_result,
                message=f"저장소 '{repo_name_from_result}' 인덱싱 작업이 시작되었거나 이미 진행 중입니다. 상태 API로 확인하세요.",
                status_code=202,  # Accepted
            )
        elif current_status == "completed":
            return success_response(
                data=initial_status_result,
                message=f"저장소 '{repo_name_from_result}'은(는) 이미 성공적으로 인덱싱되었습니다.",
                status_code=200,
            )
        # prepare_indexing이 실패 상태를 직접 반환하지 않으므로, 추가적인 else/elif는 불필요.
        # 실패는 perform_indexing 내부에서 상태 업데이트로 처리됨.

    except ValidationError as e:
        logger.warning(f"입력 값 검증 오류: {e}")
        return error_response(
            message=str(e), error_code="VALIDATION_ERROR", status_code=400
        )
    except ServiceError as e:
        logger.error(f"서비스 오류: {e}")
        # ServiceError에서 error_code를 포함하도록 수정 필요 (선택 사항)
        return error_response(
            message=str(e), error_code="SERVICE_ERROR", status_code=400
        )
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        return error_response(
            message="서버 내부 오류가 발생했습니다.",
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500,
        )


@repository_bp.route("/repository/search", methods=["POST"])
def search_repository():
    """저장소 검색 API"""
    try:
        if not request.is_json:
            return error_response(
                message="Content-Type이 application/json이어야 합니다.",
                error_code="INVALID_CONTENT_TYPE",
                status_code=400,
            )
        data = request.get_json()

        repo_url, query, search_type = validate_search_request(data)  # 검증 함수 사용

        # 검색 실행
        result = repo_service.search_repository(repo_url, query, search_type)

        return success_response(
            data=result,
            message="검색이 완료되었습니다.",
        )

    except ValidationError as e:
        logger.warning(f"입력 값 검증 오류: {e}")
        return error_response(
            message=str(e), error_code="VALIDATION_ERROR", status_code=400
        )
    except ServiceError as e:
        logger.error(f"서비스 오류: {e}")
        return error_response(
            message=str(e), error_code="SERVICE_ERROR", status_code=400
        )
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        return error_response(
            message="서버 내부 오류가 발생했습니다.",
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500,
        )


@repository_bp.route(
    "/repository/status/<path:repo_name>", methods=["GET"]
)  # repo_name에 URL 경로가 포함될 수 있으므로 path 사용
def get_repository_status(repo_name):
    """저장소 인덱싱 상태 확인 API"""
    try:
        status_data = repo_service.get_repository_status(repo_name)

        if status_data.get("status") == "not_indexed":
            return error_response(
                message=f"저장소 '{repo_name}'에 대한 인덱싱 정보를 찾을 수 없습니다.",
                error_code="NOT_FOUND",
                status_code=404,  # Not Found
            )
        if (
            status_data.get("status") == "indexing"
            or status_data.get("status") == "pending"
        ):  # 'pending' 상태 추가 고려
            return success_response(  # 202 대신 200으로 성공 상태를 알리고, data에 진행상황 포함
                data=status_data,
                message=f"저장소 '{repo_name}' 인덱싱 진행 중입니다.",
                status_code=200,  # OK, 하지만 내용은 진행 중
            )
        if status_data.get("status") == "failed":
            return error_response(
                message=f"저장소 '{repo_name}' 인덱싱에 실패했습니다: {status_data.get('error', '알 수 없는 오류')}",
                error_code="INDEXING_FAILED",
                status_code=409,  # HTTP 500 대신 409 (Conflict) 사용
            )

        return success_response(data=status_data)
    except ServiceError as e:  # 서비스 로직 내에서 발생할 수 있는 예상된 오류
        logger.error(f"상태 조회 서비스 오류: {e}")
        return error_response(
            message=str(e), error_code="SERVICE_ERROR", status_code=400
        )
    except Exception as e:
        logger.error(f"상태 조회 중 예상치 못한 오류: {e}", exc_info=True)
        return error_response(
            message="상태 조회 중 서버 내부 오류가 발생했습니다.",
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500,
        )
