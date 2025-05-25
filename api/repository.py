from flask import Blueprint, request
import logging
from datetime import datetime

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
    """저장소 인덱싱 API"""
    try:
        if not request.is_json:
            return error_response(
                message="Content-Type이 application/json이어야 합니다.",
                error_code="INVALID_CONTENT_TYPE",
                status_code=400,
            )
        data = request.get_json(
            force=True
        )  # force=True는 Content-Type이 application/json이 아니어도 파싱 시도. 위에서 이미 검증했으므로 get_json()으로 변경 가능

        repo_url = validate_repo_url(data)  # 검증 함수 사용

        # 저장소 인덱싱 실행
        result = repo_service.index_repository(repo_url)

        # 서비스 결과 상태에 따른 응답 처리
        current_status = result.get("status")
        repo_name_from_result = result.get("repo_name", "알 수 없는 저장소")

        if current_status == "indexing" or current_status == "pending":
            # 이미 처리 중이거나 대기 중인 경우
            return success_response(
                data=result,
                message=f"저장소 '{repo_name_from_result}' 인덱싱이 이미 진행 중이거나 대기 중입니다. 상태 API로 확인하세요.",
                status_code=202,
            )
        elif current_status == "completed":
            # 인덱싱 완료 (신규 또는 기존 완료 건)
            message = result.get("progress_message", "저장소 인덱싱이 완료되었습니다.")

            # 서비스에서 제공하는 메시지를 우선 사용.
            # 아래는 API 레벨에서 최초 완료/기존 완료를 구분하려는 예시 (서비스에서 명확한 상태를 주는 것이 더 좋음)
            try:
                start_dt = datetime.fromisoformat(result.get("start_time", ""))
                end_dt = datetime.fromisoformat(result.get("end_time", ""))
                # 단순 시간차로 신규/기존 완료 구분 (정확도 낮음, 서비스 로직 개선 권장)
                if (end_dt - start_dt).total_seconds() < 5:
                    message = (
                        f"저장소 '{repo_name_from_result}' 인덱싱이 완료되었습니다."
                    )
                else:
                    message = f"저장소 '{repo_name_from_result}'은(는) 이전에 성공적으로 인덱싱되었습니다."
            except (TypeError, ValueError):
                pass  # 타임스탬프 파싱 오류 시 기본 메시지 사용

            return success_response(
                data=result,
                message=message,
                status_code=200,
            )

        # index_repository가 completed, indexing, pending 외 다른 상태를 직접 반환하지 않으므로,
        # 이 지점은 현재 로직상 도달하기 어려움.
        # 필요시 기본 성공 응답 또는 오류 처리 추가.

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
        # repo_name이 URL 인코딩되어 올 수 있으므로 디코딩 (Flask가 자동으로 처리해줄 수 있음)
        # 여기서는 repo_name을 그대로 사용한다고 가정
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
                status_code=200,  # 상태는 가져왔으므로 200, 내용은 실패
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
