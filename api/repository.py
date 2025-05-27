from flask import request
from flask_restx import Resource, Namespace
import logging
import threading

from repo_rag_analyzer.service import RepositoryService
from common.exceptions import ServiceError, ValidationError
from common.response_utils import (
    success_response,
    error_response,
    in_progress_response,
)
from common.validators import (
    validate_repo_url,
    validate_search_request,
)
from .swagger_config import (
    repo_index_request,
    repo_search_request,
    success_response_with_data,
    error_response as error_model,
    progress_response,
)

# Namespace 생성 (Blueprint 대신)
repository_ns = Namespace("repository", description="저장소 인덱싱 및 검색 API")

# 로거 설정
logger = logging.getLogger(__name__)

# Repository 서비스 인스턴스 생성
repo_service = RepositoryService()


@repository_ns.route("/index")
class RepositoryIndex(Resource):
    @repository_ns.doc("index_repository")
    @repository_ns.expect(repo_index_request, validate=True)
    @repository_ns.response(202, "인덱싱 작업 시작됨", progress_response)
    @repository_ns.response(200, "이미 인덱싱 완료됨", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소 인덱싱 요청. 요청 즉시 응답 후 백그라운드에서 인덱싱 진행."""
        try:
            data = request.get_json(force=True)
            repo_url = validate_repo_url(data)

            # 인덱싱 준비 및 초기 상태 확인
            initial_status_result = repo_service.prepare_indexing(repo_url)

            current_status = initial_status_result.get("status")
            repo_name_from_result = initial_status_result.get(
                "repo_name", "알 수 없는 저장소"
            )
            is_new_request = initial_status_result.get("is_new_request", False)

            if current_status == "indexing" or current_status == "pending":
                if is_new_request:
                    thread = threading.Thread(
                        target=repo_service.perform_indexing, args=(repo_url,)
                    )
                    thread.daemon = True
                    thread.start()
                    message = f"저장소 '{repo_name_from_result}' 인덱싱 작업이 시작되었습니다. 상태 API로 확인하세요."
                else:
                    message = f"저장소 '{repo_name_from_result}' 인덱싱 작업이 이미 진행 중입니다. 상태 API로 확인하세요."

                # 진행 중 응답으로 변경
                return in_progress_response(
                    progress_data=initial_status_result,
                    message=message,
                    status_code=202,
                )
            elif current_status == "completed":
                return success_response(
                    data=initial_status_result,
                    message=f"저장소 '{repo_name_from_result}'은(는) 이미 성공적으로 인덱싱되었습니다.",
                    status_code=200,
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


@repository_ns.route("/search")
class RepositorySearch(Resource):
    @repository_ns.doc("search_repository")
    @repository_ns.expect(repo_search_request, validate=True)
    @repository_ns.response(200, "검색 완료", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소에서 코드 또는 문서 검색"""
        try:
            data = request.get_json()

            repo_name, query, search_type = validate_search_request(data)

            # repo_name을 repo_url로 변환
            # 여기서는 GitHub URL을 가정합니다. 다른 Git 호스팅 서비스를 지원하려면 수정 필요.
            repo_url = f"https://github.com/{repo_name}"

            result = repo_service.search_repository(
                repo_url, query, search_type
            )  # repo_url 전달

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


@repository_ns.route("/status/<path:repo_name>")
class RepositoryStatus(Resource):
    @repository_ns.doc("get_repository_status")
    @repository_ns.param(
        "repo_name", "저장소 이름 (owner/repository 형식)", required=True
    )
    @repository_ns.response(200, "상태 조회 성공", success_response_with_data)
    @repository_ns.response(202, "인덱싱 진행 중", progress_response)
    @repository_ns.response(404, "저장소를 찾을 수 없음", error_model)
    @repository_ns.response(409, "인덱싱 실패", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def get(self, repo_name):
        """저장소 인덱싱 상태 확인"""
        try:
            status_data = repo_service.get_repository_status(repo_name)

            if status_data.get("status") == "not_indexed":
                return error_response(
                    message=f"저장소 '{repo_name}'에 대한 인덱싱 정보를 찾을 수 없습니다.",
                    error_code="NOT_FOUND",
                    status_code=404,
                )
            if (
                status_data.get("status") == "indexing"
                or status_data.get("status") == "pending"
            ):
                return in_progress_response(
                    progress_data=status_data,
                    message=f"저장소 '{repo_name}' 인덱싱 진행 중입니다.",
                    status_code=202,
                )
            if status_data.get("status") == "failed":
                return error_response(
                    message=f"저장소 '{repo_name}' 인덱싱에 실패했습니다: {status_data.get('error', '알 수 없는 오류')}",
                    error_code="INDEXING_FAILED",
                    status_code=409,
                )

            return success_response(data=status_data)
        except ServiceError as e:
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
