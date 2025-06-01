from flask import request
from flask_restx import Resource, fields
import logging
import threading

from . import api

# 서비스, 예외, 유틸리티, 검증기 임포트 경로 수정
from ..services.repository_service import RepositoryService
from ..core.exceptions import ServiceError, ValidationError
from ..core.response_utils import (
    success_response,
    error_response,
    in_progress_response,
)
from ..core.validators import (
    validate_repo_url,
    validate_search_request,
)

# --- swagger_config.py 에서 가져온 모델 정의 시작 ---
# 공통 응답 모델들
base_response = api.model(
    "BaseResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(required=True, description="응답 메시지"),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-01-01T00:00:00Z",
        ),
    },
)

error_model = api.model(
    "ErrorResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="error"
        ),
        "message": fields.String(required=True, description="오류 메시지"),
        "error_code": fields.String(description="오류 코드"),
        "timestamp": fields.String(required=True, description="응답 시간 (ISO 8601)"),
    },
)

# 저장소 관련 모델들
repo_index_request = api.model(
    "RepositoryIndexRequest",
    {
        "repo_url": fields.String(
            required=True,
            description="GitHub 저장소 URL",
            example="https://github.com/pallets/flask",
        )
    },
)

repo_search_request = api.model(
    "RepositorySearchRequest",
    {
        "repo_name": fields.String(
            required=True,
            description="GitHub 저장소 이름 (owner/repository 형식)",
            example="pallets/flask",
        ),
        "query": fields.String(
            required=True,
            description="검색 질의문",
            example="Flask에서 request 객체를 어떻게 사용하나요?",
        ),
        "search_type": fields.String(
            required=False,
            description="검색 유형 (code 또는 document)",
            enum=["code", "document", "doc"],
            default="code",
            example="code",
        ),
    },
)

status_data = api.model(
    "StatusData",
    {
        "status": fields.String(description="인덱싱 상태", example="completed"),
        "repo_name": fields.String(description="저장소 이름", example="pallets/flask"),
        "repo_url": fields.String(
            description="저장소 URL", example="https://github.com/pallets/flask"
        ),
        "progress": fields.Float(description="진행률 (0-100)", example=100.0),
        "current_step": fields.String(
            description="현재 단계", example="Flask 저장소 인덱싱 완료"
        ),
        "total_files": fields.Integer(description="전체 파일 수", example=850),
        "processed_files": fields.Integer(description="처리된 파일 수", example=850),
        "start_time": fields.String(
            description="시작 시간", example="2024-07-22T11:00:00Z"
        ),
        "completion_time": fields.String(
            description="완료 시간", example="2024-07-22T11:15:00Z"
        ),
        "error": fields.String(description="오류 메시지", example=None),
    },
)

search_result_item = api.model(
    "SearchResultItem",
    {
        "content": fields.String(
            description="검색된 내용",
            example="from flask import request\\\\n\\\\n@app.route(\'/login\', methods=[\'POST\'])\\\\ndef login():\\\\n    username = request.form[\'username\']\\\\n    # ...",
        ),
        "file_path": fields.String(
            description="파일 경로", example="examples/flaskr/flaskr/auth.py"
        ),
        "score": fields.Float(description="유사도 점수", example=0.91),
        "metadata": fields.Raw(
            description="추가 메타데이터",
            example={"line_number": 42, "type": "function_definition"},
        ),
    },
)

search_results_model = api.model(
    "SearchResults",
    {
        "query": fields.String(
            description="검색 질의문", example="Flask에서 request 객체 사용법"
        ),
        "search_type": fields.String(description="검색 유형", example="code"),
        "repo_name": fields.String(description="저장소 이름", example="pallets/flask"),
        "results": fields.List(
            fields.Nested(search_result_item),
            description="검색 결과 목록"
        ),
        "total_results": fields.Integer(description="총 결과 수", example=2),
        "search_time": fields.Float(description="검색 소요 시간 (초)", example=0.250),
        "result_generated_at": fields.String(
            description="검색 결과 생성 시간 (ISO 8601)",
            example="2024-07-22T11:18:00Z",
        ),
    },
)

success_response_with_data = api.model(
    "SuccessResponseWithData",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="pallets/flask 저장소 검색 결과입니다.",
        ),
        "data": fields.Nested(
            search_results_model, description="응답 데이터"
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:20:00Z",
        ),
    },
)

progress_response = api.model(
    "ProgressResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="in_progress"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="pallets/flask 저장소 인덱싱이 진행 중입니다.",
        ),
        "progress": fields.Nested(
            status_data,
            description="진행 상황 데이터"
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:05:00Z",
        ),
    },
)
# --- swagger_config.py 에서 가져온 모델 정의 끝 ---


# Namespace 생성 (Blueprint 아래에 위치)
# api 객체는 app.api.__init__ 에서 가져온 것을 사용
repository_ns = api.namespace("repository", description="저장소 인덱싱 및 검색 API")


# 로거 설정
logger = logging.getLogger(__name__)

# Repository 서비스 인스턴스 생성
try:
    repo_service = RepositoryService()
except Exception as e:
    logger.error(f"RepositoryService 인스턴스 생성 실패: {e}", exc_info=True)
    repo_service = None


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
        if not repo_service:
            return error_response(message="RepositoryService가 로드되지 않았습니다.", error_code="SERVICE_UNAVAILABLE", status_code=503)
        try:
            data = request.get_json(force=True)
            repo_url = validate_repo_url(data)
            initial_status_result = repo_service.prepare_indexing(repo_url)
            current_status = initial_status_result.get("status")
            repo_name_from_result = initial_status_result.get("repo_name", "알 수 없는 저장소")
            is_new_request = initial_status_result.get("is_new_request", False)

            if current_status == "indexing" or current_status == "pending":
                if is_new_request:
                    thread = threading.Thread(target=repo_service.perform_indexing, args=(repo_url,))
                    thread.daemon = True
                    thread.start()
                    message = f"저장소 '{repo_name_from_result}' 인덱싱 작업이 시작되었습니다. 상태 API로 확인하세요."
                else:
                    message = f"저장소 '{repo_name_from_result}' 인덱싱 작업이 이미 진행 중입니다. 상태 API로 확인하세요."
                return in_progress_response(progress_data=initial_status_result, message=message, status_code=202)
            elif current_status == "completed":
                return success_response(data=initial_status_result, message=f"저장소 '{repo_name_from_result}'은(는) 이미 성공적으로 인덱싱되었습니다.", status_code=200)
        except ValidationError as e:
            logger.warning(f"입력 값 검증 오류: {e}")
            return error_response(message=str(e), error_code="VALIDATION_ERROR", status_code=400)
        except ServiceError as e:
            logger.error(f"서비스 오류: {e}")
            return error_response(message=str(e), error_code="SERVICE_ERROR", status_code=400)
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}", exc_info=True)
            return error_response(message="서버 내부 오류가 발생했습니다.", error_code="INTERNAL_SERVER_ERROR", status_code=500)


@repository_ns.route("/search")
class RepositorySearch(Resource):
    @repository_ns.doc("search_repository")
    @repository_ns.expect(repo_search_request, validate=True)
    @repository_ns.response(200, "검색 완료", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소에서 코드 또는 문서 검색"""
        if not repo_service:
            return error_response(message="RepositoryService가 로드되지 않았습니다.", error_code="SERVICE_UNAVAILABLE", status_code=503)
        try:
            data = request.get_json()
            repo_name, query, search_type = validate_search_request(data)
            repo_url = f"https://github.com/{repo_name}"
            result = repo_service.search_repository(repo_url, query, search_type)
            return success_response(data=result, message="검색이 완료되었습니다.")
        except ValidationError as e:
            logger.warning(f"입력 값 검증 오류: {e}")
            return error_response(message=str(e), error_code="VALIDATION_ERROR", status_code=400)
        except ServiceError as e:
            logger.error(f"서비스 오류: {e}")
            return error_response(message=str(e), error_code="SERVICE_ERROR", status_code=400)
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}", exc_info=True)
            return error_response(message="서버 내부 오류가 발생했습니다.", error_code="INTERNAL_SERVER_ERROR", status_code=500)


@repository_ns.route("/status/<path:repo_name>")
class RepositoryStatus(Resource):
    @repository_ns.doc("get_repository_status")
    @repository_ns.param("repo_name", "저장소 이름 (owner/repository 형식)", required=True)
    @repository_ns.response(200, "상태 조회 성공", success_response_with_data)
    @repository_ns.response(202, "인덱싱 진행 중", progress_response)
    @repository_ns.response(404, "저장소를 찾을 수 없음", error_model)
    @repository_ns.response(409, "인덱싱 실패", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def get(self, repo_name):
        """저장소 인덱싱 상태 확인"""
        if not repo_service:
            return error_response(message="RepositoryService가 로드되지 않았습니다.", error_code="SERVICE_UNAVAILABLE", status_code=503)
        try:
            status_data_result = repo_service.get_repository_status(repo_name)
            if status_data_result.get("status") == "not_indexed":
                return error_response(message=f"저장소 '{repo_name}'에 대한 인덱싱 정보를 찾을 수 없습니다.", error_code="NOT_FOUND", status_code=404)
            if status_data_result.get("status") == "indexing" or status_data_result.get("status") == "pending":
                return in_progress_response(progress_data=status_data_result, message=f"저장소 '{repo_name}' 인덱싱 진행 중입니다.", status_code=202)
            if status_data_result.get("status") == "failed":
                return error_response(message=f"저장소 '{repo_name}' 인덱싱에 실패했습니다: {status_data_result.get('error', '알 수 없는 오류')}", error_code="INDEXING_FAILED", status_code=409)
            return success_response(data=status_data_result)
        except ServiceError as e:
            logger.error(f"상태 조회 서비스 오류: {e}")
            return error_response(message=str(e), error_code="SERVICE_ERROR", status_code=400)
        except Exception as e:
            logger.error(f"상태 조회 중 예상치 못한 오류: {e}", exc_info=True)
            return error_response(message="상태 조회 중 서버 내부 오류가 발생했습니다.", error_code="INTERNAL_SERVER_ERROR", status_code=500) 