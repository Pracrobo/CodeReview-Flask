from flask import request
from flask_restx import Resource, fields
import logging

from . import api

from ..services.status_service import StatusService
from ..services.indexing_service import IndexingService
from ..services.search_service import SearchService
from ..services.readme_summarizer import ReadmeSummarizer

from ..core.exceptions import ServiceError, ValidationError
from ..core.response_utils import (
    success_response,
    error_response,
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
        ),
        "repository_info": fields.Raw(
            required=False,
            description="Express에서 전달한 저장소 정보",
            example={
                "githubRepoId": 123456,
                "fullName": "pallets/flask",
                "name": "flask",
                "description": "A lightweight WSGI web application framework",
                "programmingLanguage": "Python",
                "star": 65000,
                "fork": 16000,
            },
        ),
        "callback_url": fields.String(
            required=False,
            description="분석 완료 시 콜백할 Express API URL",
            example="http://localhost:3001/internal/analysis-complete",
        ),
        "user_id": fields.Integer(
            required=False, description="분석을 요청한 사용자 ID", example=1
        ),
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
            description="검색 유형 (code만 지원)",
            enum=["code"],
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
            example="from flask import request\\\\n\\\\n@app.route('/login', methods=['POST'])\\\\ndef login():\\\\n    username = request.form['username']\\\\n    # ...",
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
            fields.Nested(search_result_item), description="검색 결과 목록"
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
        "data": fields.Nested(search_results_model, description="응답 데이터"),
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
        "progress": fields.Nested(status_data, description="진행 상황 데이터"),
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

# 서비스 인스턴스 생성
# StatusService는 싱글톤이므로 직접 인스턴스화
status_service = StatusService()
indexing_service = IndexingService(status_service)
search_service = SearchService(status_service)
readme_summarizer = ReadmeSummarizer()


@repository_ns.route("/index")
class RepositoryIndex(Resource):
    @repository_ns.doc("index_repository")
    @repository_ns.expect(repo_index_request, validate=True)
    @repository_ns.response(202, "인덱싱 작업 시작됨/진행중", progress_response)
    @repository_ns.response(200, "이미 인덱싱 완료됨", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소 인덱싱 요청. Express에서 호출되는 API."""
        data = {}
        try:
            data = request.get_json(force=True)
            repo_url = validate_repo_url(data)
            repository_info = data.get("repository_info", {})
            callback_url = data.get("callback_url")
            user_id = data.get("user_id")

            log_payload = {
                "repo_url": repo_url,
                "repo_full_name": repository_info.get("fullName"),
                "callback_url": callback_url,
                "user_id": user_id,
            }
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 요청 데이터: {log_payload}"
            )

            if repository_info:
                logger.debug(f"전달된 저장소 상세 정보 ({repo_url}): {repository_info}")

            # 인덱싱 서비스에 콜백 URL과 사용자 ID 전달
            initial_status_result = indexing_service.prepare_and_start_indexing(
                repo_url, callback_url, user_id
            )
            repo_name = status_service._get_repo_name_from_url(repo_url)

            current_status = initial_status_result.get("status")
            is_new_request = initial_status_result.get("is_new_request", False)

            # Express가 기대하는 응답 형식으로 변환
            response_data = {
                "analysis_id": repo_name,  # Flask에서 사용하는 분석 ID
                "repo_name": repo_name,
                "status": current_status,
                "progress": initial_status_result.get("progress", 0),
                "message": initial_status_result.get("progress_message", ""),
                "started_at": initial_status_result.get("start_time"),
                "estimated_completion": initial_status_result.get("end_time"),
            }

            if current_status == "completed":
                message = (
                    f"저장소 '{repo_name}'은(는) 이미 성공적으로 인덱싱되었습니다."
                )
                logger.info(
                    f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 200. 메시지: {message}"
                )
                return success_response(
                    data=response_data, message=message, status_code=200
                )

            # 새로운 요청이거나 진행 중인 요청 처리
            if is_new_request and (current_status in ["pending", "indexing"]):
                message = f"저장소 '{repo_name}' 인덱싱 작업이 시작되었습니다."
            else:
                message = f"저장소 '{repo_name}' 인덱싱 작업이 이미 진행 중입니다."

            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 202. 메시지: {message}"
            )
            return success_response(
                data=response_data, message=message, status_code=202
            )

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            # logger.error(f"API 요청 오류: [{request.method}] {request.path}. 오류 메시지: '{error_message}', 코드: {error_code}, 상태 코드: {status_code}") # error_response에서 처리하도록 변경 예정
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = (
                400 if e.error_code else 500
            )  # ServiceError가 상태 코드를 결정할 수 있도록
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


@repository_ns.route("/search")
class RepositorySearch(Resource):
    @repository_ns.doc("search_repository")
    @repository_ns.expect(repo_search_request, validate=True)
    @repository_ns.response(200, "검색 완료", success_response_with_data)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(404, "인덱스 없음", error_model)  # INDEX_NOT_FOUND 경우
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """저장소에서 코드 검색"""
        data = {}
        try:
            data = request.get_json()
            repo_name, query, search_type = validate_search_request(data)

            # search_type은 이제 항상 "code" 또는 기본값 "code"
            if search_type != "code":
                # 혹시 다른 값이 들어올 경우를 대비한 방어 코드 (validate_search_request에서 처리될 수도 있음)
                logger.warning(
                    f"지원하지 않는 search_type: {search_type}. 'code'로 강제합니다."
                )
                search_type = "code"

            log_payload = {
                "repo_name": repo_name,
                "query": query,
                "search_type": search_type,  # 항상 "code"
            }
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 요청 데이터: {log_payload}"
            )

            # SearchService의 메서드 호출
            result = search_service.search_repository(
                f"https://github.com/{repo_name}",
                query,
                search_type,  # search_type은 "code"
            )

            message = "검색이 완료되었습니다."
            logger.info(
                f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 200. 메시지: {message} (저장소: {repo_name}, 질의: '{query}')"
            )
            return success_response(data=result, message=message)

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            if e.error_code == "INDEX_NOT_FOUND":
                status_code = 404
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


@repository_ns.route("/status/<path:repo_name>")
class RepositoryStatus(Resource):
    @repository_ns.doc("get_repository_status")
    @repository_ns.param(
        "repo_name", "저장소 이름 (owner/repository 형식)", required=True
    )
    @repository_ns.response(200, "상태 조회 성공 (완료됨)", success_response_with_data)
    @repository_ns.response(202, "인덱싱 진행 중", progress_response)
    @repository_ns.response(404, "저장소 정보 없음", error_model)
    @repository_ns.response(409, "인덱싱 실패", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def get(self, repo_name):
        """저장소 인덱싱 상태 확인. Express에서 호출되는 API."""
        try:
            log_payload = {"repo_name": repo_name}
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 요청 데이터: {log_payload}"
            )

            # StatusService의 메서드 호출
            status_data_result = status_service.get_repository_status_data(repo_name)
            current_status = status_data_result.get("status")

            # Express가 기대하는 응답 형식으로 변환
            response_data = {
                "analysis_id": repo_name,
                "repo_name": repo_name,
                "status": current_status,
                "progress": status_data_result.get("progress", 0),
                "message": status_data_result.get("progress_message", ""),
                "current_step": status_data_result.get("progress_message", ""),
                "error": status_data_result.get("error"),
                "error_code": status_data_result.get("error_code"),
                "started_at": status_data_result.get("start_time"),
                "completed_at": status_data_result.get("completion_time"),
                "estimated_completion": status_data_result.get("estimated_completion"),
                "eta_text": status_data_result.get("eta_text", "계산 중..."),
            }

            if current_status == "not_indexed":
                error_message = (
                    f"저장소 '{repo_name}'에 대한 인덱싱 정보를 찾을 수 없습니다."
                )
                error_code = "NOT_FOUND"
                status_code = 404
                logger.warning(
                    f"API 요청 처리 중 정보 없음 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}"
                )
                return error_response(
                    message=error_message,
                    error_code=error_code,
                    status_code=status_code,
                )
            elif current_status == "failed":
                error_message = f"저장소 '{repo_name}' 인덱싱에 실패했습니다: {status_data_result.get('error', '알 수 없는 오류')}"
                error_code = status_data_result.get("error_code", "INDEXING_FAILED")
                status_code = 409
                logger.error(
                    f"API 요청 처리 중 오류 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}"
                )
                return error_response(
                    message=error_message,
                    error_code=error_code,
                    status_code=status_code,
                )
            elif current_status in ["pending", "indexing"]:
                message = f"저장소 '{repo_name}' 인덱싱 진행 중입니다."
                logger.info(
                    f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 202. 메시지: {message}"
                )
                return success_response(
                    data=response_data, message=message, status_code=202
                )
            elif current_status == "completed":
                message = f"저장소 '{repo_name}' 인덱싱이 완료되었습니다."
                logger.info(
                    f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 200. 메시지: {message}"
                )
                return success_response(data=response_data, message=message)
            else:  # 알 수 없는 상태
                error_message = f"알 수 없는 상태값({current_status})입니다."
                error_code = "UNKNOWN_STATUS"
                status_code = 500
                logger.error(
                    f"API 요청 처리 중 알 수 없는 상태값 [{request.method} {request.path}]: 저장소 '{repo_name}'의 상태 '{current_status}'는 처리할 수 없습니다."
                )
                return error_response(
                    message=error_message,
                    error_code=error_code,
                    status_code=status_code,
                )

        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 저장소: {repo_name}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "상태 조회 중 서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 저장소: {repo_name}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


# README 요약 관련 모델들
readme_summary_request = api.model(
    "ReadmeSummaryRequest",
    {
        "repo_name": fields.String(
            required=True,
            description="GitHub 저장소 이름 (owner/repository 형식)",
            example="pallets/flask",
        ),
        "readme_content": fields.String(
            required=True,
            description="README 파일 내용",
            example="# Flask\\n\\nFlask is a lightweight WSGI web application framework...",
        ),
    },
)

readme_summary_response = api.model(
    "ReadmeSummaryResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="README 요약이 완료되었습니다.",
        ),
        "data": fields.Raw(
            description="요약 결과",
            example={
                "repo_name": "pallets/flask",
                "summary": "Flask는 Python으로 작성된 경량 웹 애플리케이션 프레임워크입니다...",
                "original_length": 1500,
                "summary_length": 120,
            },
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:20:00Z",
        ),
    },
)

# 번역 관련 모델들
translation_request = api.model(
    "TranslationRequest",
    {
        "text": fields.String(
            required=True,
            description="번역할 텍스트",
            example="A lightweight WSGI web application framework",
        ),
        "source_language": fields.String(
            required=False,
            description="원본 언어 (기본값: auto)",
            default="auto",
            example="en",
        ),
        "target_language": fields.String(
            required=False,
            description="대상 언어 (기본값: ko)",
            default="ko",
            example="ko",
        ),
    },
)

translation_response = api.model(
    "TranslationResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(
            required=True,
            description="응답 메시지",
            example="번역이 완료되었습니다.",
        ),
        "data": fields.Raw(
            description="번역 결과",
            example={
                "original_text": "A lightweight WSGI web application framework",
                "translated_text": "경량 WSGI 웹 애플리케이션 프레임워크",
                "source_language": "en",
                "target_language": "ko",
            },
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:20:00Z",
        ),
    },
)


@repository_ns.route("/summarize-readme")
class ReadmeSummary(Resource):
    @repository_ns.doc("summarize_readme")
    @repository_ns.expect(readme_summary_request, validate=True)
    @repository_ns.response(200, "README 요약 완료", readme_summary_response)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """README 내용을 AI로 요약합니다. Express에서 호출되는 API."""
        import asyncio  # 함수 내부에서 임포트

        data = {}
        try:
            data = request.get_json()
            repo_name = data.get("repo_name")
            readme_content_length = len(data.get("readme_content", ""))

            log_payload = {
                "repo_name": repo_name,
                "readme_content_length": readme_content_length,
            }
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 요청 데이터: {log_payload}"
            )

            if not data:
                error_message = "요청 데이터가 필요합니다."
                logger.warning(
                    f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
                )
                return error_response(
                    message=error_message, error_code="MISSING_DATA", status_code=400
                )

            readme_content = data.get("readme_content")

            if not repo_name:
                error_message = "저장소 이름이 필요합니다."
                logger.warning(
                    f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
                )
                return error_response(
                    message=error_message,
                    error_code="MISSING_REPO_NAME",
                    status_code=400,
                )

            if not readme_content:
                error_message = "README 내용이 필요합니다."
                logger.warning(
                    f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
                )
                return error_response(
                    message=error_message,
                    error_code="MISSING_README_CONTENT",
                    status_code=400,
                )

            # README 요약 수행 (비동기 함수 동기 실행)
            summary = asyncio.run(
                readme_summarizer.summarize_readme(repo_name, readme_content)
            )

            if summary:
                response_data = {
                    "repo_name": repo_name,
                    "summary": summary,
                    "original_length": len(readme_content),
                    "summary_length": len(summary),
                }
                message = f"README 요약이 완료되었습니다: {repo_name}"
                logger.info(
                    f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 200. 메시지: {message}"
                )
                return success_response(
                    data=response_data, message=message, status_code=200
                )
            else:
                fallback_description = readme_summarizer.create_fallback_description(
                    repo_name
                )
                response_data = {
                    "repo_name": repo_name,
                    "summary": fallback_description,
                    "original_length": len(readme_content),
                    "summary_length": len(fallback_description),
                    "is_fallback": True,
                }
                message = (
                    f"README 요약에 실패하여 기본 설명을 생성했습니다: {repo_name}"
                )
                logger.warning(
                    f"README 요약 실패, 기본 설명 사용: {repo_name}. API 응답 메시지: {message}"
                )
                logger.info(
                    f"API 요청 성공 (대체 응답): [{request.method}] {request.path}. 응답 코드: 200. 메시지: {message}"
                )
                return success_response(
                    data=response_data, message=message, status_code=200
                )

        except ValidationError as e:
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "README 요약 중 서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )


@repository_ns.route("/translate")
class Translation(Resource):
    """텍스트 번역 API"""

    @repository_ns.doc("translate_text")
    @repository_ns.expect(translation_request, validate=True)
    @repository_ns.response(200, "번역 완료", translation_response)
    @repository_ns.response(400, "잘못된 요청", error_model)
    @repository_ns.response(500, "서버 내부 오류", error_model)
    def post(self):
        """텍스트를 번역합니다."""
        data = {}
        try:
            data = request.get_json()
            text = data.get("text", "").strip()
            source_language = data.get("source_language", "auto")
            target_language = data.get("target_language", "ko")

            log_payload = {
                "text_length": len(text),
                "source_language": source_language,
                "target_language": target_language,
            }
            logger.info(
                f"API 요청 시작: [{request.method}] {request.path} (클라이언트 IP: {request.remote_addr}). 요청 데이터: {log_payload}"
            )

            if not text:
                error_message = "번역할 텍스트가 필요합니다."
                error_code = "MISSING_TEXT"
                status_code = 400
                logger.warning(
                    f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
                )
                return error_response(
                    message=error_message,
                    error_code=error_code,
                    status_code=status_code,
                )

            # 번역 서비스 초기화 및 실행 (임포트를 try 블록 안으로 옮겨서 필요 시에만 로드)
            from ..services.translator import Translator

            translator = Translator()

            translated_text = translator.translate_text(
                text=text,
                source_language=source_language,
                target_language=target_language,
            )

            if translated_text:
                response_data = {
                    "original_text": text,
                    "translated_text": translated_text,
                    "source_language": source_language,  # 실제 감지된 언어 또는 제공된 언어
                    "target_language": target_language,
                }
                message = "번역이 완료되었습니다."
                logger.info(
                    f"API 요청 성공: [{request.method}] {request.path}. 응답 코드: 200. 메시지: {message} ({len(text)}자 -> {len(translated_text)}자, {source_language} -> {target_language})"
                )
                return success_response(message=message, data=response_data)
            else:
                error_message = "번역에 실패했습니다. 원본 텍스트를 사용해주세요."
                error_code = "TRANSLATION_FAILED"
                status_code = 500  # 서비스 실패로 간주
                logger.error(
                    f"API 요청 처리 중 번역 실패 [{request.method} {request.path}]: Text length {len(text)}, Source: {source_language}, Target: {target_language}"
                )
                return error_response(
                    message=error_message,
                    error_code=error_code,
                    status_code=status_code,
                )

        except ValidationError as e:  # 이 경우는 보통 validate_xxx 함수에서 발생
            error_message = str(e)
            error_code = "VALIDATION_ERROR"
            status_code = 400
            logger.warning(
                f"API 요청 유효성 검사 실패 [{request.method} {request.path}]: {error_message}. 요청 데이터: {data}"
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except ServiceError as e:
            error_message = str(e)
            error_code = e.error_code or "SERVICE_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 서비스 오류 발생 [{request.method} {request.path}]: {error_message}. ErrorCode: {error_code}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
        except Exception as e:
            error_message = "번역 중 서버 내부 오류가 발생했습니다."
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            logger.error(
                f"API 요청 처리 중 예상치 못한 오류 발생 [{request.method} {request.path}]: {e}. 요청 데이터: {data}",
                exc_info=True,
            )
            return error_response(
                message=error_message, error_code=error_code, status_code=status_code
            )
