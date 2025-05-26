"""Swagger 설정 및 공통 모델 정의"""

from flask_restx import Api, fields

# API 설정
api = Api(
    version="1.0",
    title="AIssue Repository RAG 분석 API",
    description="GitHub 저장소의 코드와 문서를 분석하여 RAG 기반 검색을 제공하는 API",
    doc="/docs/",  # Swagger UI 경로
    contact="개발팀",
    contact_email="dev@aiissue.com",
)

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

error_response = api.model(
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
            example="https://github.com/owner/repository",
        )
    },
)

repo_search_request = api.model(
    "RepositorySearchRequest",
    {
        "repo_name": fields.String(
            required=True,
            description="GitHub 저장소 이름 (owner/repository 형식)",
            example="owner/repository",
        ),
        "query": fields.String(
            required=True,
            description="검색 질의문",
            example="사용자 인증 관련 함수를 찾아줘",
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

# 상태 정보 모델
status_data = api.model(
    "StatusData",
    {
        "status": fields.String(description="인덱싱 상태", example="completed"),
        "repo_name": fields.String(
            description="저장소 이름", example="owner/repository"
        ),
        "repo_url": fields.String(description="저장소 URL"),
        "progress": fields.Float(description="진행률 (0-100)", example=75.5),
        "current_step": fields.String(
            description="현재 단계", example="코드 파일 인덱싱 중"
        ),
        "total_files": fields.Integer(description="전체 파일 수", example=150),
        "processed_files": fields.Integer(description="처리된 파일 수", example=120),
        "start_time": fields.String(description="시작 시간"),
        "completion_time": fields.String(description="완료 시간"),
        "error": fields.String(description="오류 메시지"),
    },
)

# 검색 결과 모델
search_result_item = api.model(
    "SearchResultItem",
    {
        "content": fields.String(description="검색된 내용"),
        "file_path": fields.String(description="파일 경로"),
        "score": fields.Float(description="유사도 점수"),
        "metadata": fields.Raw(description="추가 메타데이터"),
    },
)

search_results = api.model(
    "SearchResults",
    {
        "query": fields.String(description="검색 질의문"),
        "search_type": fields.String(description="검색 유형"),
        "repo_name": fields.String(description="저장소 이름"),
        "results": fields.List(
            fields.Nested(search_result_item), description="검색 결과 목록"
        ),
        "total_results": fields.Integer(description="총 결과 수"),
        "search_time": fields.Float(description="검색 소요 시간 (초)"),
    },
)

# 성공 응답 모델들
success_response_with_data = api.model(
    "SuccessResponseWithData",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="success"
        ),
        "message": fields.String(required=True, description="응답 메시지"),
        "data": fields.Raw(description="응답 데이터"),
        "timestamp": fields.String(required=True, description="응답 시간 (ISO 8601)"),
    },
)

progress_response = api.model(
    "ProgressResponse",
    {
        "status": fields.String(
            required=True, description="응답 상태", example="in_progress"
        ),
        "message": fields.String(required=True, description="응답 메시지"),
        "progress": fields.Nested(status_data, description="진행 상황 데이터"),
        "timestamp": fields.String(required=True, description="응답 시간 (ISO 8601)"),
    },
)
