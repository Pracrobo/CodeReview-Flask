"""Swagger 설정 및 공통 모델 정의"""

from flask_restx import Api, fields

# API 설정
api = Api(
    version="1.0",
    title="AIssue Repository RAG 분석 API",
    description="GitHub 저장소의 코드와 문서를 분석하여 RAG 기반 검색을 제공하는 API",
    doc="/docs/",
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

# 상태 정보 모델
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

# 검색 결과 모델
search_result_item = api.model(
    "SearchResultItem",
    {
        "content": fields.String(
            description="검색된 내용",
            example="from flask import request\n\n@app.route('/login', methods=['POST'])\ndef login():\n    username = request.form['username']\n    # ...",
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

search_results = api.model(
    "SearchResults",
    {
        "query": fields.String(
            description="검색 질의문", example="Flask에서 request 객체 사용법"
        ),
        "search_type": fields.String(description="검색 유형", example="code"),
        "repo_name": fields.String(description="저장소 이름", example="pallets/flask"),
        "results": fields.List(
            fields.Nested(search_result_item),
            description="검색 결과 목록",
            example=[
                {
                    "content": "from flask import request\n\n@app.route('/submit', methods=['POST'])\ndef submit_form():\n    name = request.form.get('name')\n    email = request.form.get('email')\n    # ... process data ...\n    return 'Form submitted successfully!'",
                    "file_path": "src/flask/wrappers.py",  #
                    "score": 0.88,
                    "metadata": {"line_number": 150, "type": "code_snippet"},
                },
                {
                    "content": "The request object gives you access to incoming request data. \nFor example, `request.args` for URL parameters, `request.form` for form data.",
                    "file_path": "docs/quickstart.rst",
                    "score": 0.82,
                    "metadata": {
                        "section_title": "Accessing Request Data",
                        "type": "documentation_snippet",
                    },
                },
            ],
        ),
        "total_results": fields.Integer(description="총 결과 수", example=2),
        "search_time": fields.Float(description="검색 소요 시간 (초)", example=0.250),
        "result_generated_at": fields.String(  # 'timestamp'에서 이름 변경
            description="검색 결과 생성 시간 (ISO 8601)",  # 설명 변경
            example="2024-07-22T11:18:00Z",  # 예시 시간 조정 가능
        ),
    },
)

# 성공 응답 모델들
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
            search_results, description="응답 데이터"
        ),  # 여기의 search_results는 위에서 변경된 모델을 따름
        "timestamp": fields.String(  # 이 timestamp는 API 응답 생성 시점의 timestamp로 유지
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
            description="진행 상황 데이터",
            example={
                "status": "indexing",
                "repo_name": "pallets/flask",
                "repo_url": "https://github.com/pallets/flask",
                "progress": 65.0,
                "current_step": "Flask 소스 코드 파일 분석 중...",
                "total_files": 850,
                "processed_files": 550,
                "start_time": "2024-07-22T11:00:00Z",
                "completion_time": None,
                "error": None,
            },
        ),
        "timestamp": fields.String(
            required=True,
            description="응답 시간 (ISO 8601)",
            example="2024-07-22T11:05:00Z",
        ),
    },
)
