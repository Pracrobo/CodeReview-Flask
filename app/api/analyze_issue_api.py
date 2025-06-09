from flask import request, current_app
from flask_restx import Namespace, Resource, fields

from app.services.issue_analyzer import issue_analyzer
from app.services.faiss_service import FAISSService
from app.services.embeddings import GeminiAPIEmbeddings
from app.core.config import Config
from app.core.utils import extract_repo_name_from_url, get_faiss_index_path
from app.core.exceptions import ServiceError, RAGError

# 네임스페이스 생성
issue_ns = Namespace("issue", description="이슈 분석 관련 API")

# API 모델 정의
issue_analysis_request_model = issue_ns.model(
    "IssueAnalysisRequest",
    {
        "title": fields.String(required=True, description="이슈 제목"),
        "body": fields.String(description="이슈 본문"),
        "issueId": fields.Integer(required=True, description="이슈 ID"),
        "repoUrl": fields.String(required=True, description="저장소 URL"),
        "defaultBranch": fields.String(
            required=False, description="저장소 기본 브랜치명", default="main"
        ),
    },
)

issue_analysis_response_model = issue_ns.model(
    "IssueAnalysisResponse",
    {
        "summary": fields.String(description="AI 요약 정보"),
        "relatedFiles": fields.List(
            fields.Nested(
                issue_ns.model(
                    "RelatedFile",
                    {
                        "path": fields.String(description="파일 경로"),
                        "relevance": fields.Float(description="관련도 점수"),
                        "githubUrl": fields.String(description="GitHub 파일 URL"),
                    },
                )
            )
        ),
        "codeSnippets": fields.List(
            fields.Nested(
                issue_ns.model(
                    "CodeSnippet",
                    {
                        "file": fields.String(description="코드 스니펫이 포함된 파일"),
                        "code": fields.String(description="코드 스니펫"),
                        "relevance": fields.Float(description="관련도 점수"),
                        "explanation": fields.String(description="코드 스니펫 설명"),
                    },
                )
            )
        ),
        "solutionSuggestion": fields.String(description="AI 해결 제안"),
    },
)


@issue_ns.route("/analyze-issue")
class AnalyzeIssue(Resource):
    @issue_ns.expect(issue_analysis_request_model)
    @issue_ns.marshal_with(issue_analysis_response_model)
    def post(self):
        """이슈 정보를 받아 AI 분석을 수행합니다."""
        data = request.json
        issue_id = data.get("issueId")
        repo_url = data.get("repoUrl")
        default_branch = (
            data.get("defaultBranch") or "main"
        )  # Express에서 전달된 defaultBranch 사용

        # 저장소 URL 필수 검증
        if not repo_url:
            current_app.logger.error(f"저장소 URL이 누락됨: 이슈 ID {issue_id}")
            return {
                "summary": "저장소 URL이 필요합니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "분석을 위해 저장소 URL을 제공해주세요.",
            }, 400

        current_app.logger.info(
            f"이슈 분석 요청 수신: 이슈 ID {issue_id}, 저장소: {repo_url}"
        )

        try:
            # 저장소 이름 추출
            repo_name = extract_repo_name_from_url(repo_url)
            current_app.logger.info(f"저장소 이름 추출: {repo_name}")
        except ValueError as e:
            current_app.logger.error(f"유효하지 않은 저장소 URL: {repo_url}, 오류: {e}")
            return {
                "summary": "유효하지 않은 저장소 URL입니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "올바른 저장소 URL을 제공해주세요.",
            }, 400

        # FAISSService 인스턴스 생성
        try:
            embeddings_instance = GeminiAPIEmbeddings(
                model_name=Config.DEFAULT_EMBEDDING_MODEL,
                document_task_type="RETRIEVAL_DOCUMENT",
                query_task_type="RETRIEVAL_QUERY",
            )
            faiss_service_instance = FAISSService(embeddings=embeddings_instance)
            current_app.logger.info("FAISSService 초기화 완료")
        except Exception as e_init:
            current_app.logger.error(
                f"FAISSService 초기화 실패: {e_init}", exc_info=True
            )
            return {
                "summary": "AI 분석 서비스 초기화 중 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "서비스 초기화 오류로 인해 분석을 수행할 수 없습니다.",
            }, 500

        # 저장소별 벡터 스토어 로드
        try:
            # 코드 인덱스 경로 확인 및 로드
            code_index_path = get_faiss_index_path(repo_name, "code")
            current_app.logger.info(f"코드 인덱스 로드 시도: {code_index_path}")

            code_vector_store = faiss_service_instance.load_index(
                code_index_path, "code"
            )

            if not code_vector_store:
                current_app.logger.warning(
                    f"저장소 '{repo_name}'의 코드 인덱스를 찾을 수 없습니다."
                )
                return {
                    "summary": f"'{repo_name}' 저장소의 분석 데이터를 찾을 수 없습니다.",
                    "relatedFiles": [],
                    "codeSnippets": [],
                    "solutionSuggestion": "저장소가 아직 인덱싱되지 않았습니다. 먼저 저장소를 분석해주세요.",
                }, 404

            # 벡터 스토어 딕셔너리 구성
            vector_stores = {"code": code_vector_store}
            current_app.logger.info(f"저장소 '{repo_name}'의 벡터 스토어 로드 완료")

        except Exception as e_load:
            current_app.logger.error(
                f"벡터 스토어 로드 실패 (저장소: {repo_name}): {e_load}", exc_info=True
            )
            return {
                "summary": "저장소 분석 데이터 로드 중 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "분석 데이터 로드 오류로 인해 분석을 수행할 수 없습니다.",
            }, 500

        # 이슈 분석 수행
        try:
            current_app.logger.info(
                f"이슈 분석 시작: 이슈 ID {issue_id}, 저장소: {repo_name}"
            )

            analysis_result = issue_analyzer.analyze_issue(
                vector_stores=vector_stores,
                issue_data=data,
                faiss_service=faiss_service_instance,
                default_branch=default_branch,  # default_branch 전달
            )

            current_app.logger.info(f"이슈 분석 완료: 이슈 ID {issue_id}")
            return analysis_result, 200

        except RAGError as e_rag:
            current_app.logger.error(
                f"RAG 분석 오류 (이슈 ID: {issue_id}): {e_rag}", exc_info=True
            )
            return {
                "summary": "AI 분석 중 검색 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "검색 엔진 오류로 인해 분석을 완료할 수 없습니다.",
            }, 500

        except ServiceError as e_service:
            current_app.logger.error(
                f"서비스 오류 (이슈 ID: {issue_id}): {e_service}", exc_info=True
            )
            return {
                "summary": "분석 서비스 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "서비스 오류로 인해 분석을 완료할 수 없습니다.",
            }, 500

        except Exception as e:
            current_app.logger.error(
                f"이슈 분석 중 예상치 못한 오류 (이슈 ID: {issue_id}): {e}",
                exc_info=True,
            )
            return {
                "summary": "AI 분석 중 예상치 못한 오류가 발생했습니다.",
                "relatedFiles": [],
                "codeSnippets": [],
                "solutionSuggestion": "시스템 오류로 인해 분석을 완료할 수 없습니다.",
            }, 500
