import logging
import time
from typing import Dict, List, Any

from app.core.config import Config
from app.core.exceptions import RAGError, EmbeddingError
from app.core.prompts import prompts
from app.services.gemini_service import gemini_service
from app.services.faiss_service import FAISSService

logger = logging.getLogger(__name__)


class IssueAnalyzer:
    """이슈 분석 및 AI 기반 해결 제안 서비스"""

    def __init__(self):
        self.llm_model = Config.DEFAULT_LLM_MODEL

    def analyze_issue(
        self,
        vector_stores: Dict,
        issue_data: Dict,
        faiss_service: FAISSService,
        default_branch: str = "main",  # 기본 브랜치명 인자 받도록 수정
    ) -> Dict[str, Any]:
        """
        이슈를 분석하여 AI 요약, 관련 파일, 코드 스니펫, 해결 제안을 생성

        Args:
            vector_stores: 벡터 저장소 딕셔너리 (빈 딕셔너리일 수 있음)
            issue_data: 이슈 정보 {'title': str, 'body': str, 'issueId': int}
            faiss_service: FAISSService 인스턴스
            default_branch: 저장소의 기본 브랜치명
        Returns:
            분석 결과 딕셔너리
        """
        try:
            issue_title = issue_data.get("title", "")
            issue_body = issue_data.get("body", "")
            repo_url = issue_data.get("repoUrl", "")
            issue_id = issue_data.get("issueId", "")

            start_time = time.time()
            logger.info(
                f"[AIssue] 이슈 분석 시작: (이슈 ID: {issue_id}, 저장소: {repo_url})"
            )

            # AI 요약 생성
            summary = self._generate_issue_summary(issue_title, issue_body)

            # 벡터 스토어가 있는 경우에만 코드 검색 수행
            if vector_stores and "code" in vector_stores:
                search_question = self._convert_issue_to_question(
                    issue_title, issue_body
                )

                logger.info(
                    f"[AIssue] 코드 검색용 질문 변환 완료: '{search_question[:80]}'"
                )

                search_results_docs_scores = self._search_related_code(
                    vector_stores=vector_stores,
                    search_question=search_question,
                    faiss_service=faiss_service,
                    top_k=Config.DEFAULT_TOP_K,
                    similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
                )

                if not search_results_docs_scores:
                    logger.warning(
                        f"[AIssue] 코드 검색 결과가 없습니다. (이슈 ID: {issue_id}, 저장소: {repo_url})"
                    )

                related_files = self._extract_related_files(
                    search_results_docs_scores,
                    repo_url,
                    default_branch,  # default_branch 전달
                )
                code_snippets = self._extract_code_snippets(search_results_docs_scores)
            else:
                logger.info("[AIssue] 벡터 스토어가 없어 코드 검색을 건너뜁니다.")
                search_results_docs_scores = []
                related_files = []
                code_snippets = []

            # 해결 제안 생성
            solution_suggestion = self._generate_solution_suggestion(
                issue_title, issue_body, related_files, code_snippets
            )

            result = {
                "summary": summary,
                "relatedFiles": related_files,
                "codeSnippets": code_snippets,
                "solutionSuggestion": solution_suggestion,
            }

            elapsed = round(time.time() - start_time, 2)
            logger.info(
                f"[AIssue] 이슈 분석 완료: (이슈 ID: {issue_id}, 저장소: {repo_url}, 소요: {elapsed}초)"
            )
            return result

        except Exception as e:
            logger.error(f"[AIssue] 이슈 분석 중 오류 발생: {e}", exc_info=True)
            raise RAGError(f"이슈 분석 실패: {e}") from e

    def _generate_issue_summary(self, issue_title: str, issue_body: str) -> str:
        """이슈 내용을 AI로 요약"""
        try:
            client = gemini_service.get_client()
            if not client:
                logger.error(
                    "이슈 요약 생성 실패: Gemini 클라이언트를 초기화할 수 없습니다."
                )
                return "AI 요약 생성 중 오류가 발생했습니다."

            prompt = prompts.get_issue_summary_prompt(issue_title, issue_body)
            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )
            summary = gemini_service.extract_text_from_response(response)
            return summary or "AI 요약을 생성할 수 없습니다."
        except Exception as e:
            logger.warning(f"이슈 요약 생성 실패: {e}")
            return "AI 요약 생성 중 오류가 발생했습니다."

    def _convert_issue_to_question(self, issue_title: str, issue_body: str) -> str:
        """이슈 내용을 검색용 질문으로 변환"""
        try:
            client = gemini_service.get_client()
            if not client:
                logger.error(
                    "이슈 질문 변환 실패: Gemini 클라이언트를 초기화할 수 없습니다."
                )
                return issue_title  # 오류 시 원본 제목 반환

            prompt = prompts.get_issue_to_question_prompt(issue_title, issue_body)
            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )
            question = gemini_service.extract_text_from_response(response)
            return question or issue_title
        except Exception as e:
            logger.warning(f"이슈 질문 변환 실패: {e}")
            return issue_title

    def _search_related_code(
        self,
        vector_stores: Dict,
        search_question: str,
        faiss_service: FAISSService,
        top_k: int = Config.DEFAULT_TOP_K,
        similarity_threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD,
    ) -> List:  # 반환 타입은 이제 List[Tuple[Document, float]]
        """관련 코드 검색 (FAISSService 사용)"""
        from app.services.searcher import (
            translate_code_query_to_english,
            preprocess_text,
        )

        if "code" not in vector_stores or not vector_stores["code"]:
            logger.warning("코드 벡터 저장소를 사용할 수 없습니다.")
            return []

        vector_store = vector_stores["code"]

        try:
            english_query = translate_code_query_to_english(
                search_question, self.llm_model
            )
            processed_query = preprocess_text(english_query)

            logger.info(
                f"FAISSService를 사용하여 코드 검색 시작: '{processed_query[:100]}...'"
            )

            doc_score_pairs = faiss_service.search_documents(
                vector_store,
                processed_query,
                top_k,
                similarity_threshold,
            )

            logger.info(
                f"FAISSService 코드 검색 결과: {len(doc_score_pairs)}개 문서 반환됨."
            )
            return doc_score_pairs  # (Document, score) 튜플 리스트 반환

        except (
            EmbeddingError
        ) as e_embed:  # search_documents 내 embed_query에서 발생 가능
            logger.error(f"코드 검색 중 쿼리 임베딩 오류: {e_embed}")
            return []
        except Exception as e:
            logger.error(f"코드 검색 중 오류: {e}", exc_info=True)
            return []

    def _extract_related_files(
        self,
        search_results: list,
        repo_url: str,
        default_branch: str,
    ) -> list:
        files_dict = {}
        owner, repo = self._extract_owner_repo_from_url(repo_url)
        for doc, score in search_results:
            raw_file_path = doc.metadata.get("source", "알 수 없음")
            if raw_file_path != "알 수 없음":
                file_path = self._normalize_file_path(raw_file_path)
                github_url = self._make_github_file_url(
                    owner, repo, default_branch, file_path
                )
                if (
                    file_path
                    and file_path not in files_dict
                    or (
                        file_path in files_dict
                        and files_dict[file_path]["relevance"] < score
                    )
                ):
                    files_dict[file_path] = {
                        "path": file_path,
                        "relevance": round(score * 100, 1),
                        "githubUrl": github_url,  # githubUrl 필드 포함
                    }
        sorted_files = sorted(
            files_dict.values(), key=lambda x: x["relevance"], reverse=True
        )
        return sorted_files[:3]

    def _extract_owner_repo_from_url(self, repo_url: str):
        # https://github.com/owner/repo 형태에서 owner, repo 추출
        try:
            parts = repo_url.rstrip("/").split("/")
            owner = parts[-2]
            repo = parts[-1]
            if repo.endswith(".git"):
                repo = repo[:-4]
            return owner, repo
        except Exception:
            return "", ""

    def _make_github_file_url(
        self, owner: str, repo: str, branch: str, file_path: str
    ) -> str:
        import re

        common_branch_pattern = (
            r"^(main|master|dev|develop|release|staging|production)/"
        )
        cleaned_path = re.sub(common_branch_pattern, "", file_path, count=1)
        cleaned_path = cleaned_path.lstrip("/")
        return f"https://github.com/{owner}/{repo}/blob/{branch}/{cleaned_path}"

    def _extract_code_snippets(self, search_results: list) -> list:
        # 코드 스니펫에는 실제 코드만, 관련 파일 정보가 섞이지 않도록 분리
        snippets = []
        for doc, score in search_results[:2]:  # 상위 2개 결과만 사용
            code_content = doc.page_content if hasattr(doc, "page_content") else ""
            raw_file_path = doc.metadata.get("source", "알 수 없음")
            file_path = (
                self._normalize_file_path(raw_file_path)
                if raw_file_path != "알 수 없음"
                else "알 수 없음"
            )

            # 코드 내용에서 "관련 파일:" 로 시작하는 메타데이터성 정보 제거
            # 또는 해당 정보가 포함된 경우 스니펫으로 사용하지 않음
            # 여기서는 간단히 해당 문자열로 시작하는 경우 스킵하도록 처리
            if code_content.strip().startswith("관련 파일:"):
                logger.debug(
                    f"코드 스니펫에서 '관련 파일:' 접두사 발견, 스킵: {file_path}"
                )
                continue

            explanation = (
                f"이 코드는 이슈와 {round(score * 100, 1)}% 관련되어 있습니다."
            )
            snippets.append(
                {
                    "file": file_path,
                    "code": code_content,
                    "relevance": round(score * 100, 1),
                    "explanation": explanation,
                }
            )
        return snippets

    def _normalize_file_path(self, raw_path: str) -> str:
        """파일 경로를 정규화하여 상대 경로만 반환"""
        import re

        if not raw_path or raw_path == "알 수 없음":
            return raw_path

        # Windows/Unix 경로 구분자 통일
        normalized_path = raw_path.replace("\\", "/")

        # 로컬 시스템 경로 패턴 제거
        # 예: C:\src\AIssue\AIssue-BE-Flask\cloned_repos\scrapy\ 부분 제거
        patterns = [
            r"^[A-Za-z]:[/\\].*?[/\\]cloned_repos[/\\][^/\\]+[/\\]",  # Windows 절대 경로
            r"^/.*?/cloned_repos/[^/]+/",  # Unix 절대 경로
            r".*cloned_repos[/\\][^/\\]+[/\\]",  # 일반적인 cloned_repos 패턴
        ]

        for pattern in patterns:
            normalized_path = re.sub(pattern, "", normalized_path)

        # 시작 부분의 슬래시 제거
        normalized_path = normalized_path.lstrip("/")

        return normalized_path if normalized_path else raw_path

    def _generate_solution_suggestion(
        self,
        issue_title: str,
        issue_body: str,
        related_files: List,
        code_snippets: List,
    ) -> str:
        """AI 해결 제안 생성"""
        try:
            client = gemini_service.get_client()
            if not client:
                logger.error(
                    "해결 제안 생성 실패: Gemini 클라이언트를 초기화할 수 없습니다."
                )
                return "AI 해결 제안 생성 중 오류가 발생했습니다."

            prompt = prompts.get_ai_solution_suggestion_prompt(
                issue_title, issue_body, related_files, code_snippets
            )
            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )
            suggestion = gemini_service.extract_text_from_response(response)
            return suggestion or "AI 해결 제안을 생성할 수 없습니다."
        except Exception as e:
            logger.warning(f"해결 제안 생성 실패: {e}")
            return "AI 해결 제안 생성 중 오류가 발생했습니다."


# 전역 인스턴스
issue_analyzer = IssueAnalyzer()
