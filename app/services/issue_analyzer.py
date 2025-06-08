import logging
from typing import Dict, List, Any

from app.core.config import Config
from app.core.exceptions import RAGError
from app.core.prompts import prompts
from app.services.gemini_service import gemini_service
from app.services.searcher import (
    translate_code_query_to_english,
    preprocess_text,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


class IssueAnalyzer:
    """이슈 분석 및 AI 기반 해결 제안 서비스"""

    def __init__(self):
        self.llm_model_name = Config.LLM_MODEL_NAME

    async def analyze_issue(
        self, vector_stores: Dict, issue_data: Dict
    ) -> Dict[str, Any]:
        """
        이슈를 분석하여 AI 요약, 관련 파일, 코드 스니펫, 해결 제안을 생성

        Args:
            vector_stores: 벡터 저장소 딕셔너리
            issue_data: 이슈 정보 {'title': str, 'body': str, 'issueId': int}

        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 1. Gemini 클라이언트 초기화 확인
            client = gemini_service.get_client()
            if not client:
                raise RAGError("Gemini 클라이언트를 초기화할 수 없습니다.")

            issue_title = issue_data.get("title", "")
            issue_body = issue_data.get("body", "")

            logger.info(f"이슈 분석 시작: {issue_title}")

            # 2. 이슈 내용을 AI 요약
            summary = await self._generate_issue_summary(
                client, issue_title, issue_body
            )

            # 3. 이슈를 검색용 질문으로 변환
            search_question = await self._convert_issue_to_question(
                client, issue_title, issue_body
            )

            # 4. 코드 검색 수행
            search_results = await self._search_related_code(
                vector_stores, search_question
            )

            # 5. 관련 파일 및 코드 스니펫 추출
            related_files = self._extract_related_files(search_results)
            code_snippets = self._extract_code_snippets(search_results)

            # 6. AI 해결 제안 생성
            solution_suggestion = await self._generate_solution_suggestion(
                client, issue_title, issue_body, related_files, code_snippets
            )

            result = {
                "summary": summary,
                "relatedFiles": related_files,
                "codeSnippets": code_snippets,
                "solutionSuggestion": solution_suggestion,
            }

            logger.info(f"이슈 분석 완료: {issue_title}")
            return result

        except Exception as e:
            logger.error(f"이슈 분석 중 오류 발생: {e}", exc_info=True)
            raise RAGError(f"이슈 분석 실패: {e}") from e

    async def _generate_issue_summary(
        self, client, issue_title: str, issue_body: str
    ) -> str:
        """이슈 내용을 AI로 요약"""
        try:
            prompt = prompts.get_issue_summary_prompt(issue_title, issue_body)
            response = client.models.generate_content(
                model=self.llm_model_name, contents=prompt
            )
            summary = gemini_service.extract_text_from_response(response)
            return summary or "AI 요약을 생성할 수 없습니다."
        except Exception as e:
            logger.warning(f"이슈 요약 생성 실패: {e}")
            return "AI 요약 생성 중 오류가 발생했습니다."

    async def _convert_issue_to_question(
        self, client, issue_title: str, issue_body: str
    ) -> str:
        """이슈 내용을 검색용 질문으로 변환"""
        try:
            prompt = prompts.get_issue_to_question_prompt(issue_title, issue_body)
            response = client.models.generate_content(
                model=self.llm_model_name, contents=prompt
            )
            question = gemini_service.extract_text_from_response(response)
            return question or issue_title
        except Exception as e:
            logger.warning(f"이슈 질문 변환 실패: {e}")
            return issue_title

    async def _search_related_code(
        self, vector_stores: Dict, search_question: str
    ) -> List:
        """관련 코드 검색"""
        if "code" not in vector_stores or not vector_stores["code"]:
            logger.warning("코드 벡터 저장소를 사용할 수 없습니다.")
            return []

        vector_store = vector_stores["code"]

        try:
            # 영어로 번역
            english_query = translate_code_query_to_english(
                search_question, self.llm_model_name
            )
            processed_query = preprocess_text(english_query)

            # 임베딩 검색
            query_embedding = vector_store.embedding_function.embed_query(
                processed_query
            )
            num_vectors = vector_store.index.ntotal

            doc_score_pairs = []
            for idx in range(num_vectors):
                docstore_id = vector_store.index_to_docstore_id[idx]
                doc = vector_store.docstore._dict[docstore_id]
                doc_embedding = vector_store.index.reconstruct(idx)
                sim = cosine_similarity(query_embedding, doc_embedding)
                doc_score_pairs.append((doc, sim))

            # 상위 10개 결과 반환
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            return doc_score_pairs[:10]

        except Exception as e:
            logger.error(f"코드 검색 중 오류: {e}")
            return []

    def _extract_related_files(self, search_results: List) -> List[Dict]:
        """검색 결과에서 관련 파일 추출 (상위 3개)"""
        files_dict = {}

        for doc, score in search_results:
            file_path = doc.metadata.get("source", "알 수 없음")
            if (
                file_path not in files_dict
                or files_dict[file_path]["relevance"] < score
            ):
                files_dict[file_path] = {
                    "path": file_path,
                    "relevance": round(score * 100, 1),
                }

        # 유사도 기준으로 정렬하여 상위 3개 반환
        sorted_files = sorted(
            files_dict.values(), key=lambda x: x["relevance"], reverse=True
        )
        return sorted_files[:3]

    def _extract_code_snippets(self, search_results: List) -> List[Dict]:
        """검색 결과에서 코드 스니펫 추출 (상위 2개)"""
        snippets = []

        for doc, score in search_results[:2]:  # 상위 2개만
            snippets.append(
                {
                    "file": doc.metadata.get("source", "알 수 없음"),
                    "code": doc.page_content[:500],  # 코드 길이 제한
                    "relevance": round(score * 100, 1),
                    "explanation": f"관련도 {round(score * 100, 1)}%의 코드 스니펫입니다.",
                }
            )

        return snippets

    async def _generate_solution_suggestion(
        self,
        client,
        issue_title: str,
        issue_body: str,
        related_files: List,
        code_snippets: List,
    ) -> str:
        """AI 해결 제안 생성"""
        try:
            prompt = prompts.get_ai_solution_suggestion_prompt(
                issue_title, issue_body, related_files, code_snippets
            )
            response = client.models.generate_content(
                model=self.llm_model_name, contents=prompt
            )
            suggestion = gemini_service.extract_text_from_response(response)
            return suggestion or "AI 해결 제안을 생성할 수 없습니다."
        except Exception as e:
            logger.warning(f"해결 제안 생성 실패: {e}")
            return "AI 해결 제안 생성 중 오류가 발생했습니다."


# 전역 인스턴스
issue_analyzer = IssueAnalyzer()
