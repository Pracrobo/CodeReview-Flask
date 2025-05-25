import os
import logging
from datetime import datetime, timezone

from config import Config
from .indexer import (
    create_index_from_repo,
    load_faiss_index,
)
from .searcher import search_and_rag
from .embeddings import GeminiAPIEmbeddings
from common.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
    RAGError,
    ServiceError,
)

logger = logging.getLogger(__name__)


class RepositoryService:
    """저장소 인덱싱/검색 서비스 클래스"""

    def __init__(self):
        # 저장소 상태 (메모리, 운영 시 DB 권장)
        self.repository_status = {}

    def _get_repo_name_from_url(self, repo_url):
        """URL에서 저장소 이름 추출"""
        return repo_url.split("/")[-1].removesuffix(".git")

    def _get_local_repo_path(self, repo_name):
        """저장소 로컬 경로 반환"""
        return os.path.join(Config.BASE_CLONED_DIR, repo_name)

    def _get_index_path(self, repo_name: str, index_type: str) -> str:
        """주어진 저장소 이름과 인덱스 타입에 대한 FAISS 인덱스 경로를 반환합니다."""
        if index_type == "code":
            return os.path.join(Config.FAISS_INDEX_BASE_DIR, f"{repo_name}_code")
        elif index_type == "document":
            return os.path.join(Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs")
        else:
            raise ValueError(f"알 수 없는 인덱스 타입입니다: {index_type}")

    def prepare_indexing(self, repo_url):
        """인덱싱 준비: 상태 확인, 초기 상태 설정 및 반환."""
        repo_name = self._get_repo_name_from_url(repo_url)

        if repo_name in self.repository_status:
            current_status_info = self.repository_status[repo_name]
            # 이미 진행 중이거나 완료/실패된 경우 해당 상태 반환
            return {**current_status_info, "is_new_request": False}

        # 신규 인덱싱 요청
        current_time_iso = datetime.now(timezone.utc).isoformat()
        initial_status_info = {
            "status": "pending",
            "repo_url": repo_url,
            "repo_name": repo_name,
            "start_time": current_time_iso,
            "last_updated_time": current_time_iso,
            "end_time": None,
            "error": None,
            "error_code": None,
            "code_index_status": "pending",
            "document_index_status": "pending",
            "progress_message": "인덱싱 작업 시작 대기 중...",
            "is_new_request": True,  # API 컨트롤러에서 스레드 시작 여부 판단용
        }
        self.repository_status[repo_name] = initial_status_info
        return initial_status_info

    def perform_indexing(self, repo_url):
        """실제 인덱싱 작업 수행 (백그라운드 실행용)."""
        repo_name = self._get_repo_name_from_url(repo_url)
        local_repo_path = self._get_local_repo_path(repo_name)

        # 이미 self.repository_status[repo_name]은 'pending' 상태로 존재해야 함
        if (
            repo_name not in self.repository_status
            or self.repository_status[repo_name]["status"] != "pending"
        ):
            logger.error(
                f"'{repo_name}'에 대한 인덱싱 작업 수행 오류: 잘못된 초기 상태입니다."
            )
            return

        try:
            logger.info(f"저장소 인덱싱 실제 작업 시작: {repo_url}")
            self.repository_status[repo_name].update(
                {
                    "status": "indexing",
                    "progress_message": "저장소 정보 확인 중...",
                    "last_updated_time": datetime.now(timezone.utc).isoformat(),
                }
            )

            vector_stores = create_index_from_repo(
                repo_url=repo_url,
                local_repo_path=local_repo_path,
                embedding_model_name=Config.DEFAULT_EMBEDDING_MODEL,
            )

            completed_time_iso = datetime.now(timezone.utc).isoformat()
            self.repository_status[repo_name].update(
                {
                    "status": "completed",
                    "end_time": completed_time_iso,
                    "last_updated_time": completed_time_iso,
                    "code_index_status": (
                        "completed"
                        if vector_stores.get("code")
                        else "not_applicable_or_failed"
                    ),
                    "document_index_status": (
                        "completed"
                        if vector_stores.get("document")
                        else "not_applicable_or_failed"
                    ),
                    "progress_message": "인덱싱 완료되었습니다.",
                }
            )
            logger.info(f"저장소 인덱싱 완료: {repo_url}")

        except RepositorySizeError as e:
            error_msg = f"저장소 크기 초과: {str(e)}"
            self._update_error_status(repo_name, error_msg, "REPO_SIZE_EXCEEDED")
            # 이 예외는 API 컨트롤러로 전달되어 처리됨
            raise ServiceError(error_msg, error_code="REPO_SIZE_EXCEEDED") from e
        except (RepositoryError, IndexingError, EmbeddingError) as e:
            error_msg = f"인덱싱 실패: {str(e)}"
            specific_error_code = "INDEXING_FAILED"
            if isinstance(e, RepositoryError):
                specific_error_code = "REPOSITORY_ERROR"
            elif isinstance(e, EmbeddingError):
                specific_error_code = "EMBEDDING_ERROR"
            self._update_error_status(repo_name, error_msg, specific_error_code)
            raise ServiceError(error_msg, error_code=specific_error_code) from e
        except Exception as e:
            error_msg = f"인덱싱 중 예상치 못한 오류: {str(e)}"
            self._update_error_status(repo_name, error_msg, "UNEXPECTED_INDEXING_ERROR")
            logger.error(f"인덱싱 중 예상치 못한 오류 ({repo_url}): {e}", exc_info=True)
            raise ServiceError(error_msg, error_code="UNEXPECTED_INDEXING_ERROR") from e

    def search_repository(self, repo_url, query, search_type="code"):
        """저장소 검색 서비스 로직"""
        repo_name = self._get_repo_name_from_url(repo_url)

        # 인덱스 존재 여부 확인
        if not self._check_index_exists(repo_name, search_type):
            raise ServiceError(
                f"{search_type} 인덱스가 존재하지 않습니다. 먼저 저장소를 인덱싱해주세요."
            )

        try:
            logger.info(f"검색 시작: {repo_url}, 질의: {query}, 타입: {search_type}")

            # 임베딩 모델 초기화
            embeddings = GeminiAPIEmbeddings(
                model_name=Config.DEFAULT_EMBEDDING_MODEL,
                document_task_type="RETRIEVAL_DOCUMENT",
                query_task_type="RETRIEVAL_QUERY",
            )

            # 인덱스 경로 설정
            index_path = self._get_index_path(repo_name, search_type)  # 헬퍼 함수 사용

            vector_store = load_faiss_index(index_path, embeddings, search_type)

            if not vector_store:
                raise ServiceError(f"{search_type} 인덱스 로드에 실패했습니다.")

            vector_stores = {search_type: vector_store}

            # 검색 및 RAG 수행
            rag_response = search_and_rag(
                vector_stores=vector_stores,
                target_index=search_type,
                search_query=query,
                llm_model_name=Config.DEFAULT_LLM_MODEL,
                top_k=Config.DEFAULT_TOP_K,
                similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
            )

            logger.info(f"검색 완료: {repo_url}")

            return {
                "repo_name": repo_name,
                "query": query,
                "search_type": search_type,
                "answer": rag_response,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),  # UTC 시간으로 변경
            }

        except RAGError as e:
            error_msg = f"검색 실패: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"검색 중 예상치 못한 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)  # exc_info=True 추가
            raise ServiceError(error_msg, error_code="UNEXPECTED_SEARCH_ERROR") from e

    def get_repository_status(self, repo_name):
        """저장소 상태 조회. repo_name은 URL에서 추출된 순수 이름이어야 함."""

        if repo_name not in self.repository_status:
            # 메모리에 상태 정보가 없는 경우 (예: 서버 재시작)
            # 디스크의 인덱스 파일 존재 여부로 간이 상태 추론
            code_index_path = self._get_index_path(repo_name, "code")  # 헬퍼 함수 사용
            doc_index_path = self._get_index_path(
                repo_name, "document"
            )  # 헬퍼 함수 사용

            code_exists = os.path.exists(code_index_path)
            doc_exists = os.path.exists(doc_index_path)

            if code_exists or doc_exists:
                # 인덱스 파일은 존재하나, 상세 진행 기록은 없는 상태
                return {
                    "status": "completed",
                    "repo_name": repo_name,
                    "code_index_status": "completed" if code_exists else "not_found",
                    "document_index_status": "completed" if doc_exists else "not_found",
                    "progress_message": "인덱스 파일은 존재하나, 상세 진행 기록은 없습니다 (서버 재시작 가능성).",
                    "last_updated_time": datetime.now(timezone.utc).isoformat(),
                }
            else:
                # 메모리 및 디스크 모두에 정보 없음
                return {
                    "status": "not_indexed",
                    "repo_name": repo_name,
                    "progress_message": "인덱싱된 정보가 없습니다.",
                }

        # 메모리에 상태가 있는 경우, 최신 시간으로 업데이트 후 반환
        self.repository_status[repo_name]["last_updated_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        return self.repository_status[repo_name]

    def _update_error_status(self, repo_name, error_msg, error_code=None):
        """인덱싱 오류 발생 시 상태를 'failed'로 업데이트합니다."""
        if repo_name in self.repository_status:
            current_time_iso = datetime.now(timezone.utc).isoformat()
            self.repository_status[repo_name].update(
                {
                    "status": "failed",
                    "end_time": current_time_iso,
                    "last_updated_time": current_time_iso,
                    "error": error_msg,
                    "error_code": error_code,
                    "progress_message": f"인덱싱 실패: {error_msg}",
                    # 실패 시 각 인덱스 상태도 'failed'로 명시
                    "code_index_status": (
                        "failed"
                        if self.repository_status[repo_name].get("code_index_status")
                        != "completed"
                        else "completed"
                    ),
                    "document_index_status": (
                        "failed"
                        if self.repository_status[repo_name].get(
                            "document_index_status"
                        )
                        != "completed"
                        else "completed"
                    ),
                }
            )

    def _check_index_exists(self, repo_name, index_type):
        """주어진 저장소 이름과 인덱스 타입에 대한 인덱스 존재 여부를 확인합니다."""
        index_path = self._get_index_path(repo_name, index_type)
        return os.path.exists(index_path)

    def _check_index_exists(self, repo_name, index_type):
        """인덱스 파일 존재 여부 확인"""
        index_path = self._get_index_path(repo_name, index_type)  # 헬퍼 함수 사용
        return os.path.exists(index_path)
