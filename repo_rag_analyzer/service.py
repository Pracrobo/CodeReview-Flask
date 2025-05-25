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
    ServiceError,  # 추가
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

    def index_repository(self, repo_url):
        """저장소 인덱싱 서비스 로직. 비동기 시작 또는 동기 완료 결과를 반환."""
        repo_name = self._get_repo_name_from_url(repo_url)
        local_repo_path = self._get_local_repo_path(repo_name)

        # 중복 인덱싱 방지 로직
        if repo_name in self.repository_status:
            current_status = self.repository_status[repo_name].get("status")
            if current_status == "indexing" or current_status == "pending":
                logger.info(
                    f"저장소 '{repo_name}'은 이미 인덱싱 중이거나 대기 중입니다."
                )
                # 현재 진행 상태를 반환하여 클라이언트가 혼동하지 않도록 함
                return self.repository_status[repo_name]
            elif current_status == "completed":
                logger.info(
                    f"저장소 '{repo_name}'은 이미 성공적으로 인덱싱되었습니다. 재인덱싱을 원하시면 기존 인덱스를 삭제 후 시도해주세요."
                )
                # 이미 완료된 상태 정보 반환
                return self.repository_status[repo_name]
            elif current_status == "failed":
                logger.info(
                    f"저장소 '{repo_name}'은 이전에 인덱싱에 실패했습니다. 재시도합니다."
                )
                # 실패한 경우, 새로 인덱싱 시도 (아래 로직으로 계속 진행)

        # 상태 초기화 또는 업데이트
        current_time_iso = datetime.now(timezone.utc).isoformat()
        self.repository_status[repo_name] = {
            "status": "indexing",  # 또는 "pending"으로 시작 후 실제 작업 시 "indexing"으로 변경
            "repo_url": repo_url,
            "repo_name": repo_name,
            "start_time": current_time_iso,
            "last_updated_time": current_time_iso,
            "end_time": None,
            "error": None,
            "error_code": None,
            "code_index_status": "pending",  # "pending", "in_progress", "completed", "failed"
            "document_index_status": "pending",
            "progress_message": "인덱싱 작업 시작 대기 중...",
        }

        try:
            logger.info(f"저장소 인덱싱 시작 요청: {repo_url}")

            # 실제 인덱싱 작업은 백그라운드에서 실행된다고 가정하고, 여기서는 작업 시작을 알리는 응답을 즉시 반환.
            # 실제 구현에서는 Celery, RQ 같은 작업 큐 사용 또는 ThreadPoolExecutor 등을 고려.
            # 여기서는 동기적으로 실행하지만, API 응답은 비동기 시작처럼 처리.

            # ---- 백그라운드 작업 시뮬레이션 시작 ----
            self.repository_status[repo_name][
                "progress_message"
            ] = "저장소 정보 확인 중..."
            self.repository_status[repo_name]["last_updated_time"] = datetime.now(
                timezone.utc
            ).isoformat()

            vector_stores = create_index_from_repo(
                repo_url=repo_url,
                local_repo_path=local_repo_path,
                embedding_model_name=Config.DEFAULT_EMBEDDING_MODEL,
            )
            # ---- 백그라운드 작업 시뮬레이션 종료 ----

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
            return self.repository_status[repo_name]  # 완료된 상태 반환

        except RepositorySizeError as e:
            error_msg = f"저장소 크기 초과: {str(e)}"
            self._update_error_status(repo_name, error_msg, "REPO_SIZE_EXCEEDED")
            raise ServiceError(error_msg, error_code="REPO_SIZE_EXCEEDED") from e
        except (RepositoryError, IndexingError, EmbeddingError) as e:
            error_msg = f"인덱싱 실패: {str(e)}"
            # 구체적인 오류 유형에 따라 error_code 세분화 가능
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
            if search_type == "code":
                index_path = os.path.join(
                    Config.FAISS_INDEX_BASE_DIR, f"{repo_name}_code"
                )
            else:
                index_path = os.path.join(
                    Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs"
                )

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
                "timestamp": datetime.now().isoformat(),  # UTC 시간으로 변경하려면 datetime.now(timezone.utc).isoformat()
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
        """저장소 상태 조회 서비스 로직. repo_name은 URL에서 추출된 순수 이름이어야 함."""
        # repo_name 정규화 (예: '.git' 제거, 소문자 변환 등) - 이미 _get_repo_name_from_url 에서 처리됨

        if repo_name not in self.repository_status:
            # 디스크에서 인덱스 존재 여부만으로 간단히 상태를 추론 (더 정교한 상태 관리 필요 시 DB 사용)
            code_index_path = os.path.join(
                Config.FAISS_INDEX_BASE_DIR, f"{repo_name}_code"
            )
            doc_index_path = os.path.join(
                Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs"
            )
            code_exists = os.path.exists(code_index_path)
            doc_exists = os.path.exists(doc_index_path)

            if code_exists or doc_exists:
                # 인덱스 파일은 존재하나, 메모리 상태가 없는 경우 (서버 재시작 등)
                # 이 경우, 마지막 성공 상태로 간주하거나, 'unknown' 상태로 처리 가능
                return {
                    "status": "completed",  # 또는 "unknown_completed"
                    "repo_name": repo_name,
                    "code_index_status": "completed" if code_exists else "not_found",
                    "document_index_status": "completed" if doc_exists else "not_found",
                    "progress_message": "인덱스 파일이 존재하지만, 상세 진행 기록은 없습니다.",
                    "last_updated_time": datetime.now(
                        timezone.utc
                    ).isoformat(),  # 현재 시간으로 설정
                }
            else:
                # 메모리에도 없고 디스크에도 파일이 없는 경우
                # raise ServiceError(f"저장소 '{repo_name}'에 대한 인덱싱 정보를 찾을 수 없습니다.", error_code="NOT_FOUND")
                # API 컨트롤러에서 NOT_FOUND 응답을 하도록 상태 반환
                return {
                    "status": "not_indexed",
                    "repo_name": repo_name,
                    "progress_message": "인덱싱된 정보가 없습니다.",
                }

        # 메모리에 상태가 있는 경우 그대로 반환
        self.repository_status[repo_name]["last_updated_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        return self.repository_status[repo_name]

    def _update_error_status(self, repo_name, error_msg, error_code=None):
        """오류 상태를 업데이트합니다."""
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
        """인덱스 파일 존재 여부 확인"""
        if index_type == "code":
            index_path = os.path.join(Config.FAISS_INDEX_BASE_DIR, f"{repo_name}_code")
        else:
            index_path = os.path.join(Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs")

        return os.path.exists(index_path)
