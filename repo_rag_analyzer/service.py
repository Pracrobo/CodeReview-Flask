import os
import logging
from typing import Dict, Any
from datetime import datetime

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
    """저장소 인덱싱 및 검색 서비스를 제공하는 클래스"""

    def __init__(self):
        # 저장소 상태를 메모리에 저장 (실제 운영에서는 데이터베이스 사용 권장)
        self.repository_status: Dict[str, Dict[str, Any]] = {}

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """URL에서 저장소 이름을 추출합니다."""
        return repo_url.split("/")[-1].removesuffix(".git")

    def _get_local_repo_path(self, repo_name: str) -> str:
        """저장소의 로컬 경로를 반환합니다."""
        return os.path.join(Config.BASE_CLONED_DIR, repo_name)

    def index_repository(self, repo_url: str) -> Dict[str, Any]:
        """저장소를 인덱싱합니다."""
        repo_name = self._get_repo_name_from_url(repo_url)
        local_repo_path = self._get_local_repo_path(repo_name)

        # 상태 초기화
        self.repository_status[repo_name] = {
            "status": "indexing",
            "repo_url": repo_url,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "error": None,
            "code_index": False,
            "document_index": False,
        }

        try:
            logger.info(f"저장소 인덱싱 시작: {repo_url}")

            # 기존 인덱싱 함수 호출
            vector_stores = create_index_from_repo(
                repo_url=repo_url,
                local_repo_path=local_repo_path,
                embedding_model_name=Config.DEFAULT_EMBEDDING_MODEL,
            )

            # 인덱싱 결과 상태 업데이트
            self.repository_status[repo_name].update(
                {
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "code_index": vector_stores.get("code") is not None,
                    "document_index": vector_stores.get("document") is not None,
                }
            )

            logger.info(f"저장소 인덱싱 완료: {repo_url}")

            return {
                "repo_name": repo_name,
                "repo_url": repo_url,
                "code_indexed": vector_stores.get("code") is not None,
                "document_indexed": vector_stores.get("document") is not None,
                "status": "completed",
            }

        except RepositorySizeError as e:
            error_msg = f"저장소 크기 초과: {str(e)}"
            self._update_error_status(repo_name, error_msg)
            raise ServiceError(error_msg) from e
        except (RepositoryError, IndexingError, EmbeddingError) as e:
            error_msg = f"인덱싱 실패: {str(e)}"
            self._update_error_status(repo_name, error_msg)
            raise ServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"예상치 못한 오류: {str(e)}"
            self._update_error_status(repo_name, error_msg)
            raise ServiceError(error_msg) from e

    def search_repository(
        self, repo_url: str, query: str, search_type: str = "code"
    ) -> Dict[str, Any]:
        """저장소에서 검색을 수행합니다."""
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
                "timestamp": datetime.now().isoformat(),
            }

        except RAGError as e:
            error_msg = f"검색 실패: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"검색 중 예상치 못한 오류: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg) from e

    def get_repository_status(self, repo_name: str) -> Dict[str, Any]:
        """저장소 상태를 반환합니다."""
        if repo_name not in self.repository_status:
            # 디스크에서 인덱스 존재 여부 확인
            code_exists = self._check_index_exists(repo_name, "code")
            doc_exists = self._check_index_exists(repo_name, "document")

            if code_exists or doc_exists:
                return {
                    "status": "completed",
                    "code_index": code_exists,
                    "document_index": doc_exists,
                    "repo_name": repo_name,
                }
            else:
                return {"status": "not_indexed", "repo_name": repo_name}

        return self.repository_status[repo_name]

    def _update_error_status(self, repo_name: str, error_msg: str):
        """오류 상태를 업데이트합니다."""
        if repo_name in self.repository_status:
            self.repository_status[repo_name].update(
                {
                    "status": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error": error_msg,
                }
            )

    def _check_index_exists(self, repo_name: str, index_type: str) -> bool:
        """인덱스 파일 존재 여부를 확인합니다."""
        if index_type == "code":
            index_path = os.path.join(Config.FAISS_INDEX_BASE_DIR, f"{repo_name}_code")
        else:
            index_path = os.path.join(Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs")

        return os.path.exists(index_path)
