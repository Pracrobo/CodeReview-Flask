import logging
from datetime import datetime, timezone
from typing import Dict, Any

from app.core.config import Config
from app.core.utils import extract_repo_name_from_url, get_faiss_index_path
from app.core.exceptions import ServiceError, RAGError

from .status_service import StatusService
from .embeddings import GeminiAPIEmbeddings
from .faiss_service import FAISSService
from .searcher import search_and_rag

logger = logging.getLogger(__name__)


class SearchService:
    """저장소 검색 처리 서비스"""

    def __init__(self, status_service: StatusService):
        """검색 서비스 초기화
        
        Args:
            status_service: 상태 관리 서비스
        """
        self.status_service = status_service
        
        # 임베딩 모델 초기화
        self.embeddings = GeminiAPIEmbeddings(
            model_name=Config.DEFAULT_EMBEDDING_MODEL,
            document_task_type="RETRIEVAL_DOCUMENT",
            query_task_type="RETRIEVAL_QUERY",
        )
        
        self.faiss_service = FAISSService(self.embeddings)
        logger.info("SearchService가 초기화되었습니다.")

    def search_repository(self, repo_url: str, query: str, search_type: str = "code") -> Dict[str, Any]:
        """저장소에서 검색을 수행합니다.
        
        Args:
            repo_url: GitHub 저장소 URL
            query: 검색 질의
            search_type: 검색 타입 ("code" 또는 "document")
            
        Returns:
            검색 결과 딕셔너리
            
        Raises:
            ServiceError: 검색 실패시
        """
        repo_name = extract_repo_name_from_url(repo_url)
        logger.info(f"검색 요청 - 저장소: '{repo_name}', 타입: {search_type}, 질의: '{query}'")

        # 인덱스 존재 여부 확인
        if not self.status_service.check_index_exists(repo_name, search_type):
            error_msg = (
                f"'{repo_name}' 저장소의 {search_type} 인덱스가 존재하지 않습니다. "
                f"먼저 저장소를 인덱싱해주세요."
            )
            logger.warning(error_msg)
            raise ServiceError(error_msg, error_code="INDEX_NOT_FOUND")

        try:
            # FAISS 인덱스 로드
            vector_store = self._load_vector_store(repo_name, search_type)
            
            # RAG 검색 수행
            search_result = self._perform_rag_search(
                vector_store, query, search_type, repo_name
            )
            
            logger.info(f"검색 완료 - 저장소: '{repo_name}'")
            return search_result

        except RAGError as e:
            logger.error(f"RAG 검색 오류 - '{repo_name}': {e}")
            raise ServiceError(f"검색 실패: {str(e)}", error_code="RAG_ERROR") from e
        except ServiceError:
            raise
        except Exception as e:
            logger.error(f"예상치 못한 검색 오류 - '{repo_name}': {e}", exc_info=True)
            raise ServiceError(
                f"검색 중 예상치 못한 오류 발생: {str(e)}", 
                error_code="UNEXPECTED_SEARCH_ERROR"
            ) from e

    def _load_vector_store(self, repo_name: str, search_type: str):
        """벡터 스토어를 로드합니다.
        
        Args:
            repo_name: 저장소 이름
            search_type: 검색 타입
            
        Returns:
            로드된 벡터 스토어
            
        Raises:
            ServiceError: 로드 실패시
        """
        index_path = get_faiss_index_path(repo_name, search_type)
        logger.debug(f"FAISS 인덱스 로드 중: {index_path}")
        
        vector_store = self.faiss_service.load_index(index_path, search_type)
        
        if not vector_store:
            error_msg = f"'{repo_name}' 저장소의 {search_type} 인덱스 로드에 실패했습니다."
            logger.error(error_msg)
            raise ServiceError(error_msg, error_code="INDEX_LOAD_FAILED")
        
        return vector_store

    def _perform_rag_search(
        self, 
        vector_store, 
        query: str, 
        search_type: str, 
        repo_name: str
    ) -> Dict[str, Any]:
        """RAG 검색을 수행합니다.
        
        Args:
            vector_store: 벡터 스토어
            query: 검색 질의
            search_type: 검색 타입
            repo_name: 저장소 이름
            
        Returns:
            검색 결과 딕셔너리
        """
        logger.debug(f"RAG 검색 수행 - '{repo_name}', 질의: '{query}'")
        
        vector_stores = {search_type: vector_store}
        
        rag_response = search_and_rag(
            vector_stores=vector_stores,
            target_index=search_type,
            search_query=query,
            llm_model_name=Config.DEFAULT_LLM_MODEL,
            top_k=Config.DEFAULT_TOP_K,
            similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
        )
        
        return {
            "repo_name": repo_name,
            "query": query,
            "search_type": search_type,
            "answer": rag_response,
            "result_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_search_statistics(self, repo_name: str, search_type: str) -> Dict[str, Any]:
        """검색 통계 정보를 반환합니다.
        
        Args:
            repo_name: 저장소 이름
            search_type: 검색 타입
            
        Returns:
            통계 정보 딕셔너리
        """
        try:
            if not self.status_service.check_index_exists(repo_name, search_type):
                return {"error": "인덱스가 존재하지 않습니다."}
            
            vector_store = self._load_vector_store(repo_name, search_type)
            stats = self.faiss_service.get_index_stats(vector_store)
            
            return {
                "repo_name": repo_name,
                "search_type": search_type,
                "index_stats": stats,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logger.warning(f"통계 수집 실패 - '{repo_name}': {e}")
            return {"error": f"통계 수집 실패: {str(e)}"}

    # 기존 코드와의 호환성을 위한 메서드
    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """URL에서 저장소 이름을 추출합니다 (호환성용).
        
        Args:
            repo_url: 저장소 URL
            
        Returns:
            저장소 이름
        """
        return extract_repo_name_from_url(repo_url) 