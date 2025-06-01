import time
import logging
from typing import Dict, Optional

from app.core.config import Config
from app.core.utils import extract_repo_name_from_url, get_local_repo_path, format_duration
from app.core.exceptions import RepositoryError, IndexingError, RepositorySizeError

from .github_service import GitHubService
from .git_service import GitService
from .document_loader import DocumentLoader
from .faiss_service import FAISSService
from .embeddings import GeminiAPIEmbeddings

logger = logging.getLogger(__name__)


class RepositoryIndexer:
    """저장소 인덱싱 통합 관리 클래스"""
    
    def __init__(self):
        """인덱서 초기화"""
        self.github_service = GitHubService()
        self.git_service = GitService()
        self.document_loader = DocumentLoader()
        
        # 임베딩 모델 초기화
        self.embeddings = GeminiAPIEmbeddings(
            model_name=Config.DEFAULT_EMBEDDING_MODEL,
            document_task_type="RETRIEVAL_DOCUMENT",
            query_task_type="RETRIEVAL_QUERY",
        )
        
        self.faiss_service = FAISSService(self.embeddings)
    
    def create_indexes_from_repository(self, repo_url: str) -> Dict[str, Optional[object]]:
        """저장소로부터 코드 및 문서 인덱스를 생성합니다.
        
        Args:
            repo_url: GitHub 저장소 URL
            
        Returns:
            생성된 벡터 스토어들의 딕셔너리 {"code": vector_store, "document": vector_store}
            
        Raises:
            RepositoryError: 저장소 관련 오류
            IndexingError: 인덱싱 관련 오류
            RepositorySizeError: 저장소 크기 초과 오류
        """
        start_time = time.time()
        vector_stores = {"code": None, "document": None}
        
        try:
            # 1. 저장소 정보 확인
            logger.info("=== 저장소 정보 확인 중 ===")
            primary_language, _ = self.github_service.get_repository_languages(repo_url)
            
            # 2. 저장소 복제/로드
            logger.info("=== 저장소 복제/로드 중 ===")
            repo = self.git_service.clone_or_load_repository(repo_url)
            repo_name = extract_repo_name_from_url(repo_url)
            local_path = get_local_repo_path(repo_name)
            
            # 3. 코드 인덱싱
            vector_stores["code"] = self._create_code_index(
                local_path, repo_name, primary_language
            )
            
            # 4. 문서 인덱싱
            vector_stores["document"] = self._create_document_index(
                local_path, repo_name
            )
            
            return vector_stores
            
        except (RepositoryError, IndexingError, RepositorySizeError):
            raise
        except Exception as e:
            logger.error(f"인덱싱 중 예상치 못한 오류: {e}", exc_info=True)
            raise IndexingError(f"인덱싱 실패: {e}") from e
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"총 인덱싱 시간: {format_duration(elapsed_time)}")
    
    def _create_code_index(
        self, 
        local_path: str, 
        repo_name: str, 
        primary_language: str
    ) -> Optional[object]:
        """코드 인덱스를 생성합니다.
        
        Args:
            local_path: 로컬 저장소 경로
            repo_name: 저장소 이름
            primary_language: 주 사용 언어
            
        Returns:
            생성된 코드 벡터 스토어 또는 None
        """
        logger.info("=== 코드 인덱싱 시작 ===")
        
        if not self.document_loader.is_supported_language(primary_language):
            logger.warning(f"지원되지 않는 언어: {primary_language}. 코드 인덱싱을 건너뜁니다.")
            return None
        
        # 인덱스 경로 설정
        from app.core.utils import get_faiss_index_path
        index_path = get_faiss_index_path(repo_name, "code")
        
        # 기존 인덱스 확인
        existing_index = self.faiss_service.load_index(index_path, "code")
        if existing_index:
            logger.info("기존 코드 인덱스를 사용합니다.")
            return existing_index
        
        # 코드 파일 로드
        file_extension = self.document_loader.get_code_file_extension(primary_language)
        if not file_extension:
            logger.warning("파일 확장자를 찾을 수 없습니다.")
            return None
        
        documents = self.document_loader.load_documents_from_directory(
            local_path, (file_extension,)
        )
        
        if not documents:
            logger.warning("인덱싱할 코드 파일을 찾지 못했습니다.")
            return None
        
        # 문서 분할
        split_documents = self.document_loader.split_documents_by_language(
            documents, primary_language
        )
        
        # FAISS 인덱스 생성
        return self.faiss_service.create_index_from_documents(
            split_documents, index_path, "code"
        )
    
    def _create_document_index(self, local_path: str, repo_name: str) -> Optional[object]:
        """문서 인덱스를 생성합니다.
        
        Args:
            local_path: 로컬 저장소 경로
            repo_name: 저장소 이름
            
        Returns:
            생성된 문서 벡터 스토어 또는 None
        """
        logger.info("=== 문서 인덱싱 시작 ===")
        
        # 인덱스 경로 설정
        from app.core.utils import get_faiss_index_path
        index_path = get_faiss_index_path(repo_name, "document")
        
        # 기존 인덱스 확인
        existing_index = self.faiss_service.load_index(index_path, "document")
        if existing_index:
            logger.info("기존 문서 인덱스를 사용합니다.")
            return existing_index
        
        # 문서 파일 로드 (최대 깊이 1로 제한)
        documents = self.document_loader.load_documents_from_directory(
            local_path, Config.DOCUMENT_FILE_EXTENSIONS, max_depth=1
        )
        
        if not documents:
            logger.warning("인덱싱할 문서 파일을 찾지 못했습니다.")
            return None
        
        # 문서 분할
        split_documents = self.document_loader.split_documents_as_text(documents)
        
        # FAISS 인덱스 생성
        return self.faiss_service.create_index_from_documents(
            split_documents, index_path, "document"
        )


# 기존 함수들과의 호환성을 위한 래퍼 함수
def create_index_from_repo(repo_url: str, local_repo_path: str, embedding_model_name: str) -> Dict[str, Optional[object]]:
    """기존 코드와의 호환성을 위한 래퍼 함수
    
    Args:
        repo_url: 저장소 URL
        local_repo_path: 로컬 경로 (사용되지 않음, 호환성을 위해 유지)
        embedding_model_name: 임베딩 모델명 (사용되지 않음, Config에서 가져옴)
        
    Returns:
        벡터 스토어 딕셔너리
    """
    indexer = RepositoryIndexer()
    return indexer.create_indexes_from_repository(repo_url)


def load_faiss_index(index_path: str, embeddings, index_type: str):
    """기존 코드와의 호환성을 위한 FAISS 인덱스 로드 함수
    
    Args:
        index_path: 인덱스 경로
        embeddings: 임베딩 객체
        index_type: 인덱스 타입
        
    Returns:
        로드된 FAISS 벡터 스토어
    """
    faiss_service = FAISSService(embeddings)
    return faiss_service.load_index(index_path, index_type) 