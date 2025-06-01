import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any

from app.core.config import Config
from app.core.utils import extract_repo_name_from_url, get_local_repo_path
from app.core.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
    ServiceError,
)

from .status_service import StatusService
from .indexer import RepositoryIndexer

logger = logging.getLogger(__name__)


class IndexingService:
    """저장소 인덱싱 처리 서비스"""

    def __init__(self, status_service: StatusService):
        """인덱싱 서비스 초기화
        
        Args:
            status_service: 상태 관리 서비스
        """
        self.status_service = status_service
        self.indexer = RepositoryIndexer()
        logger.info("IndexingService가 초기화되었습니다.")

    def prepare_and_start_indexing(self, repo_url: str) -> Dict[str, Any]:
        """인덱싱 준비 및 백그라운드 작업을 시작합니다.
        
        Args:
            repo_url: GitHub 저장소 URL
            
        Returns:
            초기 상태 결과 딕셔너리
        """
        repo_name = extract_repo_name_from_url(repo_url)
        logger.info(f"인덱싱 준비 시작: '{repo_name}' (URL: {repo_url})")

        initial_status_result = self.status_service.init_indexing_status(repo_url)
        current_status = initial_status_result.get("status")
        is_new_request = initial_status_result.get("is_new_request", False)

        if self._should_start_new_indexing(current_status, is_new_request):
            logger.info(f"새로운 인덱싱 요청: '{repo_name}'. 백그라운드 스레드를 시작합니다.")
            self._start_indexing_thread(repo_url, repo_name)
            return initial_status_result
        elif current_status == "completed":
            logger.info(f"저장소 '{repo_name}'은 이미 인덱싱되었습니다.")
            return initial_status_result
        else:
            logger.info(f"저장소 '{repo_name}'의 인덱싱이 이미 진행 중이거나 대기 중입니다.")
            return initial_status_result

    def _should_start_new_indexing(self, current_status: str, is_new_request: bool) -> bool:
        """새로운 인덱싱을 시작해야 하는지 판단합니다.
        
        Args:
            current_status: 현재 상태
            is_new_request: 새로운 요청 여부
            
        Returns:
            새로운 인덱싱 시작 여부
        """
        return (current_status in ["pending", "indexing"]) and is_new_request

    def _start_indexing_thread(self, repo_url: str, repo_name: str) -> None:
        """인덱싱 백그라운드 스레드를 시작합니다.
        
        Args:
            repo_url: 저장소 URL
            repo_name: 저장소 이름
        """
        # 상태를 'indexing'으로 즉시 변경
        self.status_service.update_repository_status(
            repo_name, 
            {
                "status": "indexing", 
                "progress_message": "인덱싱 작업 초기화 중..."
            }
        )
        
        # 백그라운드 스레드 시작
        thread = threading.Thread(
            target=self._perform_actual_indexing, 
            args=(repo_url, repo_name),
            name=f"IndexingThread-{repo_name}"
        )
        thread.daemon = True
        thread.start()

    def _perform_actual_indexing(self, repo_url: str, repo_name: str) -> None:
        """실제 인덱싱 작업을 수행합니다 (백그라운드 스레드에서 실행).
        
        Args:
            repo_url: 저장소 URL
            repo_name: 저장소 이름
        """
        local_repo_path = get_local_repo_path(repo_name)
        logger.info(f"백그라운드 인덱싱 시작: '{repo_name}' (경로: {local_repo_path})")

        try:
            # 진행 상황 업데이트
            self._update_progress(repo_name, "저장소 정보 확인 및 복제/로드 중...")

            # 실제 인덱싱 수행
            vector_stores = self.indexer.create_indexes_from_repository(repo_url)

            # 완료 상태 업데이트
            self._set_completion_status(repo_name, vector_stores)
            logger.info(f"인덱싱 성공적으로 완료: '{repo_name}'")

        except RepositorySizeError as e:
            logger.error(f"저장소 크기 초과 오류 - '{repo_name}': {e}")
            self.status_service.set_error_status(repo_name, str(e), "REPO_SIZE_EXCEEDED")
        except (RepositoryError, IndexingError, EmbeddingError) as e:
            logger.error(f"인덱싱 오류 - '{repo_name}': {e}")
            error_code = self._get_error_code(e)
            self.status_service.set_error_status(repo_name, str(e), error_code)
        except Exception as e:
            logger.error(f"예상치 못한 인덱싱 오류 - '{repo_name}': {e}", exc_info=True)
            self.status_service.set_error_status(
                repo_name, str(e), "UNEXPECTED_INDEXING_ERROR"
            )

    def _update_progress(self, repo_name: str, message: str) -> None:
        """진행 상황을 업데이트합니다.
        
        Args:
            repo_name: 저장소 이름
            message: 진행 메시지
        """
        self.status_service.update_repository_status(
            repo_name, {"progress_message": message}
        )

    def _set_completion_status(self, repo_name: str, vector_stores: Dict[str, Any]) -> None:
        """완료 상태를 설정합니다.
        
        Args:
            repo_name: 저장소 이름
            vector_stores: 생성된 벡터 스토어들
        """
        completed_time = datetime.now(timezone.utc).isoformat()
        
        final_status_update = {
            "status": "completed",
            "end_time": completed_time,
            "code_index_status": self._get_index_status(vector_stores.get("code")),
            "document_index_status": self._get_index_status(vector_stores.get("document")),
            "progress_message": "인덱싱이 완료되었습니다.",
        }
        
        self.status_service.update_repository_status(repo_name, final_status_update)

    def _get_index_status(self, vector_store) -> str:
        """벡터 스토어 상태에 따른 인덱스 상태를 반환합니다.
        
        Args:
            vector_store: 벡터 스토어 객체
            
        Returns:
            인덱스 상태 문자열
        """
        return "completed" if vector_store else "not_applicable_or_failed"

    def _get_error_code(self, exception: Exception) -> str:
        """예외 타입에 따른 오류 코드를 반환합니다.
        
        Args:
            exception: 발생한 예외
            
        Returns:
            오류 코드 문자열
        """
        if isinstance(exception, RepositoryError):
            return "REPOSITORY_ERROR"
        elif isinstance(exception, EmbeddingError):
            return "EMBEDDING_ERROR"
        elif isinstance(exception, IndexingError):
            return "INDEXING_FAILED"
        else:
            return "UNKNOWN_ERROR"

    # 기존 코드와의 호환성을 위한 메서드들
    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """URL에서 저장소 이름을 추출합니다 (호환성용).
        
        Args:
            repo_url: 저장소 URL
            
        Returns:
            저장소 이름
        """
        return extract_repo_name_from_url(repo_url)

    def _get_local_repo_path(self, repo_name: str) -> str:
        """저장소의 로컬 경로를 반환합니다 (호환성용).
        
        Args:
            repo_name: 저장소 이름
            
        Returns:
            로컬 저장소 경로
        """
        return get_local_repo_path(repo_name) 