import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any

from app.core.config import Config
from app.core.utils import (
    extract_repo_name_from_url, 
    get_local_repo_path, 
    get_faiss_index_path,
    check_index_exists
)

logger = logging.getLogger(__name__)

class StatusService:
    """저장소 인덱싱 상태 관리 서비스 (싱글톤)"""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """상태 서비스 초기화 (한 번만 실행)"""
        if not hasattr(self, '_initialized'): 
            self.repository_status: Dict[str, Dict[str, Any]] = {}
            self._status_data_lock = threading.Lock()
            self._initialized = True
            logger.info("StatusService가 싱글톤으로 초기화되었습니다.")

    def get_repository_status_data(self, repo_name: str) -> Dict[str, Any]:
        """특정 저장소의 현재 상태 데이터를 반환합니다.
        
        Args:
            repo_name: 저장소 이름
            
        Returns:
            상태 데이터 딕셔너리
        """
        with self._status_data_lock:
            if repo_name not in self.repository_status:
                return self._check_existing_indexes(repo_name)
            
            return self.repository_status[repo_name].copy()

    def _check_existing_indexes(self, repo_name: str) -> Dict[str, Any]:
        """기존 인덱스 파일 존재 여부를 확인합니다.
        
        Args:
            repo_name: 저장소 이름
            
        Returns:
            상태 데이터 딕셔너리
        """
        code_exists = check_index_exists(repo_name, "code")
        doc_exists = check_index_exists(repo_name, "document")

        if code_exists or doc_exists:
            return {
                "status": "completed",
                "repo_name": repo_name,
                "code_index_status": "completed" if code_exists else "not_found",
                "document_index_status": "completed" if doc_exists else "not_found",
                "progress_message": "인덱스 파일이 존재합니다 (상세 기록 없음)",
                "last_updated_time": self._get_current_timestamp(),
            }
        else:
            return {
                "status": "not_indexed",
                "repo_name": repo_name,
                "progress_message": "인덱싱된 정보가 없습니다.",
            }

    def update_repository_status(self, repo_name: str, status_info: Dict[str, Any]) -> None:
        """저장소 상태 정보를 업데이트합니다.
        
        Args:
            repo_name: 저장소 이름
            status_info: 업데이트할 상태 정보
        """
        with self._status_data_lock:
            status_info["last_updated_time"] = self._get_current_timestamp()
            
            if repo_name in self.repository_status:
                self.repository_status[repo_name].update(status_info)
            else:
                self.repository_status[repo_name] = status_info
                
            logger.debug(f"상태 업데이트 완료 - '{repo_name}': {status_info.get('status')}")

    def init_indexing_status(self, repo_url: str) -> Dict[str, Any]:
        """인덱싱 시작 시 초기 상태를 설정합니다.
        
        Args:
            repo_url: 저장소 URL
            
        Returns:
            초기 상태 정보 딕셔너리
        """
        repo_name = extract_repo_name_from_url(repo_url)
        
        with self._status_data_lock:
            if repo_name in self.repository_status:
                current_data = self.repository_status[repo_name].copy()
                current_data["is_new_request"] = False
                return current_data
            
            current_time = self._get_current_timestamp()
            initial_status = {
                "status": "pending",
                "repo_url": repo_url,
                "repo_name": repo_name,
                "start_time": current_time,
                "last_updated_time": current_time,
                "end_time": None,
                "error": None,
                "error_code": None,
                "code_index_status": "pending",
                "document_index_status": "pending",
                "progress_message": "인덱싱 작업 시작 대기 중...",
                "is_new_request": True,
            }
            
            self.repository_status[repo_name] = initial_status
            logger.info(f"초기 인덱싱 상태 설정 완료: '{repo_name}'")
            
            return initial_status.copy()

    def set_error_status(self, repo_name: str, error_msg: str, error_code: str = None) -> None:
        """인덱싱 오류 발생 시 상태를 'failed'로 업데이트합니다.
        
        Args:
            repo_name: 저장소 이름
            error_msg: 오류 메시지
            error_code: 오류 코드 (선택사항)
        """
        with self._status_data_lock:
            if repo_name not in self.repository_status:
                logger.warning(f"존재하지 않는 저장소에 오류 상태 설정 시도: '{repo_name}'")
                return

            current_time = self._get_current_timestamp()
            current_status = self.repository_status[repo_name]
            
            error_update = {
                "status": "failed",
                "end_time": current_time,
                "last_updated_time": current_time,
                "error": error_msg,
                "error_code": error_code,
                "progress_message": f"인덱싱 실패: {error_msg}",
                "code_index_status": self._get_failed_status(current_status.get("code_index_status")),
                "document_index_status": self._get_failed_status(current_status.get("document_index_status")),
            }
            
            self.repository_status[repo_name].update(error_update)
            logger.error(f"오류 상태 설정 완료 - '{repo_name}': {error_msg}")

    def _get_failed_status(self, current_status: str) -> str:
        """현재 상태에 따른 실패 상태를 반환합니다.
        
        Args:
            current_status: 현재 상태
            
        Returns:
            실패 상태 문자열
        """
        return "completed" if current_status == "completed" else "failed"

    def _get_current_timestamp(self) -> str:
        """현재 시간의 ISO 형식 문자열을 반환합니다.
        
        Returns:
            ISO 형식 타임스탬프
        """
        return datetime.now(timezone.utc).isoformat()

    # 기존 코드와의 호환성을 위한 메서드들
    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """URL에서 저장소 이름을 추출합니다 (호환성용).
        
        Args:
            repo_url: 저장소 URL
            
        Returns:
            저장소 이름
        """
        return extract_repo_name_from_url(repo_url)

    def get_local_repo_path(self, repo_name: str) -> str:
        """저장소의 로컬 경로를 반환합니다 (호환성용).
        
        Args:
            repo_name: 저장소 이름
            
        Returns:
            로컬 저장소 경로
        """
        return get_local_repo_path(repo_name)

    def get_index_path(self, repo_name: str, index_type: str) -> str:
        """FAISS 인덱스 경로를 반환합니다 (호환성용).
        
        Args:
            repo_name: 저장소 이름
            index_type: 인덱스 타입
            
        Returns:
            인덱스 경로
        """
        return get_faiss_index_path(repo_name, index_type)

    def check_index_exists(self, repo_name: str, index_type: str) -> bool:
        """인덱스 존재 여부를 확인합니다 (호환성용).
        
        Args:
            repo_name: 저장소 이름
            index_type: 인덱스 타입
            
        Returns:
            인덱스 존재 여부
        """
        return check_index_exists(repo_name, index_type) 