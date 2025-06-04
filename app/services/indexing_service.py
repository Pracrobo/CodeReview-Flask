import logging
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.core.utils import (
    extract_repo_name_from_url,
    get_local_repo_path,
    get_repo_owner_and_name,
)
from app.core.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
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
        self.callback_urls = {}  # repo_name -> callback_url 매핑
        self.user_ids = {}  # repo_name -> user_id 매핑
        logger.info("IndexingService가 초기화되었습니다.")

    def prepare_and_start_indexing(
        self,
        repo_url: str,
        callback_url: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """인덱싱 준비 및 백그라운드 작업을 시작합니다.

        Args:
            repo_url: GitHub 저장소 URL
            callback_url: 분석 완료 시 콜백할 URL (선택사항)
            user_id: 사용자 ID (선택사항)

        Returns:
            초기 상태 결과 딕셔너리
        """
        repo_name = extract_repo_name_from_url(repo_url)
        logger.info(f"인덱싱 준비 시작: '{repo_name}' (URL: {repo_url})")

        # 콜백 URL 저장 (owner/repo 형식으로 키 생성)
        if callback_url:
            try:
                owner, repo = get_repo_owner_and_name(repo_url)
                full_repo_name = f"{owner}/{repo}"
                self.callback_urls[full_repo_name] = callback_url
                self.user_ids[full_repo_name] = user_id
                logger.info(f"콜백 URL 등록: {full_repo_name} -> {callback_url}")
            except Exception as e:
                logger.warning(f"콜백 URL 등록 실패: {e}")

        initial_status_result = self.status_service.init_indexing_status(repo_url)
        current_status = initial_status_result.get("status")
        is_new_request = initial_status_result.get("is_new_request", False)

        if self._should_start_new_indexing(current_status, is_new_request):
            logger.info(
                f"새로운 인덱싱 요청: '{repo_name}'. 백그라운드 스레드를 시작합니다."
            )
            self._start_indexing_thread(repo_url, repo_name)
            return initial_status_result
        elif current_status == "completed":
            logger.info(f"저장소 '{repo_name}'은 이미 인덱싱되었습니다.")
            return initial_status_result
        else:
            logger.info(
                f"저장소 '{repo_name}'의 인덱싱이 이미 진행 중이거나 대기 중입니다."
            )
            return initial_status_result

    def _should_start_new_indexing(
        self, current_status: str, is_new_request: bool
    ) -> bool:
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
            {"status": "indexing", "progress_message": "인덱싱 작업 초기화 중..."},
        )

        # 백그라운드 스레드 시작
        thread = threading.Thread(
            target=self._perform_actual_indexing,
            args=(repo_url, repo_name),
            name=f"IndexingThread-{repo_name}",
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
            # 1. 저장소 정보 확인 단계
            self._update_progress(repo_name, "저장소 정보 확인 및 복제/로드 중...")

            # 2. 실제 인덱싱 수행 (단계별 진행 상황 추적)
            vector_stores = self._perform_indexing_with_progress(repo_url, repo_name)

            # 3. 완료 상태 업데이트
            self._set_completion_status(repo_name, vector_stores)
            logger.info(f"인덱싱 성공적으로 완료: '{repo_name}'")

            # Express로 완료 콜백 전송 (owner/repo 형식으로)
            self._send_completion_callback(repo_url, "completed")

        except RepositorySizeError as e:
            logger.error(f"저장소 크기 초과 오류 - '{repo_name}': {e}")
            self.status_service.set_error_status(
                repo_name, str(e), "REPO_SIZE_EXCEEDED"
            )
            self._send_completion_callback(repo_url, "failed", str(e))
        except (RepositoryError, IndexingError, EmbeddingError) as e:
            logger.error(f"인덱싱 오류 - '{repo_name}': {e}")
            error_code = self._get_error_code(e)
            self.status_service.set_error_status(repo_name, str(e), error_code)
            self._send_completion_callback(repo_url, "failed", str(e))
        except Exception as e:
            logger.error(f"예상치 못한 인덱싱 오류 - '{repo_name}': {e}", exc_info=True)
            self.status_service.set_error_status(
                repo_name, str(e), "UNEXPECTED_INDEXING_ERROR"
            )
            self._send_completion_callback(repo_url, "failed", str(e))

    def _perform_indexing_with_progress(
        self, repo_url: str, repo_name: str
    ) -> Dict[str, Any]:
        """진행 상황을 추적하면서 인덱싱을 수행합니다.

        Args:
            repo_url: 저장소 URL
            repo_name: 저장소 이름

        Returns:
            생성된 벡터 스토어들
        """
        # 저장소 정보 확인 및 복제
        self._update_progress(repo_name, "저장소 정보 확인 중...")

        # 실제 인덱서 호출 전에 각 단계별 콜백 설정
        def progress_callback(stage: str, message: str):
            """인덱싱 진행 상황 콜백"""
            if "저장소 복제" in message or "저장소를 복제" in message:
                self._update_progress(repo_name, "저장소 복제/로드 중...")
            elif "코드 인덱싱" in message or "코드 파일" in message:
                self._update_progress(repo_name, "코드 파일 로드 및 분할 중...")
            elif "code FAISS 인덱스 생성" in message:
                self._update_progress(repo_name, "코드 임베딩 생성 중...")
            elif "문서 인덱싱" in message or "문서 파일" in message:
                self._update_progress(repo_name, "문서 파일 로드 및 분할 중...")
            elif "document FAISS 인덱스 생성" in message:
                self._update_progress(repo_name, "문서 임베딩 생성 중...")

        # 인덱서에 콜백 전달하여 실행
        vector_stores = self.indexer.create_indexes_from_repository(
            repo_url, progress_callback=progress_callback
        )

        return vector_stores

    def _send_completion_callback(
        self, repo_url: str, status: str, error_message: Optional[str] = None
    ) -> None:
        """Express로 분석 완료 콜백을 전송합니다.

        Args:
            repo_url: 저장소 URL
            status: 완료 상태 ("completed" 또는 "failed")
            error_message: 오류 메시지 (실패 시)
        """
        try:
            # owner/repo 형식으로 repo_name 생성
            owner, repo = get_repo_owner_and_name(repo_url)
            full_repo_name = f"{owner}/{repo}"

            callback_url = self.callback_urls.get(full_repo_name)
            if not callback_url:
                logger.debug(f"콜백 URL이 없습니다: '{full_repo_name}'")
                return

            payload = {
                "repo_name": full_repo_name,  # owner/repo 형식으로 전송
                "status": status,
            }

            # 사용자 ID가 있으면 추가
            user_id = self.user_ids.get(full_repo_name)
            if user_id:
                payload["user_id"] = user_id

            if error_message:
                payload["error_message"] = error_message

            logger.info(f"Express로 완료 콜백 전송: {callback_url} - {payload}")

            response = requests.post(
                callback_url,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.info(f"콜백 전송 성공: '{full_repo_name}' -> {status}")
            else:
                logger.warning(
                    f"콜백 전송 실패: '{full_repo_name}' - HTTP {response.status_code}"
                )
                logger.warning(f"응답 내용: {response.text}")

        except Exception as e:
            logger.error(f"콜백 전송 중 오류: '{repo_url}' - {e}")
        finally:
            # 콜백 URL과 사용자 ID 정리
            try:
                owner, repo = get_repo_owner_and_name(repo_url)
                full_repo_name = f"{owner}/{repo}"
                self.callback_urls.pop(full_repo_name, None)
                self.user_ids.pop(full_repo_name, None)
            except Exception:
                pass

    def _update_progress(self, repo_name: str, message: str) -> None:
        """진행 상황을 업데이트합니다.

        Args:
            repo_name: 저장소 이름
            message: 진행 메시지
        """
        self.status_service.update_repository_status(
            repo_name, {"progress_message": message}
        )

    def _set_completion_status(
        self, repo_name: str, vector_stores: Dict[str, Any]
    ) -> None:
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
            "document_index_status": self._get_index_status(
                vector_stores.get("document")
            ),
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
