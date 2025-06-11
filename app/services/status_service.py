import logging
import threading
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from app.core.utils import (
    extract_repo_name_from_url,
    get_local_repo_path,
    get_faiss_index_path,
    check_index_exists,
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
        if not hasattr(self, "_initialized"):
            self.repository_status: Dict[str, Dict[str, Any]] = {}
            self._status_data_lock = threading.Lock()
            self._initialized = True

            # 진행률 매핑 정의
            self.progress_mapping = {
                "저장소 정보 확인": 5,
                "저장소 복제": 10,
                "코드 파일 로드": 20,
                "코드 임베딩": 50,
                "문서 파일 로드": 70,
                "문서 임베딩": 90,
                "완료": 100,
            }

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

            # 진행률과 예상 시간 업데이트
            status_data = self.repository_status[repo_name].copy()
            self._update_progress_and_eta(status_data)

            return status_data

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
                "progress": 100,
                "progress_message": "인덱스 파일이 존재합니다 (상세 기록 없음)",
                "last_updated_time": self._get_current_timestamp(),
                "estimated_completion": None,
            }
        else:
            return {
                "status": "not_indexed",
                "repo_name": repo_name,
                "progress": 0,
                "progress_message": "인덱싱된 정보가 없습니다.",
                "estimated_completion": None,
            }

    def update_repository_status(
        self, repo_name: str, status_info: Dict[str, Any]
    ) -> None:
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

            # 진행률 자동 계산
            self._calculate_progress(repo_name)

            logger.debug(
                f"상태 업데이트 완료 - '{repo_name}': {status_info.get('status')}"
            )

    def update_progress(
        self, repo_name: str, stage: str, message: str, batch_info: Dict = None
    ) -> None:
        """진행 상황 업데이트"""
        if repo_name not in self.repository_status:
            return

        try:
            # 진행률 계산 (인자 개수 맞춤)
            progress = self._calculate_progress_by_stage(stage, batch_info)

            # 예상 완료 시간 계산
            estimated_completion, eta_text = self._calculate_eta(repo_name, progress)

            self.repository_status[repo_name].update(
                {
                    "current_step": message,
                    "progress": progress,
                    "stage": stage,
                    "last_updated": datetime.now(timezone.utc),
                    "estimated_completion": estimated_completion,
                    "eta_text": eta_text,
                }
            )

            # 배치 정보가 있으면 저장
            if batch_info:
                self.repository_status[repo_name]["batch_info"] = batch_info

        except Exception as e:
            logger.error(f"진행 상황 업데이트 오류 ({repo_name}): {e}")

    def _calculate_progress_by_stage(
        self, stage: str, batch_info: Dict = None
    ) -> float:
        """단계별 진행률 계산 (stage, batch_info 기반)"""
        base_progress = {
            "info_check": 5,
            "clone": 10,
            "code_indexing": 20,
            "code_loading": 20,
            "code_embedding": 20,  # 임베딩 시작점
        }

        progress = base_progress.get(stage, 20)

        if stage == "code_embedding" and batch_info:
            completed_batches = batch_info.get("completed_batches", 0)
            total_batches = batch_info.get("total_batches", 1)
            if total_batches > 0:
                embedding_progress = (completed_batches / total_batches) * 80
                progress = 20 + embedding_progress

        # 소수점 없이 올림 처리
        return min(math.ceil(progress), 100)

    def _calculate_progress(self, repo_name: str) -> None:
        """현재 단계에 따른 진행률을 계산합니다.

        Args:
            repo_name: 저장소 이름
        """
        if repo_name not in self.repository_status:
            return

        status_data = self.repository_status[repo_name]
        progress_message = status_data.get("progress_message", "")

        # 진행률 계산
        calculated_progress = 0

        if "저장소 정보 확인" in progress_message or "저장소 복제" in progress_message:
            calculated_progress = self.progress_mapping["저장소 정보 확인"]
        elif (
            "저장소를 복제" in progress_message
            or "저장소 복제/로드" in progress_message
        ):
            calculated_progress = self.progress_mapping["저장소 복제"]
        elif "코드 인덱싱" in progress_message or "코드 파일" in progress_message:
            calculated_progress = self.progress_mapping["코드 파일 로드"]
        elif (
            "code FAISS 인덱스 생성" in progress_message
            or "임베딩 시작" in progress_message
        ):
            calculated_progress = self.progress_mapping["코드 임베딩"]
        elif "문서 인덱싱" in progress_message or "문서 파일" in progress_message:
            calculated_progress = self.progress_mapping["문서 파일 로드"]
        elif "document FAISS 인덱스 생성" in progress_message:
            calculated_progress = self.progress_mapping["문서 임베딩"]
        elif status_data.get("status") == "completed":
            calculated_progress = self.progress_mapping["완료"]

        # 진행률이 이전보다 높을 때만 업데이트 (역행 방지)
        current_progress = status_data.get("progress", 0)
        if calculated_progress > current_progress:
            status_data["progress"] = calculated_progress

    def _update_progress_and_eta(self, status_data: Dict[str, Any]) -> None:
        """진행률과 예상 완료 시간을 업데이트합니다.

        Args:
            status_data: 상태 데이터
        """
        if status_data.get("status") not in ["indexing", "pending"]:
            return

        start_time_str = status_data.get("start_time")
        if not start_time_str:
            return

        try:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            current_time = datetime.now(timezone.utc)
            elapsed_seconds = (current_time - start_time).total_seconds()

            current_progress = status_data.get("progress", 0)

            if current_progress > 0 and current_progress < 100:
                # 예상 총 시간 계산 (현재 진행률 기준)
                estimated_total_seconds = (elapsed_seconds / current_progress) * 100
                remaining_seconds = estimated_total_seconds - elapsed_seconds

                if remaining_seconds > 0:
                    estimated_completion = current_time + timedelta(
                        seconds=remaining_seconds
                    )
                    status_data["estimated_completion"] = (
                        estimated_completion.isoformat()
                    )

                    # 남은 시간을 사람이 읽기 쉬운 형태로 변환
                    if remaining_seconds < 60:
                        eta_text = f"약 {int(remaining_seconds)}초 남음"
                    elif remaining_seconds < 3600:
                        eta_text = f"약 {int(remaining_seconds/60)}분 남음"
                    else:
                        hours = int(remaining_seconds / 3600)
                        minutes = int((remaining_seconds % 3600) / 60)
                        eta_text = f"약 {hours}시간 {minutes}분 남음"

                    status_data["eta_text"] = eta_text
                else:
                    status_data["eta_text"] = "곧 완료"
            elif current_progress >= 100:
                status_data["eta_text"] = "완료"
            else:
                status_data["eta_text"] = "계산 중..."

        except Exception as e:
            logger.warning(f"예상 시간 계산 오류: {e}")
            status_data["eta_text"] = "계산 중..."

    def complete_indexing(self, repo_name: str) -> None:
        """인덱싱 완료 상태를 설정합니다.

        Args:
            repo_name: 저장소 이름
        """
        with self._status_data_lock:
            if repo_name not in self.repository_status:
                logger.warning(
                    f"존재하지 않는 저장소에 완료 상태 설정 시도: '{repo_name}'"
                )
                return

            current_time = self._get_current_timestamp()
            completion_update = {
                "status": "completed",
                "end_time": current_time,
                "last_updated_time": current_time,
                "progress": 100,
                "progress_message": "인덱싱이 완료되었습니다.",
                "eta_text": "완료",
                "error": None,
                "error_code": None,
            }

            self.repository_status[repo_name].update(completion_update)
            logger.info(f"인덱싱 완료 상태 설정 완료: '{repo_name}'")

    def fail_indexing(
        self, repo_name: str, error_message: str, error_code: str = None
    ) -> None:
        """인덱싱 실패 상태를 설정합니다.

        Args:
            repo_name: 저장소 이름
            error_message: 오류 메시지
            error_code: 오류 코드 (선택사항)
        """
        with self._status_data_lock:
            if repo_name not in self.repository_status:
                logger.warning(
                    f"존재하지 않는 저장소에 실패 상태 설정 시도: '{repo_name}'"
                )
                return

            current_time = self._get_current_timestamp()
            failure_update = {
                "status": "failed",
                "end_time": current_time,
                "last_updated_time": current_time,
                "progress": 0,
                "progress_message": f"인덱싱 실패: {error_message}",
                "eta_text": "실패",
                "error": error_message,
                "error_code": error_code or "INDEXING_FAILED",
            }

            self.repository_status[repo_name].update(failure_update)
            logger.error(f"인덱싱 실패 상태 설정 완료: '{repo_name}' - {error_message}")

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
                self._update_progress_and_eta(current_data)
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
                "progress": 0,
                "progress_message": "인덱싱 작업 시작 대기 중...",
                "estimated_completion": None,
                "eta_text": "계산 중...",
                "is_new_request": True,
            }

            self.repository_status[repo_name] = initial_status
            logger.info(f"초기 인덱싱 상태 설정 완료: '{repo_name}'")

            return initial_status.copy()

    def set_error_status(
        self, repo_name: str, error_msg: str, error_code: str = None
    ) -> None:
        """인덱싱 오류 발생 시 상태를 'failed'로 업데이트합니다.

        Args:
            repo_name: 저장소 이름
            error_msg: 오류 메시지
            error_code: 오류 코드 (선택사항)
        """
        with self._status_data_lock:
            if repo_name not in self.repository_status:
                logger.warning(
                    f"존재하지 않는 저장소에 오류 상태 설정 시도: '{repo_name}'"
                )
                return

            current_time = self._get_current_timestamp()
            current_status = self.repository_status[repo_name]

            error_update = {
                "status": "failed",
                "end_time": current_time,
                "last_updated_time": current_time,
                "error": error_msg,
                "error_code": error_code,
                "progress": 0,
                "progress_message": f"인덱싱 실패: {error_msg}",
                "code_index_status": self._get_failed_status(
                    current_status.get("code_index_status")
                ),
                "document_index_status": self._get_failed_status(
                    current_status.get("document_index_status")
                ),
                "eta_text": "실패",
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

    def _calculate_eta(self, repo_name: str, current_progress: float):
        """예상 완료 시간과 ETA 텍스트를 계산합니다."""
        try:
            status = self.repository_status.get(repo_name, {})
            start_time_str = status.get("start_time")
            if not start_time_str or current_progress <= 0:
                return None, "계산 중..."

            now = datetime.now(timezone.utc)
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            elapsed = (now - start_time).total_seconds()

            # 임베딩 배치 기반 ETA 계산
            stage = status.get("stage", "")
            if stage == "code_embedding":
                batch_info = status.get("batch_info", {})
                remaining_batches = batch_info.get("total_batches", 1) - batch_info.get(
                    "completed_batches", 0
                )
                remaining_seconds = remaining_batches * 5
                estimated_completion = now + timedelta(seconds=remaining_seconds)
                if remaining_seconds < 60:
                    eta_text = f"약 {math.ceil(remaining_seconds)}초 남음"
                elif remaining_seconds < 3600:
                    eta_text = f"약 {math.ceil(remaining_seconds/60)}분 남음"
                else:
                    hours = math.ceil(remaining_seconds / 3600)
                    eta_text = f"약 {hours}시간 남음"
                return estimated_completion.isoformat(), eta_text

            # 일반 단계 ETA 계산
            if current_progress >= 100:
                return now.isoformat(), "완료"

            time_per_percent = (
                elapsed / current_progress if current_progress > 0 else 60
            )
            remaining_progress = 100 - current_progress
            remaining_seconds = remaining_progress * time_per_percent
            estimated_completion = now + timedelta(seconds=remaining_seconds)
            if remaining_seconds < 60:
                eta_text = f"약 {math.ceil(remaining_seconds)}초 남음"
            elif remaining_seconds < 3600:
                eta_text = f"약 {math.ceil(remaining_seconds/60)}분 남음"
            else:
                hours = math.ceil(remaining_seconds / 3600)
                eta_text = f"약 {hours}시간 남음"
            return estimated_completion.isoformat(), eta_text
        except Exception as e:
            logger.error(f"ETA 계산 오류: {e}")
            return None, "계산 중..."

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
