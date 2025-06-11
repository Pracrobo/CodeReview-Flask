import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google.genai import types
from langchain_core.embeddings import Embeddings

from app.core.config import Config
from app.core.exceptions import EmbeddingError
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)


class GeminiAPIEmbeddings(Embeddings):
    """Gemini API 텍스트 임베딩 생성 클래스 (병렬 처리 지원)"""

    def __init__(
        self,
        model_name,
        document_task_type="RETRIEVAL_DOCUMENT",  # 문서 임베딩용
        query_task_type="RETRIEVAL_QUERY",  # 검색 쿼리 임베딩용
    ):
        self.model_name = model_name
        self.document_task_type = document_task_type
        self.query_task_type = query_task_type
        self.max_retries = Config.MAX_RETRIES
        self.lock = Lock()

    def _get_client(self):
        """필요할 때마다 새로운 클라이언트를 생성합니다."""
        try:
            return gemini_service.get_client()
        except Exception as e:
            raise EmbeddingError(f"Gemini API 클라이언트 생성 실패: {e}")

    def _get_clients_for_batch(self, num_batches):
        """배치 처리용 클라이언트들을 생성합니다."""
        clients = []
        api_keys = gemini_service.get_available_api_keys()
        if not api_keys:
            raise EmbeddingError("유효한 Gemini API 키가 설정되지 않았습니다.")
        for i in range(min(num_batches, len(api_keys))):
            try:
                client = gemini_service.get_client_with_key(api_keys[i])
                clients.append(client)
            except Exception as e:
                logger.warning(f"클라이언트 {i+1} 생성 실패: {e}")
        if not clients:
            raise EmbeddingError("사용 가능한 Gemini API 클라이언트가 없습니다.")
        return clients

    def _calculate_sleep_time(self, is_quota_error, retries_count):
        """오류 유형별 API 재시도 대기 시간 (지수 백오프 적용)"""
        base_time = (
            Config.QUOTA_ERROR_SLEEP_TIME
            if is_quota_error
            else Config.GENERAL_API_ERROR_SLEEP_TIME
        )
        sleep_time = base_time * (2 ** (retries_count - 1))
        max_sleep = 60  # 최대 대기 1분 제한
        sleep_time = min(sleep_time, max_sleep)
        logger.warning(
            f"{'할당량' if is_quota_error else '일반'} 오류 발생. {sleep_time}초 후 재시도합니다. (지수 백오프 적용)"
        )
        return sleep_time

    def _process_batch_with_client(
        self, client, batch_texts, batch_start_idx, batch_num, total_batches
    ):
        """개별 클라이언트로 배치 처리"""
        retries_count = 0

        while retries_count < self.max_retries:
            try:
                result = client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=Config.EMBEDDING_DIMENSION
                    ),
                )
                embeddings = [emb.values for emb in result.embeddings]
                success_indices = list(
                    range(batch_start_idx, batch_start_idx + len(batch_texts))
                )

                with self.lock:
                    logger.debug(f"배치 {batch_num} 임베딩 성공 ({len(embeddings)}개)")
                    # 성공 후 대기 코드 제거

                # 성공 후 대기 없음

                return {
                    "success": True,
                    "embeddings": embeddings,
                    "indices": success_indices,
                    "batch_num": batch_num,
                }

            except Exception as e:
                retries_count += 1
                with self.lock:
                    logger.warning(
                        f"배치 {batch_num} 오류 (시도 {retries_count}/{self.max_retries}): {e}"
                    )

                is_quota_error = "429" in str(e)
                if retries_count < self.max_retries:
                    sleep_time = self._calculate_sleep_time(
                        is_quota_error, retries_count
                    )
                    time.sleep(sleep_time)
                else:
                    with self.lock:
                        logger.error(
                            f"배치 {batch_num} 임베딩 최종 실패 (최대 재시도 도달)"
                        )
                    failed_indices = list(
                        range(batch_start_idx, batch_start_idx + len(batch_texts))
                    )
                    return {
                        "success": False,
                        "failed_indices": failed_indices,
                        "batch_num": batch_num,
                    }

        failed_indices = list(
            range(batch_start_idx, batch_start_idx + len(batch_texts))
        )
        return {
            "success": False,
            "failed_indices": failed_indices,
            "batch_num": batch_num,
        }

    def embed_documents(self, texts):
        """다수 문서 임베딩 (병렬 처리)"""
        if not texts:
            logger.info("빈 문서 리스트입니다.")
            return [], []

        successful_embeddings = [None] * len(texts)
        failed_original_indices = []

        batch_size = Config.EMBEDDING_BATCH_SIZE
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # 배치 처리용 클라이언트들 생성
        clients = self._get_clients_for_batch(total_batches)

        logger.info(
            f"병렬 임베딩 시작. 총 배치: {total_batches}, 클라이언트: {len(clients)}개"
        )

        # 진행 상황 콜백 함수가 있다면 사용
        progress_callback = getattr(self, "_progress_callback", None)

        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = []

            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx : batch_idx + batch_size]
                if not batch_texts:
                    continue

                current_batch_num = batch_idx // batch_size + 1
                client = clients[current_batch_num % len(clients)]

                future = executor.submit(
                    self._process_batch_with_client,
                    client,
                    batch_texts,
                    batch_idx,
                    current_batch_num,
                    total_batches,
                )
                futures.append(future)

            successful_count = 0
            failed_count = 0
            completed_batches = 0

            for future in as_completed(futures):
                try:
                    result = future.result()
                    completed_batches += 1

                    # 진행 상황 업데이트
                    if progress_callback:
                        batch_info = {
                            "completed_batches": completed_batches,
                            "total_batches": total_batches,
                        }
                        # 3개 인자로 호출
                        progress_callback(
                            "code_embedding",
                            f"임베딩 진행 중 ({completed_batches}/{total_batches})",
                            batch_info,
                        )

                    if result["success"]:
                        for i, embedding in enumerate(result["embeddings"]):
                            original_idx = result["indices"][i]
                            successful_embeddings[original_idx] = embedding
                        successful_count += len(result["embeddings"])
                    else:
                        failed_original_indices.extend(result["failed_indices"])
                        failed_count += len(result["failed_indices"])
                except Exception as e:
                    logger.error(f"병렬 처리 중 예기치 않은 오류: {e}")

        final_embeddings = [emb for emb in successful_embeddings if emb is not None]

        logger.info(
            f"병렬 임베딩 완료. 총 문서: {len(texts)}, "
            f"성공: {successful_count}, 실패: {failed_count}"
        )

        if successful_count == 0 and failed_count > 0:
            logger.warning("모든 문서의 임베딩 생성에 실패했습니다.")

        return final_embeddings, sorted(list(set(failed_original_indices)))

    def set_progress_callback(self, callback):
        """진행 상황 콜백 함수 설정"""
        self._progress_callback = callback

    def embed_query(self, text):
        """단일 쿼리 임베딩 생성"""
        client = self._get_client()
        retries_count = 0
        logger.info(f"쿼리 임베딩 중 (Task: {self.query_task_type})...")

        while retries_count < self.max_retries:
            try:
                result = client.models.embed_content(
                    model=self.model_name,
                    contents=[text],
                    config=types.EmbedContentConfig(
                        output_dimensionality=Config.EMBEDDING_DIMENSION
                    ),
                )
                logger.info("쿼리 임베딩 성공.")
                return result.embeddings[0].values
            except Exception as e_emb_query:
                retries_count += 1
                logger.warning(
                    f"쿼리 임베딩 중 오류 (시도 {retries_count}/{self.max_retries}): {e_emb_query}"
                )
                is_quota_error = "429" in str(e_emb_query)
                if retries_count < self.max_retries:
                    sleep_time = self._calculate_sleep_time(
                        is_quota_error, retries_count
                    )
                    time.sleep(sleep_time)
                else:
                    detailed_error_info = traceback.format_exc()
                    logger.error(
                        f"쿼리 임베딩 실패 (최대 재시도 도달). 오류: {e_emb_query}\n스택 트레이스:\n{detailed_error_info}"
                    )
                    raise EmbeddingError(
                        f"쿼리 임베딩에 실패했습니다 (재시도 {self.max_retries}회 시도 후): {text[:50]}..."
                    ) from e_emb_query
        raise EmbeddingError("쿼리 임베딩 로직의 예기치 않은 종료.")
