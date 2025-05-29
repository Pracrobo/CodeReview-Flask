import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

from config import Config
from common.exceptions import EmbeddingError

# 로거 설정
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

        # 2개의 클라이언트 초기화 (각각 다른 API 키 사용)
        self.client1 = genai.Client(api_key=Config.GEMINI_API_KEY1)
        self.client2 = genai.Client(api_key=Config.GEMINI_API_KEY2)
        self.clients = [self.client1, self.client2]

        # 스레드 안전성을 위한 락
        self.lock = Lock()

    def _calculate_sleep_time(self, is_quota_error):
        """오류 유형별 API 재시도 대기 시간 계산"""
        if is_quota_error:
            logger.warning(
                f"할당량 오류 발생. {Config.QUOTA_ERROR_SLEEP_TIME}초 후 재시도합니다."
            )
            return Config.QUOTA_ERROR_SLEEP_TIME
        else:
            logger.warning(
                f"일시적인 API 오류 발생. {Config.GENERAL_API_ERROR_SLEEP_TIME}초 후 재시도합니다."
            )
            return Config.GENERAL_API_ERROR_SLEEP_TIME

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
                # 성공한 임베딩과 인덱스 정보 반환
                embeddings = [emb.values for emb in result.embeddings]
                success_indices = list(
                    range(batch_start_idx, batch_start_idx + len(batch_texts))
                )

                with self.lock:
                    logger.debug(f"배치 {batch_num} 임베딩 성공 ({len(embeddings)}개)")
                    # 마지막 배치가 아닌 경우에만 대기 메시지 출력
                    if batch_num < total_batches:
                        logger.info(
                            f"성공적인 임베딩 후 {Config.SUCCESS_SLEEP_TIME}초 대기합니다."
                        )

                # 마지막 배치가 아닌 경우에만 대기
                if batch_num < total_batches:
                    time.sleep(Config.SUCCESS_SLEEP_TIME)

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
                    sleep_time = self._calculate_sleep_time(is_quota_error)
                    time.sleep(sleep_time)
                else:
                    with self.lock:
                        logger.error(
                            f"배치 {batch_num} 임베딩 최종 실패 (최대 재시도 도달)"
                        )

                    # 실패한 인덱스 정보 반환
                    failed_indices = list(
                        range(batch_start_idx, batch_start_idx + len(batch_texts))
                    )
                    return {
                        "success": False,
                        "failed_indices": failed_indices,
                        "batch_num": batch_num,
                    }

        # 이론상 도달 불가
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

        successful_embeddings = [None] * len(texts)  # 원본 순서 유지용
        failed_original_indices = []

        batch_size = Config.EMBEDDING_BATCH_SIZE
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"병렬 임베딩 시작. 총 배치: {total_batches}, 클라이언트: 2개")

        # 병렬 처리를 위한 작업 생성
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx : batch_idx + batch_size]
                if not batch_texts:
                    continue

                current_batch_num = batch_idx // batch_size + 1
                # 배치를 2개 클라이언트에 번갈아 할당
                client = self.clients[current_batch_num % 2]

                # 병렬 작업 제출 (total_batches 전달)
                future = executor.submit(
                    self._process_batch_with_client,
                    client,
                    batch_texts,
                    batch_idx,
                    current_batch_num,
                    total_batches,
                )
                futures.append(future)

            # 결과 수집
            successful_count = 0
            failed_count = 0

            for future in as_completed(futures):
                try:
                    result = future.result()

                    if result["success"]:
                        # 성공한 임베딩을 원본 위치에 저장
                        for i, embedding in enumerate(result["embeddings"]):
                            original_idx = result["indices"][i]
                            successful_embeddings[original_idx] = embedding
                        successful_count += len(result["embeddings"])
                    else:
                        # 실패한 인덱스 기록
                        failed_original_indices.extend(result["failed_indices"])
                        failed_count += len(result["failed_indices"])

                except Exception as e:
                    logger.error(f"병렬 처리 중 예기치 않은 오류: {e}")

        # None 값 제거 및 최종 결과 생성
        final_embeddings = [emb for emb in successful_embeddings if emb is not None]

        logger.info(
            f"병렬 임베딩 완료. 총 문서: {len(texts)}, "
            f"성공: {successful_count}, 실패: {failed_count}"
        )

        if successful_count == 0 and failed_count > 0:
            logger.warning("모든 문서의 임베딩 생성에 실패했습니다.")

        return final_embeddings, sorted(list(set(failed_original_indices)))

    def embed_query(self, text):
        """단일 쿼리 임베딩 생성 (첫 번째 클라이언트 사용)"""
        retries_count = 0
        logger.info(f"쿼리 임베딩 중 (Task: {self.query_task_type})...")

        while retries_count < self.max_retries:
            try:
                result = self.client1.models.embed_content(
                    model=self.model_name,
                    contents=[text],
                    config=types.EmbedContentConfig(
                        output_dimensionality=Config.EMBEDDING_DIMENSION
                    ),
                )

                logger.info("쿼리 임베딩 성공.")
                # 쿼리 임베딩 후에는 대기하지 않음

                return result.embeddings[0].values

            except Exception as e_emb_query:
                retries_count += 1
                logger.warning(
                    f"쿼리 임베딩 중 오류 (시도 {retries_count}/{self.max_retries}): {e_emb_query}"
                )
                is_quota_error = "429" in str(e_emb_query)
                if retries_count < self.max_retries:
                    sleep_time = self._calculate_sleep_time(is_quota_error)
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
