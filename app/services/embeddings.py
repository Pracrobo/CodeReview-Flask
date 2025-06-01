import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

# 수정: config 및 예외 클래스 임포트 경로 변경
from app.core.config import Config
from app.core.exceptions import EmbeddingError

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

        # API 키 로드 실패 시 앱 로딩 단계에서 알 수 있도록 try-except 추가
        try:
            self.client1 = genai.Client(api_key=Config.GEMINI_API_KEY1)
            self.client2 = genai.Client(api_key=Config.GEMINI_API_KEY2)
            self.clients = [self.client1, self.client2]
            if not Config.GEMINI_API_KEY1 or not Config.GEMINI_API_KEY2:
                logger.warning("하나 이상의 Gemini API 키가 설정되지 않았습니다. 기능이 제한될 수 있습니다.")
        except Exception as e:
            logger.error(f"Gemini 클라이언트 초기화 실패: {e}. API 키 설정을 확인하세요.")
            # 두 클라이언트 모두 None으로 설정하거나, 에러를 발생시켜 앱 실행을 중단할 수 있습니다.
            self.client1 = None
            self.client2 = None
            self.clients = [] # 빈 리스트로 설정하여 이후 로직에서 문제 발생 방지
            # raise EmbeddingError("Gemini API 클라이언트 초기화 실패") # 필요시 앱 중단

        self.lock = Lock()

    def _check_clients_available(self):
        if not self.clients or not all(self.clients):
            raise EmbeddingError("Gemini API 클라이언트가 제대로 초기화되지 않았습니다.")
        return True

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
        self._check_clients_available()
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
                    if batch_num < total_batches:
                        logger.info(
                            f"성공적인 임베딩 후 {Config.SUCCESS_SLEEP_TIME}초 대기합니다."
                        )

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
        self._check_clients_available()
        if not texts:
            logger.info("빈 문서 리스트입니다.")
            return [], []

        successful_embeddings = [None] * len(texts)
        failed_original_indices = []

        batch_size = Config.EMBEDDING_BATCH_SIZE
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"병렬 임베딩 시작. 총 배치: {total_batches}, 클라이언트: {len(self.clients)}개")
        if not self.clients : # 클라이언트가 하나도 없으면 단일 스레드로 (혹은 오류)
             logger.warning("사용 가능한 Gemini 클라이언트가 없어 단일 스레드로 임베딩을 시도합니다.")
             # 이 경우 _process_batch_with_client를 직접 호출하거나, 다른 로직 필요
             # 여기서는 간단히 빈 결과 반환 또는 예외 발생
             if len(texts) > 0:
                 raise EmbeddingError("Gemini 클라이언트 없이 문서 임베딩 불가")
             return [], []

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = []

            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx : batch_idx + batch_size]
                if not batch_texts:
                    continue

                current_batch_num = batch_idx // batch_size + 1
                client = self.clients[current_batch_num % len(self.clients)]

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

            for future in as_completed(futures):
                try:
                    result = future.result()
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

    def embed_query(self, text):
        """단일 쿼리 임베딩 생성 (첫 번째 사용 가능한 클라이언트 사용)"""
        self._check_clients_available()
        # 첫번째 클라이언트가 없으면 두번째, 둘다 없으면 에러
        active_client = self.client1 if self.client1 else self.client2
        if not active_client:
            raise EmbeddingError("쿼리 임베딩을 위한 Gemini API 클라이언트가 없습니다.")
            
        retries_count = 0
        logger.info(f"쿼리 임베딩 중 (Task: {self.query_task_type})...")

        while retries_count < self.max_retries:
            try:
                result = active_client.models.embed_content(
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