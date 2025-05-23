import logging
import time
import traceback
from typing import List, Optional

import google.generativeai as genai
from langchain_core.embeddings import Embeddings

from config import Config
from common.exceptions import EmbeddingError

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


class GeminiAPIEmbeddings(Embeddings):
    """Gemini API를 사용하여 텍스트 임베딩을 생성하는 클래스."""

    def __init__(
        self,
        model_name: str,
        document_task_type: str = "RETRIEVAL_DOCUMENT",  # 문서 임베딩용
        query_task_type: str = "RETRIEVAL_QUERY",  # 검색 쿼리 임베딩용
    ):
        self.model_name = model_name
        self.document_task_type = document_task_type
        self.query_task_type = query_task_type
        self.max_retries = Config.MAX_RETRIES
        genai.configure(api_key=Config.GEMINI_API_KEY)  # API 키 설정

    def _calculate_sleep_time(self, is_quota_error: bool) -> float:
        """오류 유형에 따라 대기 시간을 계산합니다."""
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

    def embed_documents(self, texts: List[str]) -> List[Optional[List[float]]]:
        """여러 문서에 대한 임베딩을 생성합니다."""
        all_embs: List[Optional[List[float]]] = []
        batch_size = Config.EMBEDDING_BATCH_SIZE
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for j_idx in range(0, len(texts), batch_size):
            batch_texts = texts[j_idx : j_idx + batch_size]
            if not batch_texts:
                logger.info("빈 문서 배치를 건너뛰었습니다.")
                continue

            current_batch_num = j_idx // batch_size + 1
            logger.info(
                f"임베딩 배치 처리 중 (Task: {self.document_task_type}): {current_batch_num}/{total_batches}"
            )

            retries_count = 0
            current_batch_embeddings: Optional[List[List[float]]] = None
            while retries_count < self.max_retries:
                try:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch_texts,
                        task_type=self.document_task_type,
                    )
                    current_batch_embeddings = result["embedding"]
                    all_embs.extend(current_batch_embeddings)
                    logger.debug(f"배치 {current_batch_num} 임베딩 성공.")
                    break  # 성공 시 루프 탈출
                except Exception as e_emb_docs:
                    retries_count += 1
                    logger.warning(
                        f"임베딩 중 오류 (시도 {retries_count}/{self.max_retries}): {e_emb_docs}"
                    )
                    is_quota_error = "429" in str(e_emb_docs)  # 할당량 오류인지 확인
                    if retries_count < self.max_retries:
                        sleep_time = self._calculate_sleep_time(is_quota_error)
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"배치 임베딩 실패 (최대 재시도 도달): {batch_texts[:2]}..."
                        )
                        # 실패한 배치에 대해 None을 추가하여 전체 길이를 유지
                        all_embs.extend([None] * len(batch_texts))
                        break  # 재시도 모두 실패 시 루프 탈출

            if j_idx + batch_size < len(texts):
                # 마지막 배치가 아닌 경우, 할당량 오류 방지를 위해 짧은 시간 대기
                # 실제로는 _calculate_sleep_time 에서 이미 대기하므로, 여기서는 불필요할 수 있음
                # time.sleep(1) # 필요시 활성화
                pass  # 이미 _calculate_sleep_time 에서 처리

        valid_embeddings_count = sum(1 for emb in all_embs if emb is not None)
        if valid_embeddings_count != len(all_embs):
            logger.warning(
                f"{len(all_embs) - valid_embeddings_count}개의 임베딩에 실패하여 건너뛰었습니다."
            )
        return all_embs

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리에 대한 임베딩을 생성합니다."""
        retries_count = 0
        logger.info(f"쿼리 임베딩 중 (Task: {self.query_task_type})...")
        while retries_count < self.max_retries:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=self.query_task_type,
                )
                logger.debug("쿼리 임베딩 성공.")
                return result["embedding"]
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
                    # 오류 발생 시 스택 트레이스 로깅
                    detailed_error_info = traceback.format_exc()
                    logger.error(
                        f"쿼리 임베딩 실패 (최대 재시도 도달). 오류: {e_emb_query}\n스택 트레이스:\n{detailed_error_info}"
                    )
                    raise EmbeddingError(
                        f"쿼리 임베딩에 실패했습니다 (재시도 {self.max_retries}회 시도 후): {text[:50]}..."
                    ) from e_emb_query
        # 이론적으로 이 지점에 도달해서는 안 됩니다.
        raise EmbeddingError("쿼리 임베딩 로직의 예기치 않은 종료.")
