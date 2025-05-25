import logging
import time
import traceback

import google.generativeai as genai
from langchain_core.embeddings import Embeddings

from config import Config
from common.exceptions import EmbeddingError

# 로거 설정
logger = logging.getLogger(__name__)


class GeminiAPIEmbeddings(Embeddings):
    """Gemini API 텍스트 임베딩 생성 클래스"""

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
        genai.configure(api_key=Config.GEMINI_API_KEY)  # API 키 설정

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

    def embed_documents(self, texts):
        """다수 문서 임베딩 (성공 임베딩, 실패 원본 인덱스 반환)"""
        successful_embeddings = []
        failed_original_indices = []

        successful_count = 0
        failed_count = 0

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
            try:
                while retries_count < self.max_retries:
                    try:
                        result = genai.embed_content(
                            model=self.model_name,
                            content=batch_texts,
                            task_type=self.document_task_type,
                        )
                        current_batch_embeddings = result["embedding"]
                        successful_embeddings.extend(current_batch_embeddings)
                        successful_count += len(current_batch_embeddings)
                        # 성공 시 처리 수 증가
                        logger.debug(
                            f"배치 {current_batch_num} 임베딩 성공 ({len(current_batch_embeddings)}개)."
                        )
                        break  # 성공 시 루프 탈출
                    except Exception as e_emb_docs:
                        retries_count += 1
                        logger.warning(
                            f"임베딩 배치 {current_batch_num} 오류 (시도 {retries_count}/{self.max_retries}): {e_emb_docs}"
                        )
                        is_quota_error = "429" in str(e_emb_docs)
                        if retries_count < self.max_retries:
                            sleep_time = self._calculate_sleep_time(is_quota_error)
                            time.sleep(sleep_time)
                        else:
                            logger.error(
                                f"배치 {current_batch_num} 임베딩 최종 실패 (최대 재시도 도달)."
                            )
                            # 배치 전체 실패 시, 해당 배치 모든 문서 인덱스 실패 기록
                            for k_idx in range(len(batch_texts)):
                                original_idx = j_idx + k_idx
                                failed_original_indices.append(original_idx)
                                logger.debug(
                                    f"문서 임베딩 실패 기록: 원본 인덱스 {original_idx}, 내용: '{texts[original_idx][:100]}...'"
                                )
                            failed_count += len(batch_texts)
                            # 실패 시에도 처리된 것으로 간주
                            break  # 재시도 모두 실패 시 루프 탈출
            except Exception as outer_e:
                # 예기치 않은 오류로 배치 처리 실패 시 (드문 경우)
                logger.error(
                    f"배치 {current_batch_num} 처리 중 예기치 않은 외부 오류: {outer_e}"
                )
                for k_idx in range(len(batch_texts)):
                    original_idx = j_idx + k_idx
                    failed_original_indices.append(original_idx)
                    logger.debug(
                        f"문서 임베딩 실패 기록 (외부 오류): 원본 인덱스 {original_idx}"
                    )
                failed_count += len(batch_texts)

        logger.info(
            f"임베딩 처리 완료. 총 시도 문서: {len(texts)}, "
            f"성공: {successful_count}, 실패: {failed_count} (실패 인덱스 수: {len(failed_original_indices)})"
        )

        if successful_count == 0 and failed_count > 0:
            logger.warning("모든 문서의 임베딩 생성에 실패했습니다.")

        # 반환: 성공 임베딩 리스트, 실패 문서 원본 인덱스 리스트
        return successful_embeddings, sorted(list(set(failed_original_indices)))

    def embed_query(self, text):
        """단일 쿼리 임베딩 생성"""
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
                    # 오류 시 스택 트레이스 로깅
                    detailed_error_info = traceback.format_exc()
                    logger.error(
                        f"쿼리 임베딩 실패 (최대 재시도 도달). 오류: {e_emb_query}\n스택 트레이스:\n{detailed_error_info}"
                    )
                    raise EmbeddingError(
                        f"쿼리 임베딩에 실패했습니다 (재시도 {self.max_retries}회 시도 후): {text[:50]}..."
                    ) from e_emb_query
        # 이론상 도달 불가 지점
        raise EmbeddingError("쿼리 임베딩 로직의 예기치 않은 종료.")
