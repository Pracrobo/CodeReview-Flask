import logging

import google.generativeai as genai

# FAISS CPU 전용 설정 (GPU 경고 방지)
import faiss
from config import Config
from common.exceptions import EmbeddingError, RAGError

faiss.omp_set_num_threads(1)  # CPU 스레드 수 제한으로 안정성 향상


# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


def translate_code_query_to_english(korean_text, llm_model_name):
    """코드 관련 한국어 질의 영어 번역"""
    try:
        llm = genai.GenerativeModel(llm_model_name)
        prompt = f"""
        다음 한국어 코드 관련 질문을 영어로 번역해주세요. 
        프로그래밍 용어, 함수명, 클래스명, 변수명은 정확히 유지하세요.
        코드의 의미와 맥락을 살려서 번역하세요.
        번역된 영어 텍스트만 출력하세요.
        
        한국어 질문: {korean_text}
        
        English question:
        """
        response = llm.generate_content(prompt)
        english_text = response.text.strip()
        logger.info(f"코드 질의 번역 완료: '{korean_text}' -> '{english_text}'")
        return english_text
    except Exception as e:
        logger.warning(f"코드 질의 번역 실패, 원본 텍스트 사용: {e}")
        return korean_text


def translate_to_english(korean_text, llm_model_name):
    """일반 한국어 텍스트 영어 번역"""
    try:
        llm = genai.GenerativeModel(llm_model_name)
        prompt = f"""
        다음 한국어 텍스트를 자연스러운 영어로 번역해주세요. 기술적 용어는 정확히 번역하세요.
        번역된 영어 텍스트만 출력하고 다른 설명은 하지 마세요.
        
        한국어: {korean_text}
        
        영어:
        """
        response = llm.generate_content(prompt)
        english_text = response.text.strip()
        logger.info(f"번역 완료: '{korean_text}' -> '{english_text}'")
        return english_text
    except Exception as e:
        logger.warning(f"번역 실패, 원본 텍스트 사용: {e}")
        return korean_text


def search_and_rag(
    vector_stores,
    target_index,  # "code" 또는 "document"
    search_query,
    llm_model_name,
    top_k=Config.DEFAULT_TOP_K,
    similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
):
    """벡터 저장소 검색 및 LLM 기반 답변 생성 (RAG)"""
    if target_index not in vector_stores or not vector_stores[target_index]:
        logger.warning(
            f"{target_index.capitalize()} 벡터 저장소를 사용할 수 없습니다. 검색 및 RAG를 건너뜁니다."
        )
        return None

    vector_store = vector_stores[target_index]

    # 타입별 번역 수행
    logger.info("사용자 질의를 영어로 번역 중...")
    if target_index == "code":
        english_query = translate_code_query_to_english(search_query, llm_model_name)
    else:
        english_query = translate_to_english(search_query, llm_model_name)

    logger.info(
        f"\n'{target_index.capitalize()}' 인덱스에 대한 의미론적 검색을 시작합니다 (Top K: {top_k}, 유사도 임계값: {similarity_threshold})..."
    )

    try:
        # 영어 질의로 유사도 점수와 함께 검색 수행
        search_results_with_scores = vector_store.similarity_search_with_score(
            english_query, k=top_k
        )

        # 유사도 임계값으로 필터링 (코드 검색의 경우 더 관대한 임계값 적용)
        adjusted_threshold = (
            similarity_threshold * 0.8
            if target_index == "code"
            else similarity_threshold
        )
        filtered_results = [
            (doc, score)
            for doc, score in search_results_with_scores
            if score <= (1 - adjusted_threshold)  # FAISS 거리 점수를 유사도로 변환
        ]

        logger.info(
            f"\n'{search_query}'에 대한 검색 결과 (총 {len(search_results_with_scores)}개 중 {len(filtered_results)}개가 임계값 통과, 조정된 임계값: {adjusted_threshold:.2f}):"
        )

        if not filtered_results:
            logger.info("유사도 임계값을 만족하는 검색 결과가 없습니다.")
            return "유사도가 충분히 높은 검색 결과가 없습니다."

        for i, (doc, score) in enumerate(filtered_results):
            similarity_percent = (1 - score) * 100  # 거리 점수를 유사도(%)로 변환
            logger.info(
                f"  {i+1}. 유사도: {similarity_percent:.1f}%, 소스: {doc.metadata.get('source', '알 수 없음')}, 내용 (일부): {doc.page_content[:100]}..."
            )

        # RAG 프롬프트 구성 (타입별 최적화)
        context_for_rag = "\n\n".join(
            [doc.page_content for doc, score in filtered_results]
        )

        if target_index == "code":
            prompt = f"""
            주어진 코드 컨텍스트를 바탕으로 다음 질문에 대해 한국어로 상세히 답변해 주세요.
            코드 예제가 있다면 포함하고, 함수나 클래스의 사용법을 설명해 주세요.
            만약 컨텍스트에 질문과 관련된 코드가 없다면, "컨텍스트에 관련 코드가 없습니다."라고 답변해 주세요.
            
            코드 컨텍스트:
            {context_for_rag}
            
            질문: {search_query}
            
            답변:
            """
        else:
            prompt = f"""
            주어진 컨텍스트 정보를 사용하여 다음 질문에 대해 한국어로 답변해 주세요.
            만약 컨텍스트에 질문과 관련된 정보가 없다면, "컨텍스트에 관련 정보가 없습니다."라고 답변해 주세요.
            
            컨텍스트:
            {context_for_rag}
            
            질문: {search_query}
            
            답변:
            """

        logger.info(
            f"\n'{llm_model_name}' 모델을 사용하여 RAG 답변 생성을 시작합니다..."
        )
        llm = genai.GenerativeModel(llm_model_name)
        response = llm.generate_content(prompt)

        logger.info("RAG 답변 생성 완료.")
        return response.text

    except EmbeddingError as e_embed_query_fail:
        logger.error(
            f"'{target_index.capitalize()}' 인덱스에 대한 쿼리 임베딩 중 오류 발생: {e_embed_query_fail}"
        )
        logger.error("유사도 검색 및 RAG를 진행할 수 없습니다.")
        return f"{target_index.capitalize()} 인덱스 검색 중 오류가 발생했습니다: 임베딩 실패"
    except Exception as e_rag:
        logger.error(
            f"'{target_index.capitalize()}' 유사도 검색 또는 RAG 중 예기치 않은 오류 발생: {e_rag}",
            exc_info=True,
        )
        raise RAGError(
            f"'{target_index.capitalize()}' 검색 또는 RAG 처리 중 오류 발생: {e_rag}"
        ) from e_rag
