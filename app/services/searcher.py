import logging

from google import genai
import faiss

# 수정: config 및 예외 클래스 임포트 경로 변경
from app.core.config import Config
from app.core.exceptions import EmbeddingError, RAGError
from app.core.prompts import prompts

faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)

# Gemini 클라이언트 초기화
try:
    api_key_to_use = Config.GEMINI_API_KEY1
    logger.info(f"Gemini 클라이언트 초기화를 시도합니다. API 키: '{api_key_to_use}'")
    
    # API 키가 dummy 값인지 확인
    if not api_key_to_use or api_key_to_use == "dummy_key_1":
        logger.warning("GEMINI_API_KEY1이 설정되지 않았거나 dummy 값입니다. 클라이언트를 None으로 설정합니다.")
        client = None
    else:
        client = genai.Client(api_key=api_key_to_use)
        logger.info("Gemini 클라이언트 초기화 성공")
except Exception as e:
    logger.error(f"Gemini 클라이언트 초기화 실패: {e}. API 키 설정을 확인하세요.")
    client = None

def translate_code_query_to_english(korean_text, llm_model_name):
    """코드 관련 한국어 질의 영어 번역"""
    if not client:
        logger.error("Gemini 클라이언트가 초기화되지 않아 번역을 건너뜁니다.")
        return korean_text
    try:
        prompt = prompts.get_code_query_translation_prompt(korean_text)
        response = client.models.generate_content(model=llm_model_name, contents=prompt)
        english_text = response.text.strip()
        logger.info(f"코드 질의 번역 완료: '{korean_text}' -> '{english_text}'")
        return english_text
    except Exception as e:
        logger.warning(f"코드 질의 번역 실패, 원본 텍스트 사용: {e}")
        return korean_text


def translate_to_english(korean_text, llm_model_name):
    """일반 한국어 텍스트 영어 번역"""
    if not client:
        logger.error("Gemini 클라이언트가 초기화되지 않아 번역을 건너뜁니다.")
        return korean_text
    try:
        prompt = prompts.get_general_translation_prompt(korean_text)
        response = client.models.generate_content(model=llm_model_name, contents=prompt)
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
    if not client:
        raise RAGError("Gemini 클라이언트가 초기화되지 않아 RAG를 수행할 수 없습니다.")
        
    if target_index not in vector_stores or not vector_stores[target_index]:
        logger.warning(
            f"{target_index.capitalize()} 벡터 저장소를 사용할 수 없습니다. 검색 및 RAG를 건너뜁니다."
        )
        return None # 또는 "관련 인덱스를 찾을 수 없습니다." 와 같은 메시지 반환 고려

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

        # 프롬프트 모듈에서 적절한 프롬프트 가져오기
        if target_index == "code":
            prompt = prompts.get_code_rag_prompt(context_for_rag, search_query)
        else:
            prompt = prompts.get_document_rag_prompt(context_for_rag, search_query)

        logger.info(
            f"\n'{llm_model_name}' 모델을 사용하여 RAG 답변 생성을 시작합니다..."
        )
        response = client.models.generate_content(model=llm_model_name, contents=prompt)

        logger.info("RAG 답변 생성 완료.")
        return response.text

    except EmbeddingError as e_embed_query_fail: # 이 예외는 현재 코드에서 발생하지 않을 수 있음 (주로 GeminiAPIEmbeddings 클래스에서 발생)
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