import logging

import faiss

from app.core.config import Config
from app.core.exceptions import EmbeddingError, RAGError
from app.core.prompts import prompts
from app.services.gemini_service import gemini_service
from app.services.issue_analyzer import issue_analyzer
from app.services.faiss_service import FAISSService  # FAISSService 임포트

faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)


def translate_code_query_to_english(korean_text, llm_model_name):
    """코드 관련 한국어 질의 영어 번역"""
    try:
        client = gemini_service.get_client()
    except Exception:
        logger.error("Gemini 클라이언트가 초기화되지 않아 번역을 건너뜁니다.")
        return korean_text
    try:
        prompt = prompts.get_code_query_translation_prompt(korean_text)
        response = client.models.generate_content(model=llm_model_name, contents=prompt)
        english_text = gemini_service.extract_text_from_response(response)
        logger.info(f"코드 질의 번역 완료: '{korean_text}' -> '{english_text}'")
        return english_text if english_text else korean_text
    except Exception as e:
        logger.warning(f"코드 질의 번역 실패, 원본 텍스트 사용: {e}")
        return korean_text


def preprocess_text(text):
    """임베딩 입력 전처리(공백, 특수문자, 소문자 변환 등)"""
    import re

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text


def analyze_issue_and_generate_insights(
    vector_stores,
    issue_data,
):
    """이슈 분석 및 AI 인사이트 생성"""
    try:

        from app.services.embeddings import GeminiAPIEmbeddings

        embeddings_instance = GeminiAPIEmbeddings(
            model_name=Config.DEFAULT_EMBEDDING_MODEL,
            document_task_type="RETRIEVAL_DOCUMENT",
            query_task_type="RETRIEVAL_QUERY",
        )
        faiss_service_instance = FAISSService(embeddings=embeddings_instance)
        result = issue_analyzer.analyze_issue(
            vector_stores, issue_data, faiss_service_instance
        )
        return result
    except Exception as e:
        logger.error(f"이슈 분석 중 오류 발생: {e}", exc_info=True)
        raise RAGError(f"이슈 분석 실패: {e}") from e


def search_and_rag(
    vector_stores,
    target_index,  # "code"만 지원
    search_query,
    llm_model_name,
    faiss_service: FAISSService,  # faiss_service 파라미터 추가
    top_k=Config.DEFAULT_TOP_K,
    similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
):
    """코드 벡터 저장소 검색 및 LLM 기반 답변 생성 (RAG, FAISSService 사용)"""
    try:
        client = gemini_service.get_client()
    except Exception:
        raise RAGError("Gemini 클라이언트가 초기화되지 않아 RAG를 수행할 수 없습니다.")

    if (
        target_index != "code"
        or target_index not in vector_stores
        or not vector_stores[target_index]
    ):
        logger.warning(
            f"'{target_index}' 벡터 저장소를 사용할 수 없거나 지원되지 않는 타입입니다. 코드 검색만 지원됩니다."
        )
        return None

    vector_store = vector_stores[target_index]

    logger.info("사용자 질의를 영어로 번역 중...")
    try:
        english_query = translate_code_query_to_english(search_query, llm_model_name)
    except Exception as e_translate:
        logger.error(f"질의 번역 중 오류 발생: {e_translate}")
        english_query = search_query

    processed_query = preprocess_text(english_query)

    logger.info(
        f"\nFAISSService를 사용하여 의미론적 검색을 시작합니다 (Top K: {top_k}, 유사도 임계값: {similarity_threshold})..."
    )

    try:
        # FAISSService의 search_documents 사용
        # search_documents는 동기 함수이므로 직접 호출
        filtered_results_with_scores = faiss_service.search_documents(
            vector_store=vector_store,
            query_text=processed_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        logger.info(
            f"\n'{search_query}'에 대한 검색 결과 ({len(filtered_results_with_scores)}개가 임계값 통과, 임계값: {similarity_threshold:.2f}):"
        )

        if not filtered_results_with_scores:
            logger.info("유사도 임계값을 만족하는 검색 결과가 없습니다.")
            return "유사도가 충분히 높은 검색 결과가 없습니다."

        for i, (doc, score) in enumerate(filtered_results_with_scores):
            logger.info(
                f"  {i+1}. 유사도: {score*100:.1f}%, 소스: {doc.metadata.get('source', '알 수 없음')}, 내용 (일부): {doc.page_content[:100]}..."
            )

        context_for_rag = "\n\n".join(
            [doc.page_content for doc, score in filtered_results_with_scores]
        )

        prompt = prompts.get_code_rag_prompt(context_for_rag, search_query)

        logger.info(
            f"\n'{llm_model_name}' 모델을 사용하여 RAG 답변 생성을 시작합니다..."
        )
        response = client.models.generate_content(model=llm_model_name, contents=prompt)

        logger.info("RAG 답변 생성 완료.")
        answer_text = gemini_service.extract_text_from_response(response)
        return answer_text

    except EmbeddingError as e_embed_query_fail:
        logger.error(f"인덱스에 대한 쿼리 임베딩 중 오류 발생: {e_embed_query_fail}")
        logger.error("유사도 검색 및 RAG를 진행할 수 없습니다.")
        return "인덱스 검색 중 오류가 발생했습니다: 임베딩 실패"
    except Exception as e_rag:
        logger.error(
            f"유사도 검색 또는 RAG 중 예기치 않은 오류 발생: {e_rag}",
            exc_info=True,
        )
        raise RAGError(f"검색 또는 RAG 처리 중 오류 발생: {e_rag}") from e_rag
