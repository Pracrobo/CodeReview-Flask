import logging
import numpy as np
import asyncio  # asyncio 임포트

import faiss

from app.core.config import Config
from app.core.exceptions import EmbeddingError, RAGError
from app.core.prompts import prompts
from app.services.gemini_service import gemini_service
from app.services.issue_analyzer import issue_analyzer

faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)


async def translate_code_query_to_english(
    korean_text, llm_model_name
):  # async def로 변경
    """코드 관련 한국어 질의 영어 번역"""
    try:
        client = gemini_service.get_client()
    except Exception:
        logger.error("Gemini 클라이언트가 초기화되지 않아 번역을 건너뜁니다.")
        return korean_text
    try:
        prompt = prompts.get_code_query_translation_prompt(korean_text)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, client.models.generate_content, model=llm_model_name, contents=prompt
        )
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


def normalize_vector(vec):
    """벡터 정규화"""
    v = np.array(vec)
    norm = np.linalg.norm(v) + 1e-8
    return v / norm


def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    return float(np.dot(v1, v2))


async def analyze_issue_and_generate_insights(
    vector_stores,
    issue_data,  # {'title': str, 'body': str, 'issueId': int}
):
    """이슈 분석 및 AI 인사이트 생성"""
    try:
        result = await issue_analyzer.analyze_issue(vector_stores, issue_data)
        return result
    except Exception as e:
        logger.error(f"이슈 분석 중 오류 발생: {e}", exc_info=True)
        raise RAGError(f"이슈 분석 실패: {e}") from e


def search_and_rag(
    vector_stores,
    target_index,  # "code"만 지원
    search_query,
    llm_model_name,
    top_k=Config.DEFAULT_TOP_K,
    similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,
):
    """코드 벡터 저장소 검색 및 LLM 기반 답변 생성 (RAG, 코사인 유사도 직접 적용)"""
    # 이 함수가 비동기 컨텍스트(예: FastAPI 엔드포인트)에서 직접 호출된다면
    # async def search_and_rag(...): 로 변경하고, 내부의 run_in_executor 호출은 await을 사용해야 합니다.
    # 현재는 동기 함수로 가정하고, 내부에서 실행되는 블로킹 I/O를 run_in_executor로 감싸는 것은
    # 이 함수 자체가 이벤트 루프를 블록하는 것을 막지는 못합니다.
    # 가장 좋은 방법은 이 함수를 async def로 만들고, 호출하는 곳에서 await하는 것입니다.
    # 여기서는 내부 Gemini 호출만 수정합니다.
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

    vector_store = vector_stores[target_index]  # target_index는 "code"

    logger.info("사용자 질의를 영어로 번역 중...")
    # translate_code_query_to_english는 이제 async 함수이므로,
    # 이 함수를 호출하는 search_and_rag도 async여야 하고 await으로 호출해야 합니다.
    # 임시로 asyncio.run을 사용하지만, 이상적인 구조는 아닙니다.
    # search_and_rag를 async def로 변경하는 것이 좋습니다.
    try:
        english_query = asyncio.run(
            translate_code_query_to_english(search_query, llm_model_name)
        )
    except RuntimeError:  # 이미 이벤트 루프가 실행 중인 경우
        loop = asyncio.get_event_loop()
        english_query = loop.run_until_complete(
            translate_code_query_to_english(search_query, llm_model_name)
        )

    # 2. 임베딩 입력 전처리
    processed_query = preprocess_text(english_query)

    logger.info(
        f"\n인덱스에 대한 의미론적 검색을 시작합니다 (Top K: {top_k}, 유사도 임계값: {similarity_threshold})..."
    )

    try:
        # 쿼리 임베딩 추출 - embed_query가 동기 I/O 바운드일 수 있음
        # 이 부분도 run_in_executor로 감싸는 것을 고려해야 하지만, 우선 LLM 호출부터 수정합니다.
        # loop = asyncio.get_event_loop()
        # query_embedding = await loop.run_in_executor(None, vector_store.embedding_function.embed_query, processed_query)
        query_embedding = vector_store.embedding_function.embed_query(
            processed_query
        )  # 일단 유지

        # FAISS 인덱스의 실제 벡터 개수
        num_vectors = vector_store.index.ntotal
        doc_score_pairs = []
        for idx in range(num_vectors):
            docstore_id = vector_store.index_to_docstore_id[idx]
            doc = vector_store.docstore._dict[docstore_id]
            doc_embedding = vector_store.index.reconstruct(idx)
            sim = cosine_similarity(query_embedding, doc_embedding)
            doc_score_pairs.append((doc, sim))
        # 4. Top-K 상위 결과만 추출
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_results = doc_score_pairs[:top_k]

        # 6. 임계값 동적 완화
        filtered_results = [
            (doc, score) for doc, score in top_results if score >= similarity_threshold
        ]
        if not filtered_results and similarity_threshold > 0.1:
            logger.info("유사도 임계값을 자동 완화합니다.")
            filtered_results = [
                (doc, score) for doc, score in top_results if score >= 0.1
            ]

        logger.info(
            f"\n'{search_query}'에 대한 검색 결과 (총 {len(top_results)}개 중 {len(filtered_results)}개가 임계값 통과, 임계값: {similarity_threshold:.2f}):"
        )

        if not filtered_results:
            logger.info("유사도 임계값을 만족하는 검색 결과가 없습니다.")
            return "유사도가 충분히 높은 검색 결과가 없습니다."

        for i, (doc, score) in enumerate(filtered_results):
            logger.info(
                f"  {i+1}. 유사도: {score*100:.1f}%, 소스: {doc.metadata.get('source', '알 수 없음')}, 내용 (일부): {doc.page_content[:100]}..."
            )

        context_for_rag = "\n\n".join(
            [doc.page_content for doc, score in filtered_results]
        )

        prompt = prompts.get_code_rag_prompt(context_for_rag, search_query)

        logger.info(
            f"\n'{llm_model_name}' 모델을 사용하여 RAG 답변 생성을 시작합니다..."
        )
        # response = client.models.generate_content(model=llm_model_name, contents=prompt)
        # 이 함수가 async가 아니므로 loop를 직접 사용하기 어렵습니다.
        # 만약 search_and_rag가 async def라면 아래와 같이 수정:
        # loop = asyncio.get_event_loop()
        # response = await loop.run_in_executor(None, client.models.generate_content, model=llm_model_name, contents=prompt)
        # 현재는 동기 함수이므로, 이 호출은 그대로 두지만, 이것이 블로킹의 원인이 될 수 있습니다.
        # 이상적으로는 search_and_rag 함수를 async def로 변경해야 합니다.
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
