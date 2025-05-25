import os
import logging  # 로깅 모듈 임포트

from config import Config  # 설정 파일 임포트
from .indexer import (
    create_index_from_repo,
)
from .searcher import search_and_rag
from common.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
    RAGError,
)

logger = logging.getLogger(__name__)  # 현재 모듈에 대한 로거 생성

if __name__ == "__main__":
    # 기본 로깅 레벨 설정 (필요한 경우)
    logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
    logger.info(f"임베딩 모델: {Config.DEFAULT_EMBEDDING_MODEL}")
    logger.info(f"LLM 모델: {Config.DEFAULT_LLM_MODEL}")

    repo_to_index = input(
        "인덱싱하고 검색할 GitHub 저장소 URL을 입력하세요 (예: https://github.com/pallets/flask): "
    ).strip()
    if not repo_to_index:
        logger.info(
            "저장소 URL이 입력되지 않았습니다. 기본 URL 'https://github.com/pallets/flask'을 사용합니다."
        )
        repo_to_index = "https://github.com/pallets/flask"

    repo_name_from_url = repo_to_index.split("/")[-1].removesuffix(".git")

    os.makedirs(Config.BASE_CLONED_DIR, exist_ok=True)
    dynamic_local_repo_path = os.path.join(Config.BASE_CLONED_DIR, repo_name_from_url)
    logger.info(f"선택된 저장소: {repo_to_index}")
    logger.info(f"로컬 복제 경로: {dynamic_local_repo_path}")

    vector_stores = None  # vector_stores 초기화
    try:
        vector_stores = create_index_from_repo(
            repo_url=repo_to_index,
            local_repo_path=dynamic_local_repo_path,
            embedding_model_name=Config.DEFAULT_EMBEDDING_MODEL,
        )

        if vector_stores:
            if vector_stores.get("code"):
                logger.info("코드 인덱스가 성공적으로 생성/로드되었습니다.")
                query_code = input(
                    "\n코드 관련 질문을 입력하세요 (종료하려면 Enter): "
                ).strip()
                if query_code:
                    rag_response_code = search_and_rag(
                        vector_stores=vector_stores,
                        target_index="code",
                        search_query=query_code,
                        llm_model_name=Config.DEFAULT_LLM_MODEL,
                        top_k=Config.DEFAULT_TOP_K,
                        similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,  # 임계값 전달
                    )
                    if rag_response_code:
                        logger.info(f"\n코드 RAG 답변:\n{rag_response_code}")
            else:
                logger.warning("코드 인덱스를 사용할 수 없습니다.")

            if vector_stores.get("document"):
                logger.info("문서 인덱스가 성공적으로 생성/로드되었습니다.")
                query_doc = input(
                    "\n문서 관련 질문을 입력하세요 (종료하려면 Enter): "
                ).strip()
                if query_doc:
                    rag_response_doc = search_and_rag(
                        vector_stores=vector_stores,
                        target_index="document",
                        search_query=query_doc,
                        llm_model_name=Config.DEFAULT_LLM_MODEL,
                        top_k=Config.DEFAULT_TOP_K,
                        similarity_threshold=Config.DEFAULT_SIMILARITY_THRESHOLD,  # 임계값 전달
                    )
                    if rag_response_doc:
                        logger.info(f"\n문서 RAG 답변:\n{rag_response_doc}")
            else:
                logger.warning("문서 인덱스를 사용할 수 없습니다.")
        else:
            logger.error("벡터 저장소(인덱스)를 생성하거나 로드하지 못했습니다.")

    except RepositorySizeError as e:
        logger.error(f"저장소 크기 초과: {e}")
    except RepositoryError as e:
        logger.error(f"저장소 관련 오류 발생: {e}")
    except IndexingError as e:
        logger.error(f"인덱싱 과정에서 오류 발생: {e}")
    except EmbeddingError as e:  # 임베딩 과정에서의 오류 처리
        logger.error(f"임베딩 생성 중 오류 발생: {e}")
    except RAGError as e:  # RAG 과정에서의 오류 처리
        logger.error(f"검색 또는 RAG 처리 중 오류 발생: {e}")
    except Exception as e:
        logger.error(
            f"예상치 못한 오류 발생: {e}", exc_info=True
        )  # 스택 트레이스 포함 로깅
    finally:
        logger.info("\n스크립트가 종료되었습니다.")
