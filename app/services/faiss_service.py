import os
import logging
from typing import List, Optional, Tuple

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.utils import ensure_directory_exists
from app.core.exceptions import IndexingError, EmbeddingError
from .embeddings import GeminiAPIEmbeddings

# FAISS 스레드 설정
faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)


class FAISSService:
    """FAISS 인덱스 생성 및 관리 서비스"""

    def __init__(self, embeddings: GeminiAPIEmbeddings):
        """FAISS 서비스 초기화

        Args:
            embeddings: 임베딩 생성 객체
        """
        self.embeddings = embeddings

    def create_index_from_documents(
        self, documents: List[Document], index_path: str, index_type: str
    ) -> Optional[FAISS]:
        """문서 리스트로부터 FAISS 인덱스를 생성합니다.

        Args:
            documents: 인덱싱할 문서 리스트
            index_path: 인덱스 저장 경로
            index_type: 인덱스 타입 ("code")

        Returns:
            생성된 FAISS 벡터 스토어 또는 None

        Raises:
            IndexingError: 인덱스 생성 실패
        """
        if not documents:
            logger.warning(f"{index_type} 인덱싱을 위한 문서가 없습니다.")
            return None

        logger.info(f"{index_type} FAISS 인덱스 생성 시작: {index_path}")

        try:
            # 임베딩 생성
            embeddings_data, failed_indices = self._generate_embeddings(documents)

            if not embeddings_data:
                logger.warning("모든 문서의 임베딩 생성에 실패했습니다.")
                return None

            # 성공한 문서들로 인덱스 생성
            successful_documents = self._filter_successful_documents(
                documents, failed_indices
            )
            vector_store = self._create_faiss_vector_store(
                successful_documents, embeddings_data
            )

            # 인덱스 저장
            self._save_index(vector_store, index_path)

            logger.info(
                f"{index_type} FAISS 인덱스 생성 완료 "
                f"(성공: {len(successful_documents)}, 실패: {len(failed_indices)})"
            )

            return vector_store

        except Exception as e:
            logger.error(f"{index_type} 인덱스 생성 중 오류: {e}", exc_info=True)
            raise IndexingError(f"{index_type} 인덱스 생성 실패") from e

    def load_index(self, index_path: str, index_type: str) -> Optional[FAISS]:
        """저장된 FAISS 인덱스를 로드합니다.

        Args:
            index_path: 인덱스 경로
            index_type: 인덱스 타입 ("code")

        Returns:
            로드된 FAISS 벡터 스토어 또는 None
        """
        if not os.path.exists(index_path):
            logger.info(f"{index_type} 인덱스가 존재하지 않습니다: {index_path}")
            return None

        try:
            logger.info(f"{index_type} FAISS 인덱스 로드: {index_path}")
            return FAISS.load_local(
                index_path, self.embeddings, allow_dangerous_deserialization=True
            )

        except Exception as e:
            logger.error(f"{index_type} 인덱스 로드 실패: {e}")
            return None

    def _generate_embeddings(self, documents: List[Document]) -> Tuple[List, List[int]]:
        """문서들의 임베딩을 생성합니다.

        Args:
            documents: 문서 리스트

        Returns:
            (성공한_임베딩_리스트, 실패한_인덱스_리스트) 튜플

        Raises:
            EmbeddingError: 임베딩 생성 실패
        """
        try:
            logger.info(f"총 {len(documents)}개 문서의 임베딩 생성을 시작합니다...")
            document_contents = [doc.page_content for doc in documents]

            # 임베딩 생성 시도
            embeddings_result = self.embeddings.embed_documents(document_contents)

            if isinstance(embeddings_result, tuple) and len(embeddings_result) == 2:
                embeddings_data, failed_indices = embeddings_result
                logger.info(
                    f"임베딩 생성 완료: 성공 {len(embeddings_data)}개, 실패 {len(failed_indices)}개"
                )
                return embeddings_data, failed_indices
            else:
                # 예상치 못한 반환 형식
                logger.error(f"예상치 못한 임베딩 결과 형식: {type(embeddings_result)}")
                raise EmbeddingError("임베딩 결과 형식이 올바르지 않습니다.")

        except EmbeddingError as e:
            # EmbeddingError는 이미 상세한 메시지를 포함하고 있음
            logger.error(f"문서 임베딩 생성 중 오류: {e}")

            # 에러 메시지에 해결 방법이 포함되어 있는지 확인
            error_message = str(e)
            if "해결 방법:" not in error_message:
                # 기본 해결 방법 추가
                error_message += "\n\n추가 해결 방법:\n"
                error_message += "1. Flask 서버를 재시작해보세요.\n"
                error_message += "2. .env 파일의 API 키가 올바른지 확인하세요.\n"
                error_message += "3. 네트워크 연결을 확인하세요."

            raise EmbeddingError(error_message) from e

        except Exception as e:
            logger.error(f"예상치 못한 임베딩 오류: {e}", exc_info=True)

            # 일반적인 오류에 대한 상세한 메시지 구성
            error_details = [
                f"임베딩 생성 중 예상치 못한 오류가 발생했습니다: {e}",
                "",
                "가능한 원인:",
                "1. 네트워크 연결 문제",
                "2. Gemini API 서비스 장애",
                "3. API 키 권한 문제",
                "4. 요청 데이터 형식 문제",
                "",
                "해결 방법:",
                "1. 네트워크 연결을 확인하세요.",
                "2. 잠시 후 다시 시도하세요.",
                "3. API 키가 유효하고 권한이 있는지 확인하세요.",
                "4. 문제가 지속되면 로그를 확인하세요.",
            ]

            raise EmbeddingError("\n".join(error_details)) from e

    def _filter_successful_documents(
        self, documents: List[Document], failed_indices: List[int]
    ) -> List[Document]:
        """성공한 임베딩에 해당하는 문서들만 필터링합니다.

        Args:
            documents: 전체 문서 리스트
            failed_indices: 실패한 인덱스 리스트

        Returns:
            성공한 문서들의 리스트
        """
        failed_set = set(failed_indices)
        return [doc for i, doc in enumerate(documents) if i not in failed_set]

    def _create_faiss_vector_store(
        self, documents: List[Document], embeddings_data: List
    ) -> FAISS:
        """FAISS 벡터 스토어를 생성합니다.

        Args:
            documents: 문서 리스트
            embeddings_data: 임베딩 데이터 리스트

        Returns:
            생성된 FAISS 벡터 스토어

        Raises:
            IndexingError: 벡터 스토어 생성 실패
        """
        try:
            # 텍스트-임베딩 쌍 생성
            text_embedding_pairs = [
                (doc.page_content, embedding)
                for doc, embedding in zip(documents, embeddings_data)
            ]

            # 메타데이터 추출
            metadatas = [doc.metadata for doc in documents]

            # FAISS 벡터 스토어 생성
            return FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=self.embeddings,
                metadatas=metadatas,
            )

        except Exception as e:
            logger.error(f"FAISS 벡터 스토어 생성 실패: {e}")
            raise IndexingError("벡터 스토어 생성 실패") from e

    def _save_index(self, vector_store: FAISS, index_path: str) -> None:
        """FAISS 인덱스를 저장합니다.

        Args:
            vector_store: 저장할 벡터 스토어
            index_path: 저장 경로

        Raises:
            IndexingError: 저장 실패
        """
        try:
            # 디렉토리 생성
            ensure_directory_exists(os.path.dirname(index_path))

            # 인덱스 저장
            vector_store.save_local(index_path)
            logger.info(f"FAISS 인덱스 저장 완료: {index_path}")

        except Exception as e:
            logger.error(f"FAISS 인덱스 저장 실패: {e}")
            raise IndexingError("인덱스 저장 실패") from e

    def delete_index(self, index_path: str) -> bool:
        """FAISS 인덱스를 삭제합니다.

        Args:
            index_path: 삭제할 인덱스 경로

        Returns:
            삭제 성공 여부
        """
        try:
            if os.path.exists(index_path):
                import shutil

                shutil.rmtree(index_path)
                logger.info(f"FAISS 인덱스 삭제 완료: {index_path}")
                return True
            return False

        except Exception as e:
            logger.error(f"FAISS 인덱스 삭제 실패: {e}")
            return False

    def get_index_stats(self, vector_store: FAISS) -> dict:
        """FAISS 인덱스의 통계 정보를 반환합니다.

        Args:
            vector_store: FAISS 벡터 스토어

        Returns:
            인덱스 통계 딕셔너리
        """
        try:
            index = vector_store.index
            return {
                "total_vectors": index.ntotal,
                "dimension": index.d,
                "is_trained": index.is_trained,
                "metric_type": "L2" if hasattr(index, "metric_type") else "unknown",
            }

        except Exception as e:
            logger.warning(f"인덱스 통계 수집 실패: {e}")
            return {}
