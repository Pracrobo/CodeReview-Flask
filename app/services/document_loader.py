import os
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Config, LANGUAGE_TO_DETAILS

logger = logging.getLogger(__name__)


class DocumentLoader:
    """문서 로딩 및 텍스트 분할 처리 클래스"""

    def __init__(self):
        """문서 로더 초기화"""
        self.encoding = "utf-8"

    def load_documents_from_directory(
        self,
        directory_path: str,
        file_extensions: Tuple[str, ...],
        max_depth: Optional[int] = None,
    ) -> List[Document]:
        """디렉토리에서 특정 확장자의 문서들을 로드합니다.

        Args:
            directory_path: 탐색할 디렉토리 경로
            file_extensions: 로드할 파일 확장자 튜플
            max_depth: 최대 탐색 깊이 (None이면 제한 없음)

        Returns:
            로드된 Document 객체 리스트
        """
        documents = []
        root_path = Path(directory_path).resolve()

        for file_path in self._find_files_by_extension(
            root_path, file_extensions, max_depth
        ):
            document = self._load_single_document(file_path)
            if document:
                documents.append(document)

        logger.info(
            f"디렉토리 '{directory_path}'에서 {len(documents)}개 문서 로드 완료 "
            f"(확장자: {file_extensions})"
        )

        return documents

    def _find_files_by_extension(
        self, root_path: Path, extensions: Tuple[str, ...], max_depth: Optional[int]
    ) -> List[Path]:
        """확장자별로 파일을 찾습니다.

        Args:
            root_path: 루트 경로
            extensions: 파일 확장자들
            max_depth: 최대 깊이

        Returns:
            찾은 파일 경로 리스트
        """
        found_files = []

        for current_path, dirs, files in os.walk(root_path):
            current_depth = len(Path(current_path).relative_to(root_path).parts)

            # 최대 깊이 제한 확인
            if max_depth is not None and current_depth > max_depth:
                dirs.clear()  # 더 깊은 디렉토리 탐색 중단
                continue

            for file_name in files:
                if file_name.endswith(extensions):
                    found_files.append(Path(current_path) / file_name)

        return found_files

    def _load_single_document(self, file_path: Path) -> Optional[Document]:
        """단일 문서를 로드합니다.

        Args:
            file_path: 파일 경로

        Returns:
            Document 객체 또는 None (로드 실패시)
        """
        try:
            with open(file_path, "r", encoding=self.encoding, errors="ignore") as f:
                content = f.read()

            return Document(page_content=content, metadata={"source": str(file_path)})

        except Exception as e:
            logger.warning(f"파일 로드 실패: {file_path} - {e}")
            return None

    def split_documents_by_language(
        self, documents: List[Document], language_name: str
    ) -> List[Document]:
        """언어별로 문서를 분할합니다.

        Args:
            documents: 분할할 문서 리스트
            language_name: 프로그래밍 언어 이름

        Returns:
            분할된 문서 리스트
        """
        if not documents:
            return []

        language_details = LANGUAGE_TO_DETAILS.get(language_name.lower())

        if language_details:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_details["lang_enum"],
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            )
        else:
            # 일반 텍스트 분할기 사용
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            )

        split_documents = text_splitter.split_documents(documents)

        logger.info(
            f"{len(documents)}개 문서를 {len(split_documents)}개 청크로 분할 완료 "
            f"(언어: {language_name})"
        )

        return split_documents

    def split_documents_as_text(self, documents: List[Document]) -> List[Document]:
        """문서를 일반 텍스트로 분할합니다.

        Args:
            documents: 분할할 문서 리스트

        Returns:
            분할된 문서 리스트
        """
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        split_documents = text_splitter.split_documents(documents)

        logger.info(
            f"{len(documents)}개 문서를 {len(split_documents)}개 청크로 분할 완료 "
            f"(일반 텍스트)"
        )

        return split_documents

    def get_code_file_extension(self, language_name: str) -> Optional[str]:
        """언어 이름으로부터 파일 확장자를 가져옵니다.

        Args:
            language_name: 프로그래밍 언어 이름

        Returns:
            파일 확장자 또는 None
        """
        language_details = LANGUAGE_TO_DETAILS.get(language_name.lower())
        return language_details["ext"] if language_details else None

    def is_supported_language(self, language_name: str) -> bool:
        """지원되는 언어인지 확인합니다.

        Args:
            language_name: 프로그래밍 언어 이름

        Returns:
            지원 여부
        """
        return language_name.lower() in LANGUAGE_TO_DETAILS
