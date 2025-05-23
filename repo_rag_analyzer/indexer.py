import os
import time
import logging
import shutil
from typing import Dict, List, Optional, Tuple, Generator

import requests
from git import Repo, GitCommandError
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# FAISS CPU 전용 설정
import faiss

from config import Config, LANGUAGE_TO_DETAILS
from .embeddings import GeminiAPIEmbeddings
from common.exceptions import (
    RepositoryError,
    IndexingError,
    RepositorySizeError,
    EmbeddingError,
)

faiss.omp_set_num_threads(1)  # CPU 스레드 수 제한으로 안정성 향상


# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


def get_repo_primary_language(
    repo_url: str, token: Optional[str] = None
) -> Tuple[str, int]:
    """GitHub API를 사용하여 저장소의 주 사용 언어와 총 코드 바이트 수를 가져옵니다."""
    try:
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"잘못된 GitHub URL입니다: {repo_url}")

        parts = repo_url.split("/")
        if len(parts) < 5:
            raise ValueError(
                f"URL에서 소유자/저장소 이름을 파싱할 수 없습니다: {repo_url}"
            )
        owner, repo_name_ext = parts[-2], parts[-1]
        repo_name = repo_name_ext.removesuffix(".git")

        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        languages = response.json()

        if not languages:
            logger.warning(
                f"{repo_url}에 대한 언어 데이터를 찾을 수 없습니다. 기본값으로 'unknown'을 사용합니다."
            )
            return "unknown", 0  # 언어 데이터가 없는 경우

        # 총 바이트 수 계산
        total_bytes = sum(languages.values())
        primary_language = max(languages, key=languages.get)

        logger.info(f"{repo_url}의 주 사용 언어 감지: {primary_language}")
        logger.info(
            f"총 코드 바이트 수: {total_bytes:,} bytes ({total_bytes / (1024*1024):.1f} MB)"
        )

        # 큰 저장소 검증 (3MB 초과 시 중단)
        max_bytes = 3 * 1024 * 1024  # 3MB
        if total_bytes > max_bytes:
            error_msg = f"저장소가 너무 큽니다. 총 코드 크기: {total_bytes:,} bytes ({total_bytes / (1024*1024):.1f} MB), 최대 허용: {max_bytes / (1024*1024):.1f} MB"
            logger.error(error_msg)
            raise RepositorySizeError(error_msg)

        return primary_language.lower(), total_bytes
    except RepositorySizeError:
        # 크기 오류는 그대로 재발생
        raise
    except requests.exceptions.RequestException as e:
        raise RepositoryError(f"{repo_url}의 저장소 언어 가져오기 오류: {e}") from e
    except ValueError as e:
        raise RepositoryError(f"입력 값 오류: {e}") from e
    except Exception as e:  # 예상치 못한 기타 예외
        raise RepositoryError(
            f"{repo_url}의 언어 가져오기 중 예상치 못한 오류 발생: {e}"
        ) from e


def clone_repo(repo_url: str, local_path: str) -> Repo:
    """Git 저장소를 로컬 경로에 복제하거나 이미 존재하면 로드합니다."""
    if os.path.exists(local_path):
        logger.info(f"기존 저장소를 {local_path} 에서 로드합니다.")
        try:
            return Repo(local_path)
        except GitCommandError as e:
            logger.warning(f"기존 저장소 로드 실패: {e}. 저장소를 새로 복제합니다.")
            shutil.rmtree(local_path)  # 기존 디렉토리 삭제 후 새로 복제

    logger.info(f"{repo_url}을(를) {local_path}에 복제합니다...")
    try:
        return Repo.clone_from(repo_url, local_path)
    except GitCommandError as e:
        raise RepositoryError(f"저장소 복제 실패: {e}") from e


def load_documents_from_path(
    path: str, file_extensions: Tuple[str, ...], encoding: str = "utf-8"
) -> List[Document]:
    """지정된 경로에서 특정 확장자를 가진 파일들을 로드하여 Document 객체 리스트로 반환합니다."""
    docs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(file_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                        content = f.read()
                    docs.append(
                        Document(page_content=content, metadata={"source": file_path})
                    )
                except Exception as e:
                    logger.warning(f"{file_path} 파일 로드 중 오류 발생: {e}")
    logger.info(
        f"{path} 경로에서 {len(docs)}개의 문서를 로드했습니다 ({file_extensions})."
    )
    return docs


def process_files_in_chunks(
    file_paths: List[str], chunk_size: int = 100
) -> Generator[List[str], None, None]:
    """파일 경로 리스트를 받아 지정된 크기의 청크로 나누어 반환하는 제너레이터입니다."""
    logger.info(f"총 {len(file_paths)}개의 파일을 {chunk_size}개씩 묶어 처리합니다.")
    for i in range(0, len(file_paths), chunk_size):
        yield file_paths[i : i + chunk_size]


def create_faiss_index(
    docs: List[Document],
    embeddings: GeminiAPIEmbeddings,
    index_path: str,
    index_type: str,  # "code" 또는 "document"
) -> Optional[FAISS]:
    """주어진 문서들과 임베딩 모델을 사용하여 FAISS 인덱스를 생성하고 저장합니다."""
    if not docs:
        logger.warning(
            f"{index_type} 인덱싱을 위한 문서가 없습니다. 인덱스 생성을 건너뜁니다."
        )
        return None

    logger.info(f"{index_type}에 대한 FAISS 인덱스를 {index_path}에 생성합니다...")
    try:
        # FAISS.from_documents는 내부적으로 배치 처리를 할 수 있지만,
        # GeminiAPIEmbeddings에서 이미 배치 및 재시도 로직을 처리하므로 여기서는 그대로 사용합니다.
        # 만약 GeminiAPIEmbeddings.embed_documents가 None을 반환하는 경우가 있다면,
        # 해당 문서는 FAISS 인덱스 생성 시 제외될 수 있습니다.
        # FAISS.from_documents는 유효한 임베딩만 사용합니다.
        vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
        vector_store.save_local(index_path)
        logger.info(
            f"{index_type} FAISS 인덱스를 {index_path}에 성공적으로 저장했습니다."
        )
        return vector_store
    except EmbeddingError as e:
        logger.error(f"{index_type} 인덱스 생성 중 임베딩 오류 발생: {e}")
        raise IndexingError(f"{index_type} 인덱스 생성 실패 (임베딩 오류)") from e
    except Exception as e:
        logger.error(f"{index_type} FAISS 인덱스 생성 또는 저장 중 오류 발생: {e}")
        raise IndexingError(f"{index_type} 인덱스 생성 실패") from e


def load_faiss_index(
    index_path: str, embeddings: GeminiAPIEmbeddings, index_type: str
) -> Optional[FAISS]:
    """저장된 FAISS 인덱스를 로드합니다."""
    if os.path.exists(index_path):
        logger.info(f"기존 {index_type} FAISS 인덱스를 {index_path} 에서 로드합니다...")
        try:
            return FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(
                f"{index_type} FAISS 인덱스 로드 중 오류 발생: {e}. 새로 생성합니다."
            )
            return None
    return None


def create_index_from_repo(
    repo_url: str,
    local_repo_path: str,
    embedding_model_name: str,
) -> Dict[str, Optional[FAISS]]:
    """저장소 URL로부터 코드 및 문서에 대한 FAISS 인덱스를 생성합니다."""
    overall_start_time = time.time()
    vector_stores: Dict[str, Optional[FAISS]] = {"code": None, "document": None}

    try:
        # 저장소 복제 전에 주 사용 언어와 크기 검증
        logger.info("--- 저장소 정보 확인 중 ---")
        primary_language_name, total_code_bytes = get_repo_primary_language(
            repo_url, Config.GITHUB_API_TOKEN
        )

        # 저장소 복제 또는 로드
        clone_repo(repo_url, local_repo_path)
        repo_name_for_path = os.path.basename(local_repo_path.rstrip("/\\"))

        # 임베딩 모델 초기화 (통합)
        lc_gemini_embeddings = GeminiAPIEmbeddings(
            model_name=embedding_model_name,
            document_task_type="RETRIEVAL_DOCUMENT",  # 문서 임베딩 시
            query_task_type="RETRIEVAL_QUERY",  # 검색 쿼리 임베딩 시
        )

        # --- 코드 인덱싱 ---
        logger.info("--- 코드 인덱싱 시작 ---")
        lang_details = LANGUAGE_TO_DETAILS.get(primary_language_name)

        if not lang_details:
            logger.error(
                f"지원되지 않는 언어: {primary_language_name}. 코드 인덱싱을 건너뜁니다."
            )
        else:
            code_file_extension = lang_details["ext"]
            langchain_language_enum = lang_details["lang_enum"]
            logger.info(
                f"{primary_language_name} ({code_file_extension}) 코드 파일을 처리합니다."
            )

            faiss_code_index_path = os.path.join(
                Config.FAISS_INDEX_BASE_DIR, repo_name_for_path + "_code"
            )
            os.makedirs(Config.FAISS_INDEX_BASE_DIR, exist_ok=True)

            vector_stores["code"] = load_faiss_index(
                faiss_code_index_path, lc_gemini_embeddings, "code"
            )

            if not vector_stores["code"]:
                logger.info(
                    f"'{local_repo_path}'에서 코드 파일을 로드합니다 (확장자: {code_file_extension})..."
                )
                code_docs = load_documents_from_path(
                    local_repo_path, (code_file_extension,)
                )

                if code_docs:
                    text_splitter_code = RecursiveCharacterTextSplitter.from_language(
                        language=langchain_language_enum,
                        chunk_size=Config.CHUNK_SIZE,
                        chunk_overlap=Config.CHUNK_OVERLAP,
                    )
                    split_code_docs = text_splitter_code.split_documents(code_docs)
                    logger.info(
                        f"{len(code_docs)}개의 코드 파일을 {len(split_code_docs)}개의 청크로 분할했습니다."
                    )
                    vector_stores["code"] = create_faiss_index(
                        split_code_docs,
                        lc_gemini_embeddings,
                        faiss_code_index_path,
                        "code",
                    )
                else:
                    logger.warning("인덱싱할 코드 파일을 찾지 못했습니다.")
        logger.info("--- 코드 인덱싱 종료 ---")

        # --- 문서 인덱싱 ---
        logger.info("--- 문서 인덱싱 시작 ---")
        faiss_docs_index_path = os.path.join(
            Config.FAISS_INDEX_DOCS_DIR, repo_name_for_path + "_docs"
        )
        os.makedirs(Config.FAISS_INDEX_DOCS_DIR, exist_ok=True)

        vector_stores["document"] = load_faiss_index(
            faiss_docs_index_path, lc_gemini_embeddings, "document"
        )

        if not vector_stores["document"]:
            logger.info(
                f"'{local_repo_path}'에서 문서 파일을 로드합니다 (확장자: {Config.DOCUMENT_FILE_EXTENSIONS})..."
            )
            doc_files = load_documents_from_path(
                local_repo_path, Config.DOCUMENT_FILE_EXTENSIONS
            )

            if doc_files:
                # 일반 텍스트 분할기 (Markdown, RST, TXT 등)
                text_splitter_docs = RecursiveCharacterTextSplitter(
                    chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
                )
                split_doc_files = text_splitter_docs.split_documents(doc_files)
                logger.info(
                    f"{len(doc_files)}개의 문서 파일을 {len(split_doc_files)}개의 청크로 분할했습니다."
                )
                vector_stores["document"] = create_faiss_index(
                    split_doc_files,
                    lc_gemini_embeddings,
                    faiss_docs_index_path,
                    "document",
                )
            else:
                logger.warning("인덱싱할 문서 파일을 찾지 못했습니다.")
        logger.info("--- 문서 인덱싱 종료 ---")

    except RepositorySizeError as e:
        logger.error(f"저장소 크기 초과: {e}")
        # 예외를 재발생시켜 상위 레벨에서 처리
        raise
    except RepositoryError as e:
        logger.error(f"저장소 처리 중 오류: {e}")
    except IndexingError as e:
        logger.error(f"인덱싱 중 오류: {e}")
    except EmbeddingError as e:  # GeminiAPIEmbeddings에서 직접 발생할 수 있는 오류
        logger.error(f"임베딩 생성 중 오류: {e}")
    except Exception as e:
        logger.error(f"인덱싱 파이프라인 중 예상치 못한 오류 발생: {e}", exc_info=True)
    finally:
        overall_end_time = time.time()
        logger.info(
            f"총 인덱싱 실행 시간: {format_time(overall_end_time - overall_start_time)}."
        )
        return vector_stores


def format_time(seconds: float) -> str:
    """초를 분과 초 형식의 문자열로 변환합니다."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}분 {remaining_seconds}초"
    else:
        return f"{remaining_seconds}초"
