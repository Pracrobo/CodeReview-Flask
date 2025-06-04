import os
import logging

from .config import Config

logger = logging.getLogger(__name__)


def extract_repo_name_from_url(repo_url: str) -> str:
    """GitHub URL에서 저장소 이름을 추출합니다.

    Args:
        repo_url: GitHub 저장소 URL

    Returns:
        저장소 이름 (예: "flask")

    Raises:
        ValueError: 유효하지 않은 URL인 경우
    """
    if not isinstance(repo_url, str) or not repo_url:
        raise ValueError("유효하지 않은 저장소 URL입니다.")

    return repo_url.split("/")[-1].removesuffix(".git")


def get_repo_owner_and_name(repo_url: str) -> tuple[str, str]:
    """GitHub URL에서 소유자와 저장소 이름을 추출합니다.

    Args:
        repo_url: GitHub 저장소 URL

    Returns:
        (소유자, 저장소이름) 튜플

    Raises:
        ValueError: 유효하지 않은 URL인 경우
    """
    if not repo_url.startswith("https://github.com/"):
        raise ValueError(f"유효하지 않은 GitHub URL입니다: {repo_url}")

    parts = repo_url.split("/")
    if len(parts) < 5:
        raise ValueError(f"URL에서 소유자/저장소 이름을 파싱할 수 없습니다: {repo_url}")

    owner = parts[-2]
    repo_name = parts[-1].removesuffix(".git")

    return owner, repo_name


def get_local_repo_path(repo_name: str) -> str:
    """저장소의 로컬 경로를 반환합니다.

    Args:
        repo_name: 저장소 이름

    Returns:
        로컬 저장소 경로
    """
    return os.path.join(Config.BASE_CLONED_DIR, repo_name)


def get_faiss_index_path(repo_name: str, index_type: str) -> str:
    """FAISS 인덱스 경로를 반환합니다.

    Args:
        repo_name: 저장소 이름
        index_type: 인덱스 타입 ("code" 또는 "document")

    Returns:
        FAISS 인덱스 경로

    Raises:
        ValueError: 알 수 없는 인덱스 타입인 경우
    """
    if index_type == "code":
        return os.path.join(Config.FAISS_INDEX_BASE_DIR, f"{repo_name}_code")
    elif index_type == "document":
        return os.path.join(Config.FAISS_INDEX_DOCS_DIR, f"{repo_name}_docs")
    else:
        raise ValueError(f"알 수 없는 인덱스 타입입니다: {index_type}")


def format_duration(seconds: float) -> str:
    """초를 사람이 읽기 쉬운 형식으로 변환합니다.

    Args:
        seconds: 초 단위 시간

    Returns:
        포맷된 시간 문자열 (예: "2분 30초", "45초")
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes > 0:
        return f"{minutes}분 {remaining_seconds}초"
    else:
        return f"{remaining_seconds}초"


def ensure_directory_exists(directory_path: str) -> None:
    """디렉토리가 존재하지 않으면 생성합니다.

    Args:
        directory_path: 생성할 디렉토리 경로
    """
    os.makedirs(directory_path, exist_ok=True)


def check_index_exists(repo_name: str, index_type: str) -> bool:
    """인덱스 파일이 존재하는지 확인합니다.

    Args:
        repo_name: 저장소 이름
        index_type: 인덱스 타입 ("code" 또는 "document")

    Returns:
        인덱스 존재 여부
    """
    index_path = get_faiss_index_path(repo_name, index_type)
    exists = os.path.exists(index_path)

    logger.debug(
        f"인덱스 확인 - 저장소: '{repo_name}', 타입: '{index_type}', "
        f"결과: {'존재' if exists else '없음'}, 경로: '{index_path}'"
    )

    return exists
