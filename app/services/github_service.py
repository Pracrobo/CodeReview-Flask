import logging
import requests
from typing import Tuple

from app.core.config import Config
from app.core.utils import get_repo_owner_and_name
from app.core.exceptions import RepositoryError, RepositorySizeError

logger = logging.getLogger(__name__)


class GitHubService:
    """GitHub API를 통한 저장소 정보 조회 서비스"""
    
    def __init__(self, api_token: str = None):
        """GitHub 서비스 초기화
        
        Args:
            api_token: GitHub API 토큰 (선택사항)
        """
        self.api_token = api_token or Config.GITHUB_API_TOKEN
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self) -> None:
        """HTTP 세션 설정"""
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AIssue-Repository-Analyzer/1.0"
        })
        
        if self.api_token:
            self.session.headers["Authorization"] = f"token {self.api_token}"
    
    def get_repository_languages(self, repo_url: str) -> Tuple[str, int]:
        """저장소의 주 사용 언어와 크기를 조회합니다.
        
        Args:
            repo_url: GitHub 저장소 URL
            
        Returns:
            (주_사용_언어, 주_언어_바이트수) 튜플
            
        Raises:
            RepositoryError: 저장소 조회 실패
            RepositorySizeError: 저장소 크기 초과
        """
        try:
            owner, repo_name = get_repo_owner_and_name(repo_url)
            api_url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
            
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            
            languages = response.json()
            
            if not languages:
                logger.warning(f"저장소 '{repo_url}'에서 언어 데이터를 찾을 수 없습니다.")
                return "unknown", 0
            
            # 주 사용 언어 결정
            primary_language = max(languages, key=languages.get)
            primary_language_bytes = languages[primary_language]
            
            logger.info(f"저장소 주 언어: {primary_language}")
            logger.info(f"주 언어 코드 크기: {primary_language_bytes:,} bytes")
            
            # 크기 검증
            self._validate_repository_size(primary_language_bytes, repo_url)
            
            return primary_language.lower(), primary_language_bytes
            
        except RepositorySizeError:
            raise
        except requests.exceptions.RequestException as e:
            raise RepositoryError(f"저장소 언어 조회 실패: {e}") from e
        except Exception as e:
            raise RepositoryError(f"예상치 못한 오류: {e}") from e
    
    def _validate_repository_size(self, size_bytes: int, repo_url: str) -> None:
        """저장소 크기를 검증합니다.
        
        Args:
            size_bytes: 크기 (바이트)
            repo_url: 저장소 URL
            
        Raises:
            RepositorySizeError: 크기 초과시
        """
        max_bytes = Config.MAX_REPO_SIZE_MB * 1024 * 1024
        size_mb = size_bytes / (1024 * 1024)
        
        if size_bytes > max_bytes:
            error_msg = (
                f"저장소 크기가 제한을 초과했습니다. "
                f"현재: {size_mb:.1f}MB, 최대: {Config.MAX_REPO_SIZE_MB:.1f}MB"
            )
            logger.error(error_msg)
            raise RepositorySizeError(error_msg)
    
    def get_repository_info(self, repo_url: str) -> dict:
        """저장소의 기본 정보를 조회합니다.
        
        Args:
            repo_url: GitHub 저장소 URL
            
        Returns:
            저장소 정보 딕셔너리
            
        Raises:
            RepositoryError: 저장소 조회 실패
        """
        try:
            owner, repo_name = get_repo_owner_and_name(repo_url)
            api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
            
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            
            repo_info = response.json()
            
            return {
                "name": repo_info.get("name"),
                "full_name": repo_info.get("full_name"),
                "description": repo_info.get("description"),
                "language": repo_info.get("language"),
                "size": repo_info.get("size"),  # KB 단위
                "stars": repo_info.get("stargazers_count"),
                "forks": repo_info.get("forks_count"),
                "updated_at": repo_info.get("updated_at"),
            }
            
        except requests.exceptions.RequestException as e:
            raise RepositoryError(f"저장소 정보 조회 실패: {e}") from e
        except Exception as e:
            raise RepositoryError(f"예상치 못한 오류: {e}") from e 