import os
import logging
import shutil
import requests
from typing import Tuple, Optional

from git import Repo, GitCommandError

from app.core.config import Config
from app.core.utils import get_repo_owner_and_name, get_local_repo_path, extract_repo_name_from_url
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


class GitService:
    """Git 저장소 복제 및 관리 서비스"""
    
    def clone_or_load_repository(self, repo_url: str) -> Repo:
        """저장소를 복제하거나 기존 저장소를 로드합니다.
        
        Args:
            repo_url: GitHub 저장소 URL
            
        Returns:
            Git 저장소 객체
            
        Raises:
            RepositoryError: 저장소 복제/로드 실패
        """
        repo_name = extract_repo_name_from_url(repo_url)
        local_path = get_local_repo_path(repo_name)
        
        if os.path.exists(local_path):
            return self._load_existing_repository(local_path, repo_url)
        else:
            return self._clone_new_repository(repo_url, local_path)
    
    def _load_existing_repository(self, local_path: str, repo_url: str) -> Repo:
        """기존 저장소를 로드합니다.
        
        Args:
            local_path: 로컬 저장소 경로
            repo_url: 원격 저장소 URL
            
        Returns:
            Git 저장소 객체
            
        Raises:
            RepositoryError: 저장소 로드 실패
        """
        logger.info(f"기존 저장소를 로드합니다: {local_path}")
        
        try:
            repo = Repo(local_path)
            self._update_repository(repo)
            return repo
            
        except GitCommandError as e:
            logger.warning(f"기존 저장소 로드 실패: {e}. 새로 복제합니다.")
            shutil.rmtree(local_path)
            return self._clone_new_repository(repo_url, local_path)
    
    def _clone_new_repository(self, repo_url: str, local_path: str) -> Repo:
        """새 저장소를 복제합니다.
        
        Args:
            repo_url: 원격 저장소 URL
            local_path: 로컬 저장소 경로
            
        Returns:
            Git 저장소 객체
            
        Raises:
            RepositoryError: 저장소 복제 실패
        """
        logger.info(f"저장소를 복제합니다: {repo_url} -> {local_path}")
        
        try:
            return Repo.clone_from(repo_url, local_path)
        except GitCommandError as e:
            raise RepositoryError(f"저장소 복제 실패: {e}") from e
    
    def _update_repository(self, repo: Repo) -> None:
        """저장소를 최신 상태로 업데이트합니다.
        
        Args:
            repo: Git 저장소 객체
        """
        try:
            logger.info("저장소를 최신 상태로 업데이트합니다.")
            origin = repo.remotes.origin
            origin.pull()
            logger.info("저장소 업데이트 완료")
            
        except GitCommandError as e:
            logger.warning(f"저장소 업데이트 실패: {e}. 기존 상태를 유지합니다.")
    
    def get_repository_stats(self, repo: Repo) -> dict:
        """저장소 통계 정보를 반환합니다.
        
        Args:
            repo: Git 저장소 객체
            
        Returns:
            저장소 통계 딕셔너리
        """
        try:
            commits = list(repo.iter_commits())
            branches = list(repo.branches)
            
            return {
                "total_commits": len(commits),
                "total_branches": len(branches),
                "current_branch": repo.active_branch.name,
                "last_commit": {
                    "hash": commits[0].hexsha[:8] if commits else None,
                    "message": commits[0].message.strip() if commits else None,
                    "author": str(commits[0].author) if commits else None,
                    "date": commits[0].committed_datetime.isoformat() if commits else None,
                }
            }
            
        except Exception as e:
            logger.warning(f"저장소 통계 수집 실패: {e}")
            return {}
    
    def cleanup_repository(self, repo_name: str) -> bool:
        """저장소 디렉토리를 정리합니다.
        
        Args:
            repo_name: 저장소 이름
            
        Returns:
            정리 성공 여부
        """
        local_path = get_local_repo_path(repo_name)
        
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
                logger.info(f"저장소 디렉토리 정리 완료: {local_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"저장소 디렉토리 정리 실패: {e}")
            return False 