import os
import logging
import shutil
from typing import Optional

from git import Repo, GitCommandError

from app.core.utils import get_local_repo_path, extract_repo_name_from_url
from app.core.exceptions import RepositoryError

logger = logging.getLogger(__name__)


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