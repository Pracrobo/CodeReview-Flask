"""요청 데이터 검증 유틸리티 모듈"""

import re
from typing import Tuple, Optional, Dict, Any
from .exceptions import ValidationError


def validate_repo_url(data: Optional[Dict[str, Any]]) -> str:
    """저장소 URL을 검증합니다."""
    if not data or "repo_url" not in data:
        raise ValidationError("repo_url이 필요합니다.")

    repo_url = data["repo_url"]
    if not isinstance(repo_url, str):
        raise ValidationError("repo_url은 문자열이어야 합니다.")

    repo_url = repo_url.strip()
    if not repo_url:
        raise ValidationError("유효한 repo_url을 입력해주세요.")

    # GitHub URL 형식 검증
    github_pattern = r"^https://github\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+(\.git)?/?$"
    if not re.match(github_pattern, repo_url):
        raise ValidationError(
            "유효한 GitHub URL 형식이 아닙니다. (예: https://github.com/owner/repo)"
        )

    return repo_url


def validate_search_request(data: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """검색 요청 데이터를 검증합니다."""
    if not data:
        raise ValidationError("요청 데이터가 필요합니다.")

    # repo_url 검증
    repo_url = validate_repo_url(data)

    # query 검증
    if "query" not in data:
        raise ValidationError("query가 필요합니다.")

    query = data["query"]
    if not isinstance(query, str):
        raise ValidationError("query는 문자열이어야 합니다.")

    query = query.strip()
    if not query:
        raise ValidationError("유효한 query를 입력해주세요.")

    if len(query) > 1000:
        raise ValidationError("query는 1000자를 초과할 수 없습니다.")

    # search_type 검증
    search_type = data.get("search_type", "code")
    if not isinstance(search_type, str):
        raise ValidationError("search_type은 문자열이어야 합니다.")

    search_type = search_type.strip().lower()
    if search_type not in ["code", "document", "doc"]:
        raise ValidationError("search_type은 'code' 또는 'document'여야 합니다.")

    # 'doc'를 'document'로 정규화
    if search_type == "doc":
        search_type = "document"

    return repo_url, query, search_type
