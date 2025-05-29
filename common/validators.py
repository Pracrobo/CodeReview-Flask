"""요청 데이터 검증 유틸리티 모듈"""

import re
from .exceptions import ValidationError


def validate_repo_url(data):
    """저장소 URL 유효성 검증"""
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


def validate_repo_name(data):
    """저장소 이름 유효성 검증 (owner/repo 형식)"""
    if not data or "repo_name" not in data:
        raise ValidationError("repo_name이 필요합니다.")

    repo_name = data["repo_name"]
    if not isinstance(repo_name, str):
        raise ValidationError("repo_name은 문자열이어야 합니다.")

    repo_name = repo_name.strip()
    if not repo_name:
        raise ValidationError("유효한 repo_name을 입력해주세요.")

    # owner/repo 형식 검증
    # GitHub 사용자 이름/조직 이름 및 저장소 이름 규칙을 단순화하여 적용
    # 일반적으로 영숫자, 하이픈(-), 밑줄(_), 점(.) 허용
    # 사용자 이름/조직 이름은 하이픈으로 시작하거나 끝날 수 없음, 연속된 하이픈 불가
    # 저장소 이름은 밑줄로 시작할 수 없음
    repo_name_pattern = r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"
    if not re.match(repo_name_pattern, repo_name):
        raise ValidationError(
            "유효한 repo_name 형식이 아닙니다. (예: owner/repository)"
        )
    if "/" in repo_name and (repo_name.startswith("/") or repo_name.endswith("/")):
        raise ValidationError("repo_name은 '/'로 시작하거나 끝날 수 없습니다.")
    if repo_name.count("/") > 1:
        raise ValidationError(
            "repo_name은 하나의 '/'만 포함해야 합니다. (owner/repository)"
        )

    return repo_name


def validate_search_request(data):
    """검색 요청 데이터 유효성 검증"""
    if not data:
        raise ValidationError("요청 데이터가 필요합니다.")

    # repo_name 검증
    repo_name = validate_repo_name(data)

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

    return repo_name, query, search_type
