"""
AI 프롬프트 관리 모듈

이 모듈은 Flask 애플리케이션에서 사용되는 모든 AI 프롬프트를 중앙에서 관리합니다.
프롬프트 수정이나 추가가 필요할 때 이 파일만 수정하면 됩니다.
"""

from typing import Dict, Any


class PromptTemplates:
    """AI 프롬프트 템플릿 관리 클래스"""

    # ===== 번역 관련 프롬프트 =====

    @staticmethod
    def get_code_query_translation_prompt(korean_text: str) -> str:
        """코드 관련 한국어 질의를 영어로 번역하는 프롬프트"""
        return f"""
다음 한국어 코드 관련 질문을 영어로 번역해주세요. 
프로그래밍 용어, 함수명, 클래스명, 변수명은 정확히 유지하세요.
코드의 의미와 맥락을 살려서 번역하세요.
번역된 영어 텍스트만 출력하세요.

한국어 질문: {korean_text}

English question:
"""

    @staticmethod
    def get_general_translation_prompt(korean_text: str) -> str:
        """일반 한국어 텍스트를 영어로 번역하는 프롬프트"""
        return f"""
다음 한국어 텍스트를 자연스러운 영어로 번역해주세요. 기술적 용어는 정확히 번역하세요.
번역된 영어 텍스트만 출력하고 다른 설명은 하지 마세요.

한국어: {korean_text}

영어:
"""

    # ===== RAG (검색 증강 생성) 관련 프롬프트 =====

    @staticmethod
    def get_code_rag_prompt(context: str, question: str) -> str:
        """코드 검색 결과를 바탕으로 한 RAG 프롬프트"""
        return f"""
주어진 코드 컨텍스트를 바탕으로 다음 질문에 대해 한국어로 상세히 답변해 주세요.
코드 예제가 있다면 포함하고, 함수나 클래스의 사용법을 설명해 주세요.
만약 컨텍스트에 질문과 관련된 코드가 없다면, "컨텍스트에 관련 코드가 없습니다."라고 답변해 주세요.

코드 컨텍스트:
{context}

질문: {question}

답변:
"""

    @staticmethod
    def get_document_rag_prompt(context: str, question: str) -> str:
        """문서 검색 결과를 바탕으로 한 RAG 프롬프트"""
        return f"""
주어진 컨텍스트 정보를 사용하여 다음 질문에 대해 한국어로 답변해 주세요.
만약 컨텍스트에 질문과 관련된 정보가 없다면, "컨텍스트에 관련 정보가 없습니다."라고 답변해 주세요.

컨텍스트:
{context}

질문: {question}

답변:
"""

    # ===== README 요약 관련 프롬프트 =====

    @staticmethod
    def get_readme_summary_prompt(repo_name: str, readme_content: str) -> str:
        """README 내용을 요약하는 프롬프트"""
        return f"""
GitHub 저장소 '{repo_name}'의 README를 한국어로 요약해주세요.

요약 지침:
- 프로젝트 목적과 주요 기능만 포함
- 적절한 개행으로 가독성 높이기
- 문단 구분은 반드시 2칸(\\n\\n) 개행으로 해주세요
- 2000자 이하로 작성

README:
{readme_content}

요약:
"""

    # ===== 이슈 관련 프롬프트 =====

    @staticmethod
    def get_issue_to_question_prompt(issue_title: str, issue_body: str) -> str:
        """이슈 내용을 검색용 질문으로 변환하는 프롬프트"""
        return f"""
다음 GitHub 이슈의 제목과 내용을 분석하여 코드 검색에 적합한 질문 형태로 변환해주세요.
변환된 질문은 관련 코드를 찾는데 최적화되어야 합니다.

이슈 제목: {issue_title}

이슈 내용:
{issue_body}

위 이슈를 바탕으로 관련 코드를 검색하기 위한 구체적인 질문을 영어로 작성해주세요.
질문만 출력하고 다른 설명은 하지 마세요.

검색 질문:
"""

    @staticmethod
    def get_ai_solution_suggestion_prompt(
        issue_title: str, issue_body: str, related_files: list, code_snippets: list
    ) -> str:
        """AI 해결 제안을 생성하는 프롬프트"""
        files_info = "\n".join(
            [
                f"- {file['path']} (관련도: {file['relevance']}%)"
                for file in related_files
            ]
        )

        snippets_info = "\n\n".join(
            [
                f"파일: {snippet['file']}\n코드:\n{snippet['code']}\n관련도: {snippet['relevance']}%\n설명: {snippet.get('explanation','')}"
                for snippet in code_snippets
            ]
        )

        return f"""
다음 GitHub 이슈에 대한 해결 제안을 한국어로 작성해주세요.
제공된 관련 파일들과 코드 스니펫들을 참고하여 구체적이고 실용적인 해결 방안을 제시하세요.

이슈 제목: {issue_title}

이슈 내용:
{issue_body}

관련 파일들:
{files_info}

관련 코드 스니펫들:
{snippets_info}

위 정보를 바탕으로 이슈 해결을 위한 구체적인 제안을 작성해주세요:
1. 문제 원인 분석
2. 해결 방법 제시
3. 수정이 필요한 파일과 구체적인 방법
4. 주의사항 및 추가 고려사항

해결 제안:
"""

    @staticmethod
    def get_issue_summary_prompt(issue_title: str, issue_body: str) -> str:
        """이슈 내용을 요약하는 프롬프트"""
        return f"""
다음 GitHub 이슈의 제목과 내용을 분석하여 핵심 내용을 한국어로 간결하게 요약해주세요.

이슈 제목: {issue_title}

이슈 내용:
{issue_body}

요약 지침:
- 핵심 문제점과 현상을 명확히 기술
- 기술적 세부사항 포함
- 2-3문장으로 간결하게 작성

이슈 요약:
"""

    # ===== 프롬프트 설정 관리 =====

    @staticmethod
    def get_prompt_config() -> Dict[str, Any]:
        """프롬프트 관련 설정값들을 반환"""
        return {
            "readme_summary": {
                "temperature": 0.3,  # 일관성 있는 요약을 위해 낮은 온도
                "retry_count": 3,  # 재시도 횟수
                "min_summary_length": 10,  # 최소 요약 길이
            },
            "translation": {
                "temperature": 0.2,  # 정확한 번역을 위해 낮은 온도
            },
            "rag": {
                "temperature": 0.4,  # 창의적이면서도 정확한 답변을 위한 중간 온도
            },
        }

    # ===== 프롬프트 검증 및 유틸리티 =====

    @staticmethod
    def validate_prompt_inputs(**kwargs) -> bool:
        """프롬프트 입력값들을 검증"""
        for key, value in kwargs.items():
            if not value or (isinstance(value, str) and not value.strip()):
                return False
        return True

    @staticmethod
    def get_fallback_description_templates() -> Dict[str, str]:
        """README 요약 실패 시 사용할 기본 설명 템플릿들"""
        return {
            "api_server": "{name}은(는) API 서버 또는 백엔드 서비스입니다.",
            "frontend": "{name}은(는) 프론트엔드 웹 애플리케이션입니다.",
            "library": "{name}은(는) 개발용 라이브러리 또는 패키지입니다.",
            "tool": "{name}은(는) 개발 도구 또는 유틸리티입니다.",
            "default": "{name}은(는) {owner}에서 개발한 오픈소스 프로젝트입니다.",
            "unknown": "{repo_name} 저장소입니다.",
        }

    @staticmethod
    def get_repository_context_answer_prompt(
        question: str, repo_info: dict, file_contents: dict
    ) -> str:
        """저장소 컨텍스트 기반 질문 답변 프롬프트"""

        context_parts = []

        # 저장소 정보
        repo_name = repo_info.get("full_name", "Unknown Repository")
        repo_description = repo_info.get("description", "")

        context_parts.append("## 저장소 정보")  # 개행은 join 시 처리
        context_parts.append(f"- 저장소명: {repo_name}")
        if repo_description:
            context_parts.append(f"- 설명: {repo_description}")

        # 파일 내용들
        for file_type, content in file_contents.items():
            context_parts.append(
                f"\n## {file_type} 파일 내용"
            )  # 각 파일 섹션 전에 개행 추가
            context_parts.append(f"```\n{content}\n```")  # 개행은 join 시 처리

        context_text = "\n".join(context_parts)

        return f"""당신은 GitHub 저장소에 대한 전문가입니다. 제공된 저장소의 파일 내용을 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

{context_text}

## 사용자 질문
{question}

## 답변 가이드라인

1.  저장소의 파일 내용을 기반으로, 정확하고 실용적인 정보를 제공하세요.
2.  필요시 파일 내용을 인용하여 답변의 근거를 제시해야 합니다.
3.  저장소의 최신 상태를 고려하여 한국어로 답변하세요 (기술 용어 병기).
4.  **답변 형식 (통합 규칙):**
    * **허용되는 유일한 서식:** 제목을 나타내는 `**굵은 글씨**`만 허용됩니다.
    * **제목 규칙:** `**굵은 글씨**`로 된 제목은 반드시 독립된 한 줄에 작성하고, 다음 문단과 빈 줄로 구분해주세요.
    * **금지되는 기호:** 제목을 제외한 어떤 곳에도 `*`, `-`, `#`, `--` 등 다른 모든 마크다운 기호나 특수 문자를 절대 사용하지 마세요.
    * **본문 규칙:** 본문은 오직 완전한 문장으로 구성된 일반 텍스트로만 작성하며, 이모티콘을 포함하지 않습니다.

답변:"""


# 편의를 위한 전역 인스턴스
prompts = PromptTemplates()


# 하위 호환성을 위한 함수들 (기존 코드에서 직접 호출할 수 있도록)
def get_code_query_translation_prompt(korean_text: str) -> str:
    """코드 질의 번역 프롬프트 (하위 호환성)"""
    return prompts.get_code_query_translation_prompt(korean_text)


def get_general_translation_prompt(korean_text: str) -> str:
    """일반 번역 프롬프트 (하위 호환성)"""
    return prompts.get_general_translation_prompt(korean_text)


def get_code_rag_prompt(context: str, question: str) -> str:
    """코드 RAG 프롬프트 (하위 호환성)"""
    return prompts.get_code_rag_prompt(context, question)


def get_document_rag_prompt(context: str, question: str) -> str:
    """문서 RAG 프롬프트 (하위 호환성)"""
    return prompts.get_document_rag_prompt(context, question)


def get_readme_summary_prompt(repo_name: str, readme_content: str) -> str:
    """README 요약 프롬프트 (하위 호환성)"""
    return prompts.get_readme_summary_prompt(repo_name, readme_content)


def get_issue_to_question_prompt(issue_title: str, issue_body: str) -> str:
    """이슈를 질문으로 변환하는 프롬프트 (하위 호환성)"""
    return prompts.get_issue_to_question_prompt(issue_title, issue_body)


def get_ai_solution_suggestion_prompt(
    issue_title: str, issue_body: str, related_files: list, code_snippets: list
) -> str:
    """AI 해결 제안 프롬프트 (하위 호환성)"""
    return prompts.get_ai_solution_suggestion_prompt(
        issue_title, issue_body, related_files, code_snippets
    )


def get_issue_summary_prompt(issue_title: str, issue_body: str) -> str:
    """이슈 요약 프롬프트 (하위 호환성)"""
    return prompts.get_issue_summary_prompt(issue_title, issue_body)


def get_repository_context_answer_prompt(
    question: str, repo_info: dict, file_contents: dict
) -> str:
    """저장소 컨텍스트 기반 질문 답변 프롬프트 (하위 호환성)"""
    return prompts.get_repository_context_answer_prompt(
        question, repo_info, file_contents
    )
