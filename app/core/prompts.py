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
