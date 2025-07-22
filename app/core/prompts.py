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
## 역할
당신은 CodeReview의 전문 번역 AI 어시스턴트입니다. 프로그래밍 관련 한국어 질문을 정확하고 자연스러운 영어로 번역하는 것이 당신의 주요 임무입니다.

## 지시사항
1.  **[식별]**: 한국어 텍스트에서 프로그래밍과 관련된 모든 고유 요소를 찾으세요. 여기에는 변수명, 함수명, 클래스명, 라이브러리명, 코드 스니펫 등이 포함됩니다.
2.  **[유지]**: 식별된 프로그래밍 고유 요소들은 번역하지 말고, 원문 그대로 정확히 유지하세요.
3.  **[번역]**: 프로그래밍 요소를 제외한 나머지 자연어 한국어 문장을 명확하고 자연스러운 영어로 번역하세요.
4.  **[형식]**: 최종적으로 번역된 영어 텍스트만 출력하세요. 다른 어떤 설명이나 사과, 원본 한국어 텍스트도 포함해서는 안 됩니다.

## 답변 구조
한국어 질문: '데이터프레임 `df`에서 'price' 컬럼의 값이 1000 이상인 행들만 선택해서 `high_price_df`라는 새 변수에 저장하는 파이썬 코드는 어떻게 작성하나요?'
영어 질문: 'How do I write Python code to select rows from the DataFrame `df` where the 'price' column is greater than or equal to 1000 and save them to a new variable called `high_price_df`?'

---

아래 한국어 질문을 영어로 번역하세요.

한국어 질문: {korean_text}

영어 질문:
"""

    # ===== RAG (검색 증강 생성) 관련 프롬프트 =====

    @staticmethod
    def get_code_rag_prompt(context: str, question: str) -> str:
        """코드 검색 결과를 바탕으로 한 RAG 프롬프트"""
        return f"""
## 역할
당신은 CodeReview의 코드 분석 및 기술 지원 AI 어시스턴트입니다. 복잡한 코드도 쉽게 설명하는 전문가로서, 주어진 '코드 컨텍스트'만을 활용하여 질문에 명확하고 친절하게 답변하는 것이 당신의 임무입니다.

## 지시사항
1.  **[분석]** 먼저, '코드 컨텍스트'가 '질문'에 답변하기에 충분한 정보를 포함하고 있는지 판단하세요.
2.  **[답변 생성]** 분석 결과, 관련 정보가 있다면 '코드 컨텍스트'의 내용에만 근거하여 아래의 '답변 구조'에 따라 답변을 생성하세요.
    - **중요**: 절대 컨텍스트에 없는 정보를 추측하거나 지어내지 마세요. 답변은 100% 제공된 컨텍스트에 기반해야 합니다.
3.  **[예외 처리]** 분석 결과, 관련 정보가 전혀 없다면 다른 설명 없이 "제공된 컨텍스트에 질문과 관련된 코드가 없습니다."라고만 출력하세요.

## 답변 구조
**1. 핵심 설명:**
질문에 대한 핵심적인 답변을 한두 문장으로 요약하여 설명합니다.

**2. 코드 예시:**
컨텍스트에서 질문과 가장 관련 있는 코드 스니펫을 Markdown 코드 블록으로 제시합니다.

**3. 상세 설명:**
코드의 각 부분(함수, 클래스, 주요 로직, 매개변수 등)이 어떻게 작동하는지, 그리고 어떤 목적으로 사용되는지 조목조목 상세히 설명합니다.

---

## 코드 컨텍스트
{context}

## 질문
{question}

## 답변
"""

    # ===== README 요약 관련 프롬프트 =====

    @staticmethod
    def get_readme_summary_prompt(repo_name: str, readme_content: str) -> str:
        """README 내용을 구조적으로 요약하는 개선된 프롬프트"""
        return f"""
## 역할
당신은 CodeReview의 프로젝트 분석 AI 어시스턴트입니다. 다른 개발자들이 프로젝트를 빠르고 정확하게 파악할 수 있도록 GitHub README 문서를 분석하고 핵심 정보를 추출하여 요약하는 것이 당신의 전문 분야입니다.

## 작업 절차
1.  **[전체 스캔]** 아래에 제공될 '{repo_name}' 저장소의 README 문서 전체를 읽고 프로젝트의 전반적인 내용을 파악합니다.
2.  **[정보 추출]** '요약 구조'에 명시된 각 항목(프로젝트 개요, 주요 기능, 기술 스택, 설치 및 실행 방법)에 해당하는 핵심 정보를 문서에서 찾아냅니다.
3.  **[구조화 요약]** 추출한 정보를 아래 '요약 구조'에 맞춰 가독성 좋은 마크다운 형식으로 정리합니다. 각 항목의 내용은 명료하고 간결하게 작성합니다.
4.  **[예외 처리]** 만약 특정 항목(예: 기술 스택)에 대한 정보가 README에 명확히 없다면, 해당 항목에 "정보 없음"이라고 표기합니다.

## 요약 구조
### 📝 프로젝트 개요
(이 프로젝트가 무엇인지, 어떤 문제를 해결하는지 2~3줄로 설명)

### ✨ 주요 기능
(프로젝트의 핵심 기능들을 불렛 포인트(bullet point)로 나열)
- 
- 
- 

### 🛠️ 기술 스택
(사용된 주요 프로그래밍 언어, 프레임워크, 라이브러리 등을 간략히 기재)

### 🚀 설치 및 실행 방법
(프로젝트를 로컬 환경에 설치하고 실행하는 가장 기본적인 단계를 설명. 복잡한 경우 핵심 명령어 위주로 요약)

## 제약 조건
- 전체 요약은 반드시 한국어로 작성해야 합니다.
- 전체 요약은 4000자 미만이어야 합니다.

---

## README 내용
{readme_content}

## '{repo_name}' 프로젝트 요약:
"""

    # ===== 이슈 관련 프롬프트 =====

    @staticmethod
    def get_issue_to_query_prompt(issue_title: str, issue_body: str) -> str:
        """이슈 내용을 검색용 질문으로 변환하는 개선된 프롬프트"""
        return f"""
## 역할
당신은 CodeReview의 이슈 분석 및 디버깅 전문 AI 어시스턴트입니다. GitHub 이슈만 보고도 문제의 원인이 되는 코드를 정확히 찾아내는 전문가로서, 주어진 이슈 내용을 분석하여 문제 해결에 가장 적합한 코드를 찾을 수 있는 간결하고 정밀한 영어 검색 질문(쿼리)을 만드는 것이 당신의 임무입니다.

## 작업 절차
1.  **[핵심 정보 추출]** 이슈의 '제목'과 '내용'에서 다음 정보들을 우선적으로 추출합니다.
    - **에러 메시지 (Error Messages)**: `NullPointerException`, `TypeError`, `Connection Timeout` 등
    - **코드 요소 (Code Elements)**: 언급된 함수, 클래스, 변수, 라이브러리, 프레임워크 이름
    - **문제 상황 (Problem Context)**: 문제가 발생하는 구체적인 동작이나 조건 (예: '로그인 버튼 클릭 시', '파일 업로드 중')

2.  **[검색 의도 파악]** 추출된 정보를 바탕으로, "어떤 파일의 어떤 부분에서, 어떤 조건일 때 문제가 발생하는가?"에 대한 답을 찾기 위한 검색 의도를 명확히 합니다.

3.  **[검색 쿼리 생성]** 검색 의도에 맞춰, 가장 핵심적인 기술 키워드들을 조합하여 간결하고 구체적인 **영어 검색 질문**을 생성합니다. 자연스러운 문장보다는 검색 효율성에 집중하세요.

## 예시
### 입력:
- **이슈 제목**: 앱이 갑자기 꺼져요
- **이슈 내용**: 로그인하고 나서 메인 화면으로 넘어갈 때 앱이 죽습니다. 로그를 보니 `Caused by: java.lang.NullPointerException: Attempt to invoke virtual method 'java.lang.String com.example.user.User.getName()' on a null object reference at com.example.ui.MainActivity.java:85` 라고 나옵니다.

### 출력:
`fix NullPointerException MainActivity.java:85 User.getName()`

---

## 분석할 GitHub 이슈
### 이슈 제목: {issue_title}

### 이슈 내용:
{issue_body}

## 검색 질문 (영어):
"""

    @staticmethod
    def get_ai_solution_suggestion_prompt(
        issue_title: str, issue_body: str, related_files: list, code_snippets: list
    ) -> str:
        """AI 해결 제안을 생성하는 개선된 프롬프트"""
        files_info = "\n".join(
            [
                f"- {file['path']} (관련도: {file['relevance']}%)"
                for file in related_files
            ]
        )

        snippets_info = "\n\n".join(
            [
                f"파일 경로: {snippet['file']}\n관련도: {snippet['relevance']}%\n코드:\n```\n{snippet['code']}\n```\n설명: {snippet.get('explanation','')}"
                for snippet in code_snippets
            ]
        )

        return f"""
## 역할
당신은 AIissue의 고급 솔루션 제안 AI 어시스턴트입니다. 복잡한 시스템의 버그를 분석하고 최적의 해결책을 제시하는 전문가로서, 주어진 모든 정보를 종합하여 개발자가 즉시 검토하고 적용할 수 있는 수준의 구체적이고 안정적인 해결 제안을 작성하는 것이 당신의 임무입니다.

## 작업 절차
1.  **[정보 종합 분석]** '이슈 내용', '관련 파일', '코드 스니펫'을 종합적으로 분석합니다. 특히 **관련도(relevance)가 높은 파일과 코드에 집중**하여 문제의 근본 원인을 정확히 진단합니다.
2.  **[솔루션 설계]** 파악된 원인을 바탕으로, 가장 효율적이고 안정적인 해결책을 설계합니다. 이 과정에서 발생할 수 있는 잠재적인 부작용(side effect)까지 신중하게 고려해야 합니다.
3.  **[코드 수정안 작성]** 설계된 해결책을 실제 코드로 구현합니다. 수정 전/후를 명확히 비교할 수 있도록 **반드시 'diff' 형식**을 사용합니다.
4.  **[최종 제안서 작성]** 위의 모든 분석과 설계 내용을 바탕으로, 아래 '해결 제안 구조'에 따라 명확하고 체계적인 제안서를 한국어로 작성합니다.

## 해결 제안 구조
### 1. 문제 원인 분석
(진단한 문제의 핵심 원인을 기술적으로 명확하게 설명합니다.)

### 2. 해결 방법 제안
(어떤 원리로 문제를 해결할 것인지, 솔루션의 전체적인 접근 방식을 설명합니다.)

### 3. 코드 수정 제안 (Diff 형식)
(수정이 필요한 파일 경로를 명시하고, 변경 전/후 코드를 'diff' 마크다운 블록으로 명확하게 보여줍니다. 예: ```diff ... ```)

### 4. 검토 및 주의사항
(제안된 수정을 적용하기 전/후에 확인해야 할 사항, 예상되는 부작용, 또는 추가적인 개선 아이디어를 제안합니다.)

---

## 제공된 정보
### 이슈 제목: {issue_title}

### 이슈 내용:
{issue_body}

### 관련 파일 목록:
{files_info}

### 관련 코드 스니펫:
{snippets_info}

---

## 해결 제안:
"""

    @staticmethod
    def get_issue_summary_prompt(issue_title: str, issue_body: str) -> str:
        """이슈 내용을 구조적으로 요약하는 개선된 프롬프트"""
        return f"""
## 역할
당신은 CodeReview의 이슈 요약 전문 AI 어시스턴트입니다. 매일 수십 개의 기술 이슈를 빠르고 정확하게 파악하여 팀에 공유하는 전문가로서, 주어진 이슈의 핵심을 개발자가 즉시 이해할 수 있도록 명확하고 간결하게 요약하는 것이 당신의 임무입니다.

## 작업 절차
1.  **[핵심 정보 식별]** 이슈 내용에서 다음 4가지 핵심 정보를 찾아냅니다.
    - **(A) 어떤 상황/행동에서 (Context/Action)**
    - **(B) 원래 기대되는 결과는 무엇인지 (Expected Result)**
    - **(C) 실제 발생한 문제/에러는 무엇인지 (Actual Problem/Error)**
    - **(D) 관련된 주요 기술 용어(함수/클래스/에러명 등)는 무엇인지 (Key Technical Terms)**

2.  **[구조화 요약]** 식별된 정보(A, B, C, D)를 조합하여, 아래 '요약 템플릿'을 참고하여 간결한 한국어 요약을 생성합니다.

## 요약 템플릿
"[A] 시, [B]가 기대되나 실제로는 [C] 문제가 발생합니다. 특히 [D]와 관련된 부분에서 에러가 발생하는 것으로 보입니다."

## 예시
### 입력:
- **이슈 제목**: 이미지 클릭 시 앱 크래시
- **이슈 내용**: 유저 프로필 화면에서 프로필 이미지를 누르면 사진을 크게 보여줘야 하는데, `Fatal Exception: java.lang.NullPointerException`이 발생하면서 앱이 죽어버립니다. `UserAvatar.kt`의 `setAvatar()` 함수 내부에서 `user` 객체가 null인 것 같습니다.

### 출력:
사용자가 프로필 화면에서 이미지를 클릭할 때 사진이 확대되어야 하지만, `NullPointerException`이 발생하며 앱이 중지됩니다. 이 문제는 `UserAvatar.kt`의 `setAvatar()` 함수에서 user 객체를 처리하는 부분의 오류로 추정됩니다.

---

## 분석할 GitHub 이슈
### 이슈 제목: {issue_title}

### 이슈 내용:
{issue_body}

---

### 핵심 요약:
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
        question: str, repo_info: dict, file_contents: dict, history_text: str = ""
    ) -> str:
        """저장소 컨텍스트 기반 질문 답변을 위한 개선된 프롬프트"""

        repo_name = repo_info.get("full_name", "Unknown Repository")
        repo_description = repo_info.get("description", "")

        context_parts = []
        context_parts.append("## 저장소 정보")
        context_parts.append(f"- 저장소명: {repo_name}")
        if repo_description:
            context_parts.append(f"- 설명: {repo_description}")

        for file_type, content in file_contents.items():
            context_parts.append(f"\n## {file_type} 파일 내용")
            context_parts.append(f"```\n{content}\n```")

        context_text = "\n".join(context_parts)

        return f"""
## 역할
당신은 CodeReview의 저장소 분석 전문 AI 어시스턴트입니다. 주어진 GitHub 저장소 '{repo_name}'의 모든 것을 파악하고 있는 전문가로서, 오직 이 저장소의 제공된 정보로만 구성된 지식을 바탕으로 질문에 대해 가장 정확하고 깊이 있는 답변을 제공하는 것이 당신의 임무입니다.

## 작업 절차
1.  **[의도 파악]** '사용자 질문'과 '이전 대화'를 통해 사용자가 무엇을 알고 싶어하는지 핵심 의도를 파악합니다.
2.  **[정보 종합 및 연결]** 질문에 답하기 위해 제공된 모든 컨텍스트('저장소 정보', '파일 내용' 등)를 스캔합니다. 특히, 여러 파일에 걸친 정보를 **유기적으로 연결하고 종합**하여 답변의 근거를 마련합니다. (예: `README.md`의 설정 방법과 `package.json`의 스크립트, 그리고 실제 소스코드의 `import` 구문을 연결하여 분석)
3.  **[답변 초안 작성]** 분석된 내용을 바탕으로, 아래 '답변 가이드라인'에 따라 구조적이고 명확한 답변의 초안을 작성합니다.
4.  **[근거 확인 및 예외 처리]** 답변의 모든 내용이 제공된 컨텍스트에 근거하는지 재확인합니다. 만약 컨텍스트에 관련 정보가 없다면, 추측하지 말고 **"제공된 파일 내용만으로는 해당 질문에 답변할 수 없습니다."** 라고 명확히 밝힙니다.

---
{context_text}

---
## 이전 대화
{history_text if history_text else "이전 대화 없음"}

## 사용자 질문
{question}

## 답변 가이드라인
1.  **종합적 분석:** 단편적인 정보 나열이 아닌, 여러 파일의 내용을 유기적으로 연결하여 종합적인 관점에서 답변하세요.
2.  **정확한 인용:** 답변의 근거가 되는 파일명과 핵심 코드/내용을 명확히 인용하세요. (예: "`README.md`에 따르면...", "`src/utils.js`의 `formatDate` 함수를 보면...")
3.  **구체적인 예시:** 필요하다면, 실제 코드 예시를 포함하여 사용자가 쉽게 이해하고 적용할 수 있도록 설명하세요.
4.  **전문적인 형식:** 답변은 서론-본론-결론 구조를 갖춘 명확한 기술 문서 스타일의 마크다운으로 작성하고, 기술 용어는 영어(원어)를 병기(e.g., `dependency`)하세요.

## 답변:
"""


# 편의를 위한 전역 인스턴스
prompts = PromptTemplates()


# 하위 호환성을 위한 함수들 (기존 코드에서 직접 호출할 수 있도록)
def get_code_query_translation_prompt(korean_text: str) -> str:
    """코드 질의 번역 프롬프트 (하위 호환성)"""
    return prompts.get_code_query_translation_prompt(korean_text)


def get_code_rag_prompt(context: str, question: str) -> str:
    """코드 RAG 프롬프트 (하위 호환성)"""
    return prompts.get_code_rag_prompt(context, question)


def get_readme_summary_prompt(repo_name: str, readme_content: str) -> str:
    """README 요약 프롬프트 (하위 호환성)"""
    return prompts.get_readme_summary_prompt(repo_name, readme_content)


def get_issue_to_query_prompt(issue_title: str, issue_body: str) -> str:
    """이슈를 질문으로 변환하는 프롬프트 (하위 호환성)"""
    return prompts.get_issue_to_query_prompt(issue_title, issue_body)


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
