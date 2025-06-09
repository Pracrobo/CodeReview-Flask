"""README 요약 서비스 모듈"""

import logging
import re
from typing import Optional
import asyncio  # asyncio 임포트

from google.genai import types

from app.core.config import Config
from app.core.prompts import prompts
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)


class ReadmeSummarizer:
    """README 내용을 AI로 요약하는 서비스"""

    def __init__(self):
        """README 요약 서비스 초기화"""
        self.llm_model = Config.DEFAULT_LLM_MODEL
        # 프롬프트 설정에서 값 가져오기
        prompt_config = prompts.get_prompt_config()
        self.max_retries = prompt_config["readme_summary"]["retry_count"]
        self.min_summary_length = prompt_config["readme_summary"]["min_summary_length"]
        self.temperature = prompt_config["readme_summary"]["temperature"]
        self._client = None

    def _get_client(self):
        """Gemini API 클라이언트 가져오기"""
        return gemini_service.get_client()

    def _clean_readme_content(
        self, content: str, repo_name: str
    ) -> str:  # repo_name 인자 추가
        """README 내용 전처리"""
        if not content:
            return ""

        # 마크다운 이미지 제거
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

        # 마크다운 링크를 텍스트만 남기기
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)

        # 배지(badge) 제거
        content = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", content)

        return content.strip()

    async def summarize_readme(
        self, repo_name: str, readme_content: str
    ) -> Optional[str]:  # async def로 변경
        """README 내용을 요약합니다.

        Args:
            repo_name: 저장소 이름 (owner/repo 형식)
            readme_content: README 파일 내용

        Returns:
            요약된 내용 (한국어) 또는 None (실패 시)
        """
        if not readme_content or not readme_content.strip():
            logger.warning(
                f"'{repo_name}' 저장소의 README 내용이 비어있어 요약을 건너뜁니다."
            )
            return None

        # 프롬프트 입력값 검증
        if not prompts.validate_prompt_inputs(
            repo_name=repo_name, readme_content=readme_content
        ):
            logger.warning(
                f"'{repo_name}' 저장소 요약 중 프롬프트 입력값이 유효하지 않습니다."
            )
            return None

        try:
            # README 내용 전처리
            cleaned_content = self._clean_readme_content(
                readme_content, repo_name
            )  # repo_name 전달
            if not cleaned_content:
                logger.warning(
                    f"'{repo_name}' 저장소의 README 내용이 전처리 후 비어있어 요약을 건너뜁니다."
                )
                return None

            # 프롬프트 생성
            prompt = prompts.get_readme_summary_prompt(repo_name, cleaned_content)

            # Gemini API 호출
            client = self._get_client()
            loop = asyncio.get_event_loop()

            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"'{repo_name}' 저장소 README 요약을 위해 Gemini API 호출 시작. 모델: {self.llm_model}, 온도: {self.temperature}"
                    )
                    logger.info(
                        f"'{repo_name}' 저장소 README 요약 API 호출 시도 ({attempt + 1}/{self.max_retries})"
                    )

                    # run_in_executor에 키워드 인자를 직접 넘기면 안 되므로 래퍼 함수 사용
                    def call_generate_content():
                        return client.models.generate_content(
                            model=self.llm_model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=self.temperature,
                            ),
                        )

                    response = await loop.run_in_executor(None, call_generate_content)
                    summary = gemini_service.extract_text_from_response(response)

                    if summary:
                        logger.info(
                            f"'{repo_name}' 저장소 README 요약 성공. 요약 길이: {len(summary)}자"
                        )
                        return summary
                    else:
                        logger.warning(
                            f"'{repo_name}' 저장소의 Gemini API 응답에서 요약 텍스트를 추출하지 못했습니다."
                        )
                        logger.warning(f"응답 전체 내용: {response}")
                        continue  # 다음 재시도

                except Exception as e:
                    logger.warning(
                        f"'{repo_name}' 저장소 README 요약 API 호출 실패 (시도 {attempt + 1}/{self.max_retries}). 오류: {e}"
                    )
                    if attempt == self.max_retries - 1:
                        # 모든 재시도 실패 시 여기서 루프를 빠져나가 아래의 최종 에러 로깅으로 연결
                        break

                    # 재시도 전 잠시 대기
                    import time

                    time.sleep(2**attempt)  # 지수 백오프

            # for 루프가 break 없이 완료되었거나 (모든 재시도 실패), summary를 못 찾은 경우
            logger.error(
                f"'{repo_name}' 저장소 README 요약 API 호출 최종 실패 (모든 재시도 소진)."
            )
            return None

        except Exception as e:
            logger.error(f"'{repo_name}' 저장소 README 요약 처리 중 예외 발생: {e}")
            return None

    def create_fallback_description(
        self, repo_name: str, repo_info: dict = None
    ) -> str:
        """README 요약이 실패했을 때 사용할 기본 설명 생성"""
        try:
            # 저장소 이름에서 정보 추출
            if "/" in repo_name:
                owner, name = repo_name.split("/", 1)
            else:
                owner, name = "알 수 없음", repo_name  # 기본값 한국어화

            # 기본 설명 템플릿 가져오기
            templates = prompts.get_fallback_description_templates()

            # 기본 설명 템플릿
            if repo_info and repo_info.get("description"):
                return repo_info["description"]

            # 저장소 이름 기반 기본 설명
            if any(keyword in name.lower() for keyword in ["api", "server", "backend"]):
                return templates["api_server"].format(name=name)
            elif any(
                keyword in name.lower() for keyword in ["frontend", "ui", "web", "app"]
            ):
                return templates["frontend"].format(name=name)
            elif any(
                keyword in name.lower() for keyword in ["lib", "library", "package"]
            ):
                return templates["library"].format(name=name)
            elif any(keyword in name.lower() for keyword in ["tool", "cli", "util"]):
                return templates["tool"].format(name=name)
            else:
                return templates["default"].format(name=name, owner=owner)

        except Exception as e:
            logger.warning(f"'{repo_name}' 저장소의 기본 설명 생성 중 오류 발생: {e}")
            templates = prompts.get_fallback_description_templates()
            return templates["unknown"].format(repo_name=repo_name)
