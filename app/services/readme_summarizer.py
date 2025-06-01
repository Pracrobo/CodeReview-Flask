"""README 요약 서비스 모듈"""

import logging
import re
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import Config
from app.core.exceptions import ServiceError

logger = logging.getLogger(__name__)


class ReadmeSummarizer:
    """README 내용을 AI로 요약하는 서비스"""
    
    def __init__(self):
        """README 요약 서비스 초기화"""
        self.llm_model = Config.DEFAULT_LLM_MODEL
        self.max_retries = 3
        self._client = None
    
    def _get_client(self):
        """Gemini API 클라이언트 가져오기"""
        if self._client is None:
            try:
                # 기본적으로 첫 번째 API 키 사용
                api_key = Config.GEMINI_API_KEY1
                if not api_key or api_key == "dummy_key_1":
                    # 두 번째 API 키 시도
                    api_key = Config.GEMINI_API_KEY2
                    if not api_key or api_key == "dummy_key_2":
                        raise ServiceError("유효한 Gemini API 키가 설정되지 않았습니다.")
                
                self._client = genai.Client(api_key=api_key)
                logger.info("README 요약용 Gemini 클라이언트 초기화 완료")
            except Exception as e:
                raise ServiceError(f"Gemini API 클라이언트 생성 실패: {e}")
        
        return self._client
    
    def _clean_readme_content(self, content: str) -> str:
        """README 내용 전처리"""
        if not content:
            return ""
        
        # 마크다운 이미지 제거
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        
        # 마크다운 링크를 텍스트만 남기기
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # HTML 태그 제거
        content = re.sub(r'<[^>]+>', '', content)
        
        # 코드 블록 간소화 (```로 둘러싸인 부분)
        content = re.sub(r'```[\s\S]*?```', '[코드 예제]', content)
        
        # 인라인 코드 간소화
        content = re.sub(r'`([^`]+)`', r'[\1]', content)
        
        # 연속된 공백과 줄바꿈 정리
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # 너무 긴 내용은 잘라내기 (토큰 제한 고려)
        if len(content) > 8000:
            content = content[:8000] + "..."
            logger.warning("README 내용이 너무 길어서 8000자로 제한했습니다.")
        
        return content.strip()
    
    def _create_summary_prompt(self, repo_name: str, readme_content: str) -> str:
        """README 요약을 위한 프롬프트 생성"""
        return f"""
다음은 GitHub 저장소 '{repo_name}'의 README 파일 내용입니다.
이 내용을 바탕으로 한국어로 간결하고 명확한 요약을 작성해주세요.

요약 작성 지침:
1. 2-3문장으로 핵심 내용만 요약
2. 프로젝트의 목적과 주요 기능을 포함
3. 기술적 용어는 적절히 한국어로 번역하되, 널리 알려진 용어는 원문 유지
4. 설치나 사용법 등의 세부사항은 제외
5. 마케팅성 문구나 과장된 표현은 피하고 객관적으로 서술
6. 요약문만 출력하고 다른 설명은 하지 마세요

README 내용:
{readme_content}

요약:
"""
    
    def summarize_readme(self, repo_name: str, readme_content: str) -> Optional[str]:
        """README 내용을 요약합니다.
        
        Args:
            repo_name: 저장소 이름 (owner/repo 형식)
            readme_content: README 파일 내용
            
        Returns:
            요약된 내용 (한국어) 또는 None (실패 시)
        """
        if not readme_content or not readme_content.strip():
            logger.warning(f"README 내용이 비어있습니다: {repo_name}")
            return None
        
        try:
            # README 내용 전처리
            cleaned_content = self._clean_readme_content(readme_content)
            if not cleaned_content:
                logger.warning(f"전처리 후 README 내용이 비어있습니다: {repo_name}")
                return None
            
            # 프롬프트 생성
            prompt = self._create_summary_prompt(repo_name, cleaned_content)
            
            # Gemini API 호출
            client = self._get_client()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"README 요약 시도 {attempt + 1}/{self.max_retries}: {repo_name}")
                    
                    response = client.models.generate_content(
                        model=self.llm_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,  # 일관성 있는 요약을 위해 낮은 온도
                            max_output_tokens=500,  # 요약문 길이 제한
                        )
                    )
                    
                    summary = response.text.strip()
                    
                    if summary and len(summary) > 10:  # 최소 길이 검증
                        logger.info(f"README 요약 완료: {repo_name}")
                        return summary
                    else:
                        logger.warning(f"요약 결과가 너무 짧습니다: {repo_name}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"README 요약 시도 {attempt + 1} 실패: {repo_name} - {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    
                    # 재시도 전 잠시 대기
                    import time
                    time.sleep(2 ** attempt)  # 지수 백오프
            
            logger.error(f"모든 재시도 실패: {repo_name}")
            return None
            
        except Exception as e:
            logger.error(f"README 요약 중 오류 발생: {repo_name} - {e}")
            return None
    
    def create_fallback_description(self, repo_name: str, repo_info: dict = None) -> str:
        """README 요약이 실패했을 때 사용할 기본 설명 생성"""
        try:
            # 저장소 이름에서 정보 추출
            if '/' in repo_name:
                owner, name = repo_name.split('/', 1)
            else:
                owner, name = 'unknown', repo_name
            
            # 기본 설명 템플릿
            if repo_info and repo_info.get('description'):
                return repo_info['description']
            
            # 저장소 이름 기반 기본 설명
            if any(keyword in name.lower() for keyword in ['api', 'server', 'backend']):
                return f"{name}은(는) API 서버 또는 백엔드 서비스입니다."
            elif any(keyword in name.lower() for keyword in ['frontend', 'ui', 'web', 'app']):
                return f"{name}은(는) 프론트엔드 웹 애플리케이션입니다."
            elif any(keyword in name.lower() for keyword in ['lib', 'library', 'package']):
                return f"{name}은(는) 개발용 라이브러리 또는 패키지입니다."
            elif any(keyword in name.lower() for keyword in ['tool', 'cli', 'util']):
                return f"{name}은(는) 개발 도구 또는 유틸리티입니다."
            else:
                return f"{name}은(는) {owner}에서 개발한 오픈소스 프로젝트입니다."
                
        except Exception as e:
            logger.warning(f"기본 설명 생성 실패: {repo_name} - {e}")
            return f"{repo_name} 저장소입니다." 