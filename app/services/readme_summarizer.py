"""README 요약 서비스 모듈"""

import logging
import re
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import Config
from app.core.exceptions import ServiceError
from app.core.prompts import prompts

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
        self.max_output_tokens = prompt_config["readme_summary"]["max_output_tokens"]
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
        
        # 배지(badge) 제거
        content = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', content)
        
        # 목차(Table of Contents) 제거
        content = re.sub(r'(?i)## table of contents.*?(?=##|\Z)', '', content, flags=re.DOTALL)
        content = re.sub(r'(?i)## contents.*?(?=##|\Z)', '', content, flags=re.DOTALL)
        
        # 설치 방법, 사용법 등 긴 섹션 간소화
        content = re.sub(r'(?i)## installation.*?(?=##|\Z)', '[설치 방법 생략]', content, flags=re.DOTALL)
        content = re.sub(r'(?i)## usage.*?(?=##|\Z)', '[사용법 생략]', content, flags=re.DOTALL)
        content = re.sub(r'(?i)## examples.*?(?=##|\Z)', '[예제 생략]', content, flags=re.DOTALL)
        content = re.sub(r'(?i)## contributing.*?(?=##|\Z)', '[기여 방법 생략]', content, flags=re.DOTALL)
        content = re.sub(r'(?i)## license.*?(?=##|\Z)', '[라이선스 정보 생략]', content, flags=re.DOTALL)
        
        # 연속된 공백과 줄바꿈 정리
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # 토큰 제한을 고려하여 더 적극적으로 단축 (프롬프트 토큰 절약)
        if len(content) > 4000:  # 8000에서 4000으로 줄임
            content = content[:4000] + "..."
            logger.warning("README 내용이 너무 길어서 4000자로 제한했습니다.")
        
        return content.strip()
    
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
        
        # 프롬프트 입력값 검증
        if not prompts.validate_prompt_inputs(repo_name=repo_name, readme_content=readme_content):
            logger.warning(f"프롬프트 입력값이 유효하지 않습니다: {repo_name}")
            return None
        
        try:
            # README 내용 전처리
            cleaned_content = self._clean_readme_content(readme_content)
            if not cleaned_content:
                logger.warning(f"전처리 후 README 내용이 비어있습니다: {repo_name}")
                return None
            
            # 프롬프트 생성
            prompt = prompts.get_readme_summary_prompt(repo_name, cleaned_content)
            
            # Gemini API 호출
            client = self._get_client()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"README 요약 시도 {attempt + 1}/{self.max_retries}: {repo_name}")
                    
                    response = client.models.generate_content(
                        model=self.llm_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                        )
                    )
                    
                    # 응답에서 텍스트 안전하게 추출
                    summary = None
                    
                    # 응답 구조 상세 디버깅
                    logger.debug(f"응답 타입: {type(response)}")
                    logger.debug(f"응답 속성들: {dir(response)}")
                    
                    # 응답을 JSON으로 변환하여 구조 확인
                    try:
                        if hasattr(response, '_pb'):
                            logger.debug(f"응답 _pb 구조: {response._pb}")
                        if hasattr(response, 'to_dict'):
                            response_dict = response.to_dict()
                            logger.debug(f"응답 딕셔너리: {response_dict}")
                        elif hasattr(response, '__dict__'):
                            logger.debug(f"응답 __dict__: {response.__dict__}")
                    except Exception as debug_e:
                        logger.debug(f"응답 구조 디버깅 중 오류: {debug_e}")
                    
                    # 방법 1: 직접 text 속성 확인
                    if hasattr(response, 'text') and response.text:
                        summary = response.text.strip()
                        logger.debug(f"방법 1 성공 - 직접 text 속성: {len(summary)}자")
                    
                    # 방법 2: candidates 구조에서 추출
                    elif hasattr(response, 'candidates') and response.candidates:
                        logger.debug(f"candidates 수: {len(response.candidates)}")
                        
                        for i, candidate in enumerate(response.candidates):
                            logger.debug(f"candidate {i} 속성들: {dir(candidate)}")
                            
                            # content.parts에서 텍스트 추출
                            if hasattr(candidate, 'content') and candidate.content:
                                logger.debug(f"candidate {i} content 속성들: {dir(candidate.content)}")
                                
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    logger.debug(f"candidate {i} parts 수: {len(candidate.content.parts)}")
                                    
                                    for j, part in enumerate(candidate.content.parts):
                                        logger.debug(f"part {j} 속성들: {dir(part)}")
                                        
                                        if hasattr(part, 'text') and part.text:
                                            summary = part.text.strip()
                                            logger.debug(f"방법 2 성공 - candidate {i} part {j}: {len(summary)}자")
                                            break
                                
                                # content에 직접 text가 있는 경우
                                elif hasattr(candidate.content, 'text') and candidate.content.text:
                                    summary = candidate.content.text.strip()
                                    logger.debug(f"방법 2-2 성공 - candidate {i} content.text: {len(summary)}자")
                                    break
                            
                            # candidate에 직접 text가 있는 경우
                            elif hasattr(candidate, 'text') and candidate.text:
                                summary = candidate.text.strip()
                                logger.debug(f"방법 2-3 성공 - candidate {i}.text: {len(summary)}자")
                                break
                            
                            if summary:
                                break
                    
                    # 방법 3: 문자열 변환 후 파싱
                    if not summary:
                        try:
                            response_str = str(response)
                            logger.debug(f"응답 문자열 (처음 500자): {response_str[:500]}")
                            
                            # 일반적인 패턴으로 텍스트 추출 시도
                            import re
                            
                            # "text": "내용" 패턴 찾기
                            text_match = re.search(r'"text":\s*"([^"]+)"', response_str)
                            if text_match:
                                summary = text_match.group(1).strip()
                                logger.debug(f"방법 3-1 성공 - 정규식 text 패턴: {len(summary)}자")
                            
                            # 'text': '내용' 패턴 찾기
                            elif re.search(r"'text':\s*'([^']+)'", response_str):
                                text_match = re.search(r"'text':\s*'([^']+)'", response_str)
                                summary = text_match.group(1).strip()
                                logger.debug(f"방법 3-2 성공 - 정규식 text 패턴 (단일 따옴표): {len(summary)}자")
                                
                        except Exception as str_e:
                            logger.debug(f"문자열 파싱 중 오류: {str_e}")
                    
                    # 방법 4: 최신 API 방식 시도
                    if not summary:
                        try:
                            # 최신 Gemini API에서 사용하는 방식
                            if hasattr(response, 'parts') and response.parts:
                                for part in response.parts:
                                    if hasattr(part, 'text') and part.text:
                                        summary = part.text.strip()
                                        logger.debug(f"방법 4 성공 - 직접 parts: {len(summary)}자")
                                        break
                        except Exception as parts_e:
                            logger.debug(f"parts 접근 중 오류: {parts_e}")
                    
                    if summary:
                        logger.info(f"README 요약 성공: {repo_name} ({len(summary)}자)")
                        return summary
                    else:
                        logger.warning(f"응답에서 텍스트를 추출할 수 없습니다: {repo_name}")
                        # 실제 응답 내용을 더 자세히 로그
                        logger.warning(f"응답 전체 내용: {response}")
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
            
            # 기본 설명 템플릿 가져오기
            templates = prompts.get_fallback_description_templates()
            
            # 기본 설명 템플릿
            if repo_info and repo_info.get('description'):
                return repo_info['description']
            
            # 저장소 이름 기반 기본 설명
            if any(keyword in name.lower() for keyword in ['api', 'server', 'backend']):
                return templates["api_server"].format(name=name)
            elif any(keyword in name.lower() for keyword in ['frontend', 'ui', 'web', 'app']):
                return templates["frontend"].format(name=name)
            elif any(keyword in name.lower() for keyword in ['lib', 'library', 'package']):
                return templates["library"].format(name=name)
            elif any(keyword in name.lower() for keyword in ['tool', 'cli', 'util']):
                return templates["tool"].format(name=name)
            else:
                return templates["default"].format(name=name, owner=owner)
                
        except Exception as e:
            logger.warning(f"기본 설명 생성 실패: {repo_name} - {e}")
            templates = prompts.get_fallback_description_templates()
            return templates["unknown"].format(repo_name=repo_name) 