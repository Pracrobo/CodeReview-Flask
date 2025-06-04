"""텍스트 번역 서비스 모듈"""

import logging
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import Config
from app.core.exceptions import ServiceError
from app.core.prompts import prompts

logger = logging.getLogger(__name__)


class Translator:
    """텍스트 번역 서비스"""
    
    def __init__(self):
        """번역 서비스 초기화"""
        self.llm_model = Config.DEFAULT_LLM_MODEL
        # 프롬프트 설정에서 값 가져오기
        prompt_config = prompts.get_prompt_config()
        self.max_retries = prompt_config["translation"]["retry_count"] if "retry_count" in prompt_config["translation"] else 3
        self.temperature = prompt_config["translation"]["temperature"]
        self.max_output_tokens = prompt_config["translation"]["max_output_tokens"]
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
                logger.info("번역용 Gemini 클라이언트 초기화 완료")
            except Exception as e:
                raise ServiceError(f"Gemini API 클라이언트 생성 실패: {e}")
        
        return self._client
    
    def translate_text(self, text: str, source_language: str = "auto", target_language: str = "ko") -> Optional[str]:
        """텍스트를 번역합니다.
        
        Args:
            text: 번역할 텍스트
            source_language: 원본 언어 (기본값: auto)
            target_language: 대상 언어 (기본값: ko)
            
        Returns:
            번역된 텍스트 또는 None (실패 시)
        """
        if not text or not text.strip():
            logger.warning("번역할 텍스트가 비어있습니다.")
            return None
        
        # 이미 한국어인 경우 번역하지 않음
        if self._is_korean_text(text) and target_language == "ko":
            logger.info("이미 한국어 텍스트로 판단되어 번역을 생략합니다.")
            return text
        
        try:
            # 번역 프롬프트 생성
            prompt = self._get_translation_prompt(text, source_language, target_language)
            
            # Gemini API 호출
            client = self._get_client()
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"번역 시도 {attempt + 1}/{self.max_retries}: {len(text)}자")
                    
                    response = client.models.generate_content(
                        model=self.llm_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                        )
                    )
                    
                    # 응답에서 텍스트 추출
                    translated_text = self._extract_text_from_response(response)
                    
                    if translated_text:
                        logger.info(f"번역 성공: {len(text)}자 -> {len(translated_text)}자")
                        return translated_text.strip()
                    else:
                        logger.warning(f"번역 응답에서 텍스트를 추출할 수 없습니다.")
                        continue
                        
                except Exception as e:
                    logger.warning(f"번역 시도 {attempt + 1} 실패: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    
                    # 재시도 전 잠시 대기
                    import time
                    time.sleep(2 ** attempt)  # 지수 백오프
            
            logger.error("모든 번역 재시도 실패")
            return None
            
        except Exception as e:
            logger.error(f"번역 중 오류 발생: {e}")
            return None
    
    def _get_translation_prompt(self, text: str, source_language: str, target_language: str) -> str:
        """번역 프롬프트 생성"""
        if target_language == "ko":
            return f"""다음 텍스트를 자연스러운 한국어로 번역해주세요. 기술적 용어는 적절히 번역하되, 널리 알려진 용어는 원문을 유지하세요.
번역된 텍스트만 출력하고 다른 설명은 하지 마세요.

텍스트: {text}

번역:"""
        else:
            return f"""다음 텍스트를 {target_language}로 번역해주세요. 번역된 텍스트만 출력하고 다른 설명은 하지 마세요.

텍스트: {text}

번역:"""
    
    def _extract_text_from_response(self, response) -> Optional[str]:
        """Gemini API 응답에서 텍스트 추출"""
        try:
            # 방법 1: 직접 text 속성 확인
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # 방법 2: candidates 구조에서 추출
            elif hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    return part.text.strip()
                        elif hasattr(candidate.content, 'text') and candidate.content.text:
                            return candidate.content.text.strip()
                    elif hasattr(candidate, 'text') and candidate.text:
                        return candidate.text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"응답 텍스트 추출 중 오류: {e}")
            return None
    
    def _is_korean_text(self, text: str) -> bool:
        """텍스트가 한국어인지 간단히 판단"""
        if not text:
            return False
        
        # 한글 문자가 전체 텍스트의 30% 이상이면 한국어로 판단
        korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7af')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return False
        
        korean_ratio = korean_chars / total_chars
        return korean_ratio >= 0.3 