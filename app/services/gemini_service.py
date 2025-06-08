import logging
from google import genai
from google.genai.types import HttpOptions
from app.core.config import Config
from app.core.exceptions import ServiceError

logger = logging.getLogger(__name__)


class GeminiService:
    """Gemini API 공통 서비스"""

    def __init__(self):
        self._client = None

    def get_client(self):
        """Gemini API 클라이언트 반환 (온디멘드 결제, 요청 시 shared 헤더 사용)"""
        if self._client is None:
            try:
                api_key = Config.GEMINI_API_KEY1
                if not api_key or api_key == "dummy_key_1":
                    api_key = Config.GEMINI_API_KEY2
                    if not api_key or api_key == "dummy_key_2":
                        raise ServiceError(
                            "유효한 Gemini API 키가 설정되지 않았습니다."
                        )
                # 클라이언트는 별도 옵션 없이 생성
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=HttpOptions(
                        headers={"X-Vertex-AI-LLM-Request-Type": "shared"}
                    ),
                )
                logger.info("Gemini 클라이언트(온디멘드) 초기화 완료")
            except Exception as e:
                raise ServiceError(f"Gemini API 클라이언트 생성 실패: {e}")
        return self._client

    def get_client_with_key(self, api_key):
        """특정 API 키로 클라이언트 반환 (온디멘드 결제, 요청 시 shared 헤더 사용)"""
        try:
            return genai.Client(
                api_key=api_key,
                http_options=HttpOptions(
                    headers={"X-Vertex-AI-LLM-Request-Type": "shared"}
                ),
            )
        except Exception as e:
            raise ServiceError(f"Gemini API 클라이언트 생성 실패: {e}")

    def get_available_api_keys(self):
        """사용 가능한 API 키 리스트 반환"""
        keys = []
        if Config.GEMINI_API_KEY1 and Config.GEMINI_API_KEY1 != "dummy_key_1":
            keys.append(Config.GEMINI_API_KEY1)
        if Config.GEMINI_API_KEY2 and Config.GEMINI_API_KEY2 != "dummy_key_2":
            keys.append(Config.GEMINI_API_KEY2)
        return keys

    def extract_text_from_response(self, response):
        """Gemini 응답에서 텍스트 추출 (공통)"""
        try:
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if (
                            hasattr(candidate.content, "parts")
                            and candidate.content.parts
                        ):
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    return part.text.strip()
                        elif (
                            hasattr(candidate.content, "text")
                            and candidate.content.text
                        ):
                            return candidate.content.text.strip()
                    elif hasattr(candidate, "text") and candidate.text:
                        return candidate.text.strip()
            return None
        except Exception as e:
            logger.error(f"응답 텍스트 추출 중 오류: {e}")
            return None


# 전역 인스턴스
gemini_service = GeminiService()
