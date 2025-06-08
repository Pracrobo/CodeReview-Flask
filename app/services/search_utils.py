import logging
import numpy as np
import re

from app.core.prompts import prompts
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)


def translate_code_query_to_english(korean_text, llm_model_name):
    """코드 관련 한국어 질의 영어 번역"""
    try:
        client = gemini_service.get_client()
    except Exception:
        logger.error("Gemini 클라이언트가 초기화되지 않아 번역을 건너뜁니다.")
        return korean_text
    try:
        prompt = prompts.get_code_query_translation_prompt(korean_text)
        response = client.models.generate_content(model=llm_model_name, contents=prompt)
        english_text = gemini_service.extract_text_from_response(response)
        logger.info(f"코드 질의 번역 완료: '{korean_text}' -> '{english_text}'")
        return english_text if english_text else korean_text
    except Exception as e:
        logger.warning(f"코드 질의 번역 실패, 원본 텍스트 사용: {e}")
        return korean_text


def preprocess_text(text):
    """임베딩 입력 전처리(공백, 특수문자, 소문자 변환 등)"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text


def normalize_vector(vec):
    """벡터 정규화"""
    v = np.array(vec)
    norm = np.linalg.norm(v) + 1e-8
    return v / norm


def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    return float(np.dot(v1, v2))
