import os
from dotenv import load_dotenv, dotenv_values
from langchain_text_splitters import Language
import warnings

# .env 환경 변수 로드 - 명시적 경로 지정
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(dotenv_path=env_path)

# .env 파일에서 환경 변수 직접 로드
env_vars = dotenv_values(dotenv_path=env_path)


class Config:
    """애플리케이션 설정 클래스"""

    # API 키 - .env 파일에서 직접 로드
    GITHUB_API_TOKEN = env_vars.get("GITHUB_API_TOKEN") or os.getenv("GITHUB_API_TOKEN")
    GEMINI_API_KEY1 = env_vars.get("GEMINI_API_KEY1") or os.getenv("GEMINI_API_KEY1")
    GEMINI_API_KEY2 = env_vars.get("GEMINI_API_KEY2") or os.getenv("GEMINI_API_KEY2")

    # 모델 이름 (지정된 최신 모델, 변경 주의)
    DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
    DEFAULT_LLM_MODEL = "gemini-2.5-flash-preview-05-20"

    # 디렉토리 경로
    BASE_CLONED_DIR = "./cloned_repos"
    FAISS_INDEX_BASE_DIR = "faiss_indexes"

    # 인덱싱 및 임베딩 설정
    CHUNK_SIZE = 2000  # 의미 단위로 분할
    CHUNK_OVERLAP = 500  # 문맥 보존 강화
    EMBEDDING_BATCH_SIZE = 50  # Gemini API 임베딩 요청 시 배치 크기
    FAISS_ADD_BATCH_SIZE = 100  # FAISS 인덱스에 문서 추가 시 배치 크기
    MAX_RETRIES = 10  # API 호출 최대 재시도 횟수
    EMBEDDING_DIMENSION = 1024  # 임베딩 벡터 차원 설정

    # API 오류별 대기 시간 설정 (초 단위)
    QUOTA_ERROR_SLEEP_TIME = 5  # 할당량 오류 시 대기 시간
    GENERAL_API_ERROR_SLEEP_TIME = 3  # 일반 API 오류 시 대기 시간

    # 로깅 설정
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 검색 설정
    DEFAULT_TOP_K = 10
    DEFAULT_SIMILARITY_THRESHOLD = 0.3

    # API 서비스 설정 추가
    API_VERSION = "v1"
    MAX_REPOSITORIES = 100  # 최대 저장소 개수 제한
    REQUEST_TIMEOUT = 300  # 요청 타임아웃 (초)
    MAX_REPO_SIZE_MB = 5  # 최대 저장소 크기 (MB)


# 지원 언어, 확장자, Langchain Enum 매핑
LANGUAGE_TO_DETAILS = {
    "python": {"ext": ".py", "lang_enum": Language.PYTHON},
    "javascript": {"ext": ".js", "lang_enum": Language.JS},
    "typescript": {"ext": ".ts", "lang_enum": Language.TS},
    "java": {"ext": ".java", "lang_enum": Language.JAVA},
    "c++": {"ext": ".cpp", "lang_enum": Language.CPP},
    "c": {"ext": ".c", "lang_enum": Language.C},
    "c#": {"ext": ".cs", "lang_enum": Language.CSHARP},
    "go": {"ext": ".go", "lang_enum": Language.GO},
    "ruby": {"ext": ".rb", "lang_enum": Language.RUBY},
    "php": {"ext": ".php", "lang_enum": Language.PHP},
    "swift": {"ext": ".swift", "lang_enum": Language.SWIFT},
    "kotlin": {"ext": ".kt", "lang_enum": Language.KOTLIN},
    "rust": {"ext": ".rs", "lang_enum": Language.RUST},
    "scala": {"ext": ".scala", "lang_enum": Language.SCALA},
    "html": {"ext": ".html", "lang_enum": Language.HTML},
    "markdown": {"ext": ".md", "lang_enum": Language.MARKDOWN},
    "solidity": {"ext": ".sol", "lang_enum": Language.SOL},
    # 추가 언어 지원 시 여기에 추가
}

if not Config.GEMINI_API_KEY1 or not Config.GEMINI_API_KEY2:
    warnings.warn(
        "Gemini API 키(GEMINI_API_KEY1 또는 GEMINI_API_KEY2)가 .env 파일이나 환경 변수에 설정되어 있지 않습니다. "
        "일부 기능(예: 임베딩) 사용이 제한될 수 있습니다.",
        UserWarning,
    )
    # API 키가 없어도 서버가 시작되도록 기본값 설정
    Config.GEMINI_API_KEY1 = Config.GEMINI_API_KEY1 or "dummy_key_1"
    Config.GEMINI_API_KEY2 = Config.GEMINI_API_KEY2 or "dummy_key_2"
