import os
from dotenv import load_dotenv
from langchain_text_splitters import Language

# .env 환경 변수 로드
load_dotenv()


class Config:
    """애플리케이션 설정 클래스"""

    # API 키
    GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
    GEMINI_API_KEY1 = os.getenv("GEMINI_API_KEY1")
    GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2")

    # 모델 이름 (지정된 최신 모델, 변경 주의)
    DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
    DEFAULT_LLM_MODEL = "gemini-2.5-flash-preview-05-20"

    # 디렉토리 경로
    BASE_CLONED_DIR = "./cloned_repos"
    FAISS_INDEX_BASE_DIR = "faiss_indexes"
    FAISS_INDEX_DOCS_DIR = "faiss_indexes_docs"

    # 인덱싱 및 임베딩 설정
    CHUNK_SIZE = 4000  # RecursiveCharacterTextSplitter 청크 크기 (문서 분할 시)
    CHUNK_OVERLAP = 400  # RecursiveCharacterTextSplitter 청크 오버랩 (문서 분할 시)
    EMBEDDING_BATCH_SIZE = 50  # Gemini API 임베딩 요청 시 배치 크기
    FAISS_ADD_BATCH_SIZE = 100  # FAISS 인덱스에 문서 추가 시 배치 크기
    MAX_RETRIES = 10  # API 호출 최대 재시도 횟수
    EMBEDDING_DIMENSION = 768  # 임베딩 벡터 차원 설정

    # API 오류별 대기 시간 설정 (초 단위)
    QUOTA_ERROR_SLEEP_TIME = 30  # 할당량 오류 시 대기 시간
    GENERAL_API_ERROR_SLEEP_TIME = 5  # 일반 API 오류 시 대기 시간

    # 문서 파일 확장자
    DOCUMENT_FILE_EXTENSIONS = (".md", ".markdown", ".rst", ".txt")

    # 로깅 설정
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 검색 설정
    DEFAULT_TOP_K = 5  # 코드 검색을 위해 증가
    DEFAULT_SIMILARITY_THRESHOLD = 0.4  # 코드 검색을 위해 완화

    # API 서비스 설정 추가
    API_VERSION = "v1"
    MAX_REPOSITORIES = 100  # 최대 저장소 개수 제한
    REQUEST_TIMEOUT = 300  # 요청 타임아웃 (초)
    MAX_REPO_SIZE_MB = 3  # 최대 저장소 크기 (MB)


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
    raise ValueError(
        "GEMINI_API_KEY1 또는 GEMINI_API_KEY2를 .env 파일이나 환경 변수에서 찾을 수 없습니다."
    )
