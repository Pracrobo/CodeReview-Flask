"""공통 예외 클래스 정의"""


class EmbeddingError(Exception):
    """임베딩 관련 오류를 위한 사용자 정의 예외 클래스"""

    pass


class RepositoryError(Exception):
    """저장소 관련 오류를 위한 사용자 정의 예외 클래스"""

    pass


class IndexingError(Exception):
    """인덱싱 관련 오류를 위한 사용자 정의 예외 클래스"""

    pass


class RepositorySizeError(IndexingError):
    """저장소 크기 초과 오류를 위한 사용자 정의 예외 클래스"""

    pass


class RAGError(Exception):
    """RAG 관련 오류를 위한 사용자 정의 예외 클래스"""

    pass


class ServiceError(Exception):
    """서비스 레벨 오류를 위한 사용자 정의 예외 클래스"""

    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code


class ValidationError(ValueError):
    """입력 값 검증 오류를 위한 사용자 정의 예외 클래스"""

    pass
