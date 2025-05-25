from flask import jsonify
from datetime import datetime, timezone
from typing import Any, Optional, Dict


def get_iso_timestamp() -> str:
    """현재 시간을 ISO 8601 형식의 UTC 타임스탬프로 반환합니다."""
    return datetime.now(timezone.utc).isoformat()


def success_response(
    data: Optional[Any] = None,
    message: str = "요청이 성공적으로 처리되었습니다.",
    status_code: int = 200,
) -> tuple[Dict[str, Any], int]:
    """성공 응답을 생성합니다."""
    response_data = {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": get_iso_timestamp(),
    }
    return jsonify(response_data), status_code


def error_response(
    message: str, error_code: Optional[str] = None, status_code: int = 500
) -> tuple[Dict[str, Any], int]:
    """오류 응답을 생성합니다."""
    response_data = {
        "status": "error",
        "message": message,
        "error_code": error_code,
        "timestamp": get_iso_timestamp(),
    }
    return jsonify(response_data), status_code


def in_progress_response(
    progress: int, message: str = "요청이 처리 중입니다.", status_code: int = 202
) -> tuple[Dict[str, Any], int]:
    """진행률 응답을 생성합니다."""
    response_data = {
        "status": "in_progress",
        "progress": progress,
        "message": message,
        "timestamp": get_iso_timestamp(),
    }
    return jsonify(response_data), status_code
