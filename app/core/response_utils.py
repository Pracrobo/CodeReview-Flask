from datetime import datetime, timezone


def get_iso_timestamp():
    """현재 시간을 ISO 8601 형식의 UTC 타임스탬프로 반환합니다."""
    return datetime.now(timezone.utc).isoformat()


def success_response(
    data=None, message="요청이 성공적으로 처리되었습니다.", status_code=200
):
    """성공 응답을 생성합니다."""
    response_data = {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": get_iso_timestamp(),
    }
    return response_data, status_code


def error_response(message, error_code=None, status_code=500):
    """오류 응답을 생성합니다."""
    response_data = {
        "status": "error",
        "message": message,
        "error_code": error_code,
        "timestamp": get_iso_timestamp(),
    }
    return response_data, status_code
