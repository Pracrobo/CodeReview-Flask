import os
from app import create_app

app = create_app()


def main():
    """메인 실행 함수"""

    # 환경변수에서 설정 읽기
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 3002))
    debug = os.getenv("FLASK_DEBUG", "true").lower() in ("true", "1", "yes", "on")

    app.logger.info("플라스크 애플리케이션을 시작합니다:")
    app.logger.info(f"  - 호스트: {host}")
    app.logger.info(f"  - 포트: {port}")
    app.logger.info(f"  - 디버그 모드: {debug}")
    app.logger.info(f"  - Swagger UI: http://{host}:{port}/docs/")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
