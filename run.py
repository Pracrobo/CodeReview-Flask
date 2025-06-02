import os
from app import create_app

app = create_app()


def main():
    """메인 실행 함수"""

    # 환경변수에서 설정 읽기
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 3002))
    debug = os.getenv("FLASK_DEBUG", "true").lower() in ("true", "1", "yes", "on")

    print("Flask 애플리케이션 시작:")
    print(f"  - Host: {host}")
    print(f"  - Port: {port}")
    print(f"  - Debug: {debug}")
    print(f"  - Swagger UI: http://{host}:{port}/docs/")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
