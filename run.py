from app import create_app

app = create_app()

if __name__ == '__main__':
    # host, port, debug 설정은 config 파일 또는 환경변수에서 관리하는 것이 좋습니다.
    # 예: app.run(host=app.config.get('HOST', '0.0.0.0'), 
    #             port=app.config.get('PORT', 5000), 
    #             debug=app.config.get('DEBUG', False))
    app.run(debug=True) # 개발 중에는 True로 설정 