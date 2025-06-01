# This file makes the 'api' directory a Python package. 

from flask import Blueprint
from flask_restx import Api

# Blueprint 생성
api_bp = Blueprint('api', __name__)

# Flask-RESTX Api 객체 생성 및 Blueprint에 연결
# doc='/docs/'를 설정하여 Swagger UI 경로를 /docs/ 로 만듭니다.
api = Api(api_bp, 
          version='1.0', 
          title='AIssue Repository RAG 분석 API',
          description='GitHub 저장소의 코드와 문서를 분석하여 RAG 기반 검색을 제공하는 API',
          doc='/docs/')

# 네임스페이스 임포트 및 등록
from .repository_api import repository_ns # repository_api.py 에서 정의된 네임스페이스
api.add_namespace(repository_ns, path='/repository')

# 다른 네임스페이스가 있다면 여기에 추가
# from .another_api import another_ns
# api.add_namespace(another_ns, path='/another')

# 공통 모델 정의 (swagger_config.py의 내용을 이곳으로 옮기거나, 여기서 직접 정의)
# 예시: base_response = api.model(...) 
# 현재는 swagger_config.py의 내용을 repository_api.py에서 직접 임포트하도록 유지하고,
# 추후 필요시 공통 모델들을 이 파일로 옮기는 것을 고려할 수 있습니다. 