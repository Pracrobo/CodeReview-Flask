# AIssue-BE-Flask

AIssue 프로젝트의 백엔드 Flask 애플리케이션입니다. GitHub 저장소의 코드 및 문서
를 분석하고, RAG(Retrieval Augmented Generation)를 통해 질의응답 기능을 제공합니
다.

## 주요 기능

- GitHub 저장소 인덱싱 (코드 및 문서)
- 의미론적 검색
- LLM을 활용한 질의응답 (RAG)
- RESTful API 제공

## API 사용 예시

아래 예시들은 서비스가 로컬 환경의 3002 포트에서 실행 중이라고 가정합니다. **참
고:** Windows `cmd.exe`에서 아래 curl 명령어를 실행할 경우, JSON 데이터 내의 큰
따옴표(`"`)를 `\`로 이스케이프 처리해야 할 수 있습니다. (예:
`"{ \"repo_url\": \"...\" }"`) 또는 PowerShell을 사용하거나, JSON 데이터를 파일
로 저장 후 `-d @filename.json` 형태로 사용하는 것을 권장합니다. 아래 예시는 Unix
계열 쉘 또는 PowerShell에서 바로 사용 가능합니다.

### 1. 저장소 인덱싱 요청

특정 GitHub 저장소의 인덱싱을 시작합니다.

**Unix/PowerShell:**

```bash
curl -X POST http://localhost:3002/api/repository/index \
-H "Content-Type: application/json" \
-d '{
  "repo_url": "https://github.com/pallets/flask"
}'
```

**Windows cmd.exe (큰따옴표 이스케이프):**

```bash
curl -X POST http://localhost:3002/api/repository/index ^
-H "Content-Type: application/json" ^
-d "{ \"repo_url\": \"https://github.com/pallets/flask\" }"
```

**응답 예시 (인덱싱 시작 또는 진행 중):**

```json
{
  "status": "success",
  "message": "저장소 'flask' 인덱싱이 이미 진행 중이거나 대기 중입니다. 상태 API로 확인하세요.",
  "data": {
    "status": "pending",
    "repo_url": "https://github.com/pallets/flask",
    "repo_name": "flask",
    "start_time": "2024-05-24T10:00:00.123Z",
    "last_updated_time": "2024-05-24T10:00:00.123Z",
    "end_time": null,
    "error": null,
    "error_code": null,
    "code_index_status": "pending",
    "document_index_status": "pending",
    "progress_message": "인덱싱 작업 시작 대기 중..."
  },
  "timestamp": "2024-05-24T10:00:00.456Z"
}
```

**응답 예시 (인덱싱 완료):**

```json
{
  "status": "success",
  "message": "저장소 'flask' 인덱싱이 완료되었습니다.",
  "data": {
    "status": "completed",
    "repo_url": "https://github.com/pallets/flask",
    "repo_name": "flask",
    "start_time": "2024-05-24T10:00:00.123Z",
    "last_updated_time": "2024-05-24T10:05:00.789Z",
    "end_time": "2024-05-24T10:05:00.789Z",
    "error": null,
    "error_code": null,
    "code_index_status": "completed",
    "document_index_status": "completed",
    "progress_message": "인덱싱 완료되었습니다."
  },
  "timestamp": "2024-05-24T10:05:00.999Z"
}
```

### 2. 저장소 검색 요청

인덱싱된 저장소에서 코드 또는 문서에 대해 질의합니다.

**Unix/PowerShell:**

```bash
curl -X POST http://localhost:3002/api/repository/search \
-H "Content-Type: application/json" \
-d '{
  "repo_url": "https://github.com/pallets/flask",
  "query": "Flask에서 블루프린트는 어떻게 사용하나요?",
  "search_type": "document"
}'
```

**Windows cmd.exe (큰따옴표 이스케이프):**

```bash
curl -X POST http://localhost:3002/api/repository/search ^
-H "Content-Type: application/json" ^
-d "{ \"repo_url\": \"https://github.com/pallets/flask\", \"query\": \"Flask에서 블루프린트는 어떻게 사용하나요?\", \"search_type\": \"document\" }"
```

**응답 예시:**

```json
{
  "status": "success",
  "message": "검색이 완료되었습니다.",
  "data": {
    "repo_name": "flask",
    "query": "Flask에서 블루프린트는 어떻게 사용하나요?",
    "search_type": "document",
    "answer": "Flask에서 블루프린트는 애플리케이션을 모듈화하고 라우트를 구성하는 데 사용됩니다. 예를 들어, `app.register_blueprint(my_blueprint)`와 같이 등록하여 사용할 수 있습니다...",
    "timestamp": "2024-05-24T10:10:00.123Z"
  },
  "timestamp": "2024-05-24T10:10:00.456Z"
}
```

### 3. 저장소 인덱싱 상태 확인

특정 저장소의 현재 인덱싱 상태를 확인합니다. `repo_name`은 URL에서 추출된 순수 저장소
이름입니다.

```bash
curl -X GET http://localhost:3002/api/repository/status/flask
```

**응답 예시 (진행 중):**

```json
{
  "status": "success",
  "message": "저장소 'flask' 인덱싱 진행 중입니다.",
  "data": {
    "status": "indexing",
    "repo_url": "https://github.com/pallets/flask",
    "repo_name": "flask",
    "start_time": "2024-05-24T10:00:00.123Z",
    "last_updated_time": "2024-05-24T10:02:00.456Z",
    "end_time": null,
    "error": null,
    "error_code": null,
    "code_index_status": "indexing",
    "document_index_status": "pending",
    "progress_message": "코드 파일 청크 분할 중..."
  },
  "timestamp": "2024-05-24T10:02:05.789Z"
}
```

**응답 예시 (실패):**

```json
{
  "status": "error",
  "message": "저장소 'some-repo' 인덱싱에 실패했습니다: 저장소 크기 초과",
  "error_code": "INDEXING_FAILED",
  "data": {
    "status": "failed",
    "repo_url": "https://github.com/user/some-repo",
    "repo_name": "some-repo",
    "start_time": "2024-05-24T10:14:00.000Z",
    "last_updated_time": "2024-05-24T10:15:00.123Z",
    "end_time": "2024-05-24T10:15:00.123Z",
    "error": "저장소 크기 초과: 총 코드 크기: 5.0 MB, 최대 허용: 3.0 MB",
    "error_code": "REPO_SIZE_EXCEEDED",
    "code_index_status": "failed",
    "document_index_status": "pending",
    "progress_message": "인덱싱 실패: 저장소 크기 초과: 총 코드 크기: 5.0 MB, 최대 허용: 3.0 MB"
  },
  "timestamp": "2024-05-24T10:15:00.123Z"
}
```

## 설치 및 실행

### 1. 사전 준비

- Python 3.12
- Git
- Anaconda 또는 Miniconda

### 2. 프로젝트 클론

```bash
git clone https://github.com/your-username/AIssue-BE-Flask.git
cd AIssue-BE-Flask
```

### 3. 가상 환경 생성 및 활성화 (Anaconda 사용)

Anaconda Prompt 또는 터미널에서 다음 명령어를 실행합니다.

```bash
# 'py312_AIssue'라는 이름으로 Python 3.12 가상 환경 생성
conda create -n py312_AIssue python=3.12

# 생성된 가상 환경 활성화
conda activate py312_AIssue
```

### 4. 의존성 설치

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다. (의존성 파일
`requirements.txt`가 있다고 가정)

```bash
pip install -r requirements.txt
```

_참고: `requirements.txt` 파일이 없다면, 주요 라이브러리 (`Flask`,
`python-dotenv`, `requests`, `GitPython`, `langchain`, `langchain-community`,
`google-generativeai`, `faiss-cpu` 등)를 직접 설치해야 합니다._

### 5. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 필요한 API 키를 입력합니다.

```env
# .env 파일 예시
GITHUB_API_TOKEN="your_github_api_token"  # 선택 사항, GitHub API 요청 제한 증가에 사용
GEMINI_API_KEY="your_gemini_api_key"      # 필수, Google Gemini API 사용
```

`GEMINI_API_KEY`는 [Google AI Studio](https://aistudio.google.com/app/apikey)에
서 발급받을 수 있습니다.

### 6. 애플리케이션 실행

```bash
python app.py
```

기본적으로 Flask 개발 서버는 `http://127.0.0.1:3002`에서 실행됩니다.

## 설정

애플리케이션의 주요 설정은 `config.py` 파일 및 `.env` 파일을 통해 관리됩니다.

### `.env` 파일

- `GITHUB_API_TOKEN`: GitHub API 요청 시 사용되는 토큰입니다. 공개 저장소 분석에
  는 필수는 아니지만, API 요청 제한을 늘리는 데 도움이 됩니다.
- `GEMINI_API_KEY`: Google Gemini API 사용을 위한 필수 키입니다.

### `config.py` 파일

`config.py` 파일 내 `Config` 클래스에서 다음과 같은 주요 설정들을 확인할 수 있으
며, 필요에 따라 수정할 수 있습니다.

- `DEFAULT_EMBEDDING_MODEL`: 사용할 임베딩 모델 이름.
- `DEFAULT_LLM_MODEL`: 사용할 LLM 모델 이름.
- `BASE_CLONED_DIR`: 복제된 GitHub 저장소가 저장될 로컬 디렉토리 경로.
- `FAISS_INDEX_BASE_DIR`, `FAISS_INDEX_DOCS_DIR`: 생성된 FAISS 인덱스가 저장될디
  렉토리 경로.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: 문서 및 코드 분할 시 청크 크기 및 오버랩.
- `MAX_REPO_SIZE_MB`: 인덱싱을 허용할 최대 저장소 크기 (MB 단위).
- 기타 로깅, API 재시도, 검색 파라미터 등.

대부분의 설정은 기본값으로도 잘 동작하도록 설정되어 있으나, 특정 환경이나 요구사
항에 맞춰 조정할 수 있습니다.
