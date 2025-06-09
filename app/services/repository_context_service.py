import logging
import os
from typing import Dict, Optional, Any

from app.core.config import Config
from app.core.exceptions import ServiceError
from app.core.prompts import prompts
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)


class RepositoryContextService:
    """저장소 컨텍스트 기반 질문 답변 서비스"""

    def __init__(self):
        self.llm_model = Config.DEFAULT_LLM_MODEL

    def answer_question_with_context(
        self,
        repo_id: int,
        question: str,
        readme_filename: Optional[str] = None,
        license_filename: Optional[str] = None,
        contributing_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        저장소 파일들을 컨텍스트로 하여 질문에 답변

        Args:
            repo_id: 저장소 ID
            question: 질문 내용
            readme_filename: README 파일명
            license_filename: LICENSE 파일명
            contributing_filename: CONTRIBUTING 파일명

        Returns:
            답변 결과 딕셔너리
        """
        try:
            logger.info(f"저장소 컨텍스트 질문 답변 시작: repo_id={repo_id}")

            # 저장소 정보 조회 (실제 구현시 DB에서 조회)
            repo_info = self._get_repository_info_from_db(repo_id)
            repo_name = repo_info.get("full_name", f"repo_{repo_id}")

            # 저장소 파일들 읽기
            context_files = []
            file_contents = {}

            # README 파일 읽기
            if readme_filename:
                readme_content = self._read_repository_file(repo_name, readme_filename)
                if readme_content:
                    file_contents["README"] = readme_content
                    context_files.append(readme_filename)

            # LICENSE 파일 읽기
            if license_filename:
                license_content = self._read_repository_file(
                    repo_name, license_filename
                )
                if license_content:
                    file_contents["LICENSE"] = license_content
                    context_files.append(license_filename)

            # CONTRIBUTING 파일 읽기
            if contributing_filename:
                contributing_content = self._read_repository_file(
                    repo_name, contributing_filename
                )
                if contributing_content:
                    file_contents["CONTRIBUTING"] = contributing_content
                    context_files.append(contributing_filename)

            if not file_contents:
                raise FileNotFoundError("읽을 수 있는 저장소 파일이 없습니다.")

            # Gemini로 질문 답변 생성
            answer = self._generate_answer_with_context(
                question=question, repo_info=repo_info, file_contents=file_contents
            )

            result = {
                "answer": answer,
                "context_files": context_files,
                "repo_info": {
                    "repo_id": repo_id,
                    "name": repo_info.get("name", ""),
                    "full_name": repo_name,
                    "description": repo_info.get("description", ""),
                },
            }

            logger.info(f"저장소 컨텍스트 질문 답변 완료: repo_id={repo_id}")
            return result

        except Exception as e:
            logger.error(f"저장소 컨텍스트 질문 답변 중 오류: {e}", exc_info=True)
            raise ServiceError(f"질문 답변 실패: {e}") from e

    def _get_repository_info_from_db(self, repo_id: int) -> Dict[str, Any]:
        """DB에서 저장소 정보 조회"""
        try:
            # 실제로는 SQLAlchemy나 다른 ORM을 사용하여 DB에서 조회
            # 여기서는 간단히 파일 시스템에서 추정

            # 클론된 저장소 목록에서 repo_id에 해당하는 저장소 찾기
            cloned_dirs = os.listdir(Config.BASE_CLONED_DIR)

            # repo_id를 통해 저장소를 식별하는 로직이 필요
            # 임시로 첫 번째 디렉토리 사용 (실제로는 DB 연동 필요)
            if cloned_dirs:
                for dir_name in cloned_dirs:
                    full_path = os.path.join(Config.BASE_CLONED_DIR, dir_name)
                    if os.path.isdir(full_path):
                        return {
                            "repo_id": repo_id,
                            "name": (
                                dir_name.split("/")[-1] if "/" in dir_name else dir_name
                            ),
                            "full_name": dir_name,
                            "description": f"{dir_name} 저장소",
                        }

            # 기본값 반환
            return {
                "repo_id": repo_id,
                "name": f"repository_{repo_id}",
                "full_name": f"owner/repository_{repo_id}",
                "description": "저장소 설명",
            }

        except Exception as e:
            logger.warning(f"저장소 정보 조회 실패: {e}")
            return {
                "repo_id": repo_id,
                "name": f"repository_{repo_id}",
                "full_name": f"owner/repository_{repo_id}",
                "description": "저장소 설명",
            }

    def _read_repository_file(self, repo_name: str, filename: str) -> Optional[str]:
        """저장소에서 특정 파일 내용 읽기"""
        try:
            # 클론된 저장소 경로 구성
            cloned_repo_path = os.path.join(Config.BASE_CLONED_DIR, repo_name)
            file_path = os.path.join(cloned_repo_path, filename)

            if not os.path.exists(file_path):
                logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
                return None

            # 파일 크기 확인 (너무 큰 파일은 제외)
            file_size = os.path.getsize(file_path)
            max_file_size = 500 * 1024  # 500KB 제한

            if file_size > max_file_size:
                logger.warning(f"파일이 너무 큽니다 ({file_size} bytes): {file_path}")
                return f"[파일이 너무 큽니다 - {file_size:,} bytes]"

            # 파일 내용 읽기
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 내용 길이 제한
            if len(content) > 50000:  # 50,000자 제한
                content = content[:50000] + "\n\n[내용이 잘렸습니다...]"

            logger.info(f"파일 읽기 성공: {filename} ({len(content)} 문자)")
            return content

        except Exception as e:
            logger.error(f"파일 읽기 실패 ({filename}): {e}")
            return None

    def _generate_answer_with_context(
        self, question: str, repo_info: Dict[str, Any], file_contents: Dict[str, str]
    ) -> str:
        """컨텍스트를 기반으로 질문에 대한 답변 생성"""
        try:
            client = gemini_service.get_client()
            if not client:
                logger.error(
                    "질문 답변 생성 실패: Gemini 클라이언트를 초기화할 수 없습니다."
                )
                return "AI 답변 생성 중 오류가 발생했습니다."

            prompt = prompts.get_repository_context_answer_prompt(
                question=question, repo_info=repo_info, file_contents=file_contents
            )

            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )

            answer = gemini_service.extract_text_from_response(response)
            return answer or "AI 답변을 생성할 수 없습니다."

        except Exception as e:
            logger.warning(f"질문 답변 생성 실패: {e}")
            return "AI 답변 생성 중 오류가 발생했습니다."


# 전역 인스턴스
repository_context_service = RepositoryContextService()
