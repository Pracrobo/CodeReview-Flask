from flask import request, current_app
from flask_restx import Namespace, Resource, fields

from app.services.repository_context_service import repository_context_service
from app.core.exceptions import ServiceError

# 네임스페이스 생성
chatbot_ns = Namespace("chatbot", description="챗봇 관련 API")

# 저장소 컨텍스트 기반 질문 응답 모델
repository_context_request_model = chatbot_ns.model(
    "RepositoryContextRequest",
    {
        "repo_name": fields.String(
            required=True, description="저장소 이름 (예: owner/repo_name)"
        ),
        "messages": fields.List(
            fields.Nested(
                chatbot_ns.model(
                    "Message",
                    {
                        "role": fields.String(
                            required=True, description="메시지 역할 (user/assistant)"
                        ),
                        "content": fields.String(
                            required=True, description="메시지 내용"
                        ),
                    },
                )
            ),
            required=True,
            description="대화 메시지 목록",
        ),
        "readme_filename": fields.String(description="README 파일명"),
        "license_filename": fields.String(description="LICENSE 파일명"),
        "contributing_filename": fields.String(description="CONTRIBUTING 파일명"),
    },
)

repository_context_response_model = chatbot_ns.model(
    "RepositoryContextResponse",
    {
        "answer": fields.String(description="AI 답변"),
        "context_files": fields.List(
            fields.String, description="컨텍스트로 사용된 파일 목록"
        ),
        "repo_info": fields.Raw(description="저장소 정보"),
    },
)


@chatbot_ns.route("/ask-repository")
class AskRepositoryQuestion(Resource):
    @chatbot_ns.doc("ask_repository_question")
    @chatbot_ns.expect(repository_context_request_model, validate=True)
    @chatbot_ns.marshal_with(repository_context_response_model)
    @chatbot_ns.response(400, "잘못된 요청")
    @chatbot_ns.response(404, "저장소 또는 파일을 찾을 수 없음")
    @chatbot_ns.response(500, "서버 내부 오류")
    def post(self):
        """저장소 컨텍스트를 기반으로 질문에 답변"""
        try:
            data = request.get_json()

            repo_name = data.get("repo_name")
            messages = data.get("messages")
            readme_filename = data.get("readme_filename")
            license_filename = data.get("license_filename")
            contributing_filename = data.get("contributing_filename")

            if (
                not repo_name
                or not messages
                or not isinstance(messages, list)
                or len(messages) == 0
            ):
                chatbot_ns.abort(
                    400,
                    "repo_name과 messages(배열)는 필수 항목이며, 비어있을 수 없습니다.",
                )

            # 마지막 사용자 메시지를 질문으로 사용
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                chatbot_ns.abort(400, "사용자 메시지가 필요합니다.")

            # 가장 최근의 사용자 메시지를 질문으로 사용
            question = user_messages[-1].get("content", "")
            if not question.strip():
                chatbot_ns.abort(400, "질문 내용이 비어있습니다.")

            # 저장소 컨텍스트 기반 질문 답변
            result = repository_context_service.answer_question_with_context(
                repo_name=repo_name,
                question=question,
                readme_filename=readme_filename,
                license_filename=license_filename,
                contributing_filename=contributing_filename,
                messages=messages,
            )

            return result

        except FileNotFoundError as e:
            current_app.logger.warning(f"파일 찾기 오류 (저장소 컨텍스트 질문): {e}")
            chatbot_ns.abort(404, str(e))
        except ServiceError as e:
            current_app.logger.error(
                f"서비스 오류 (저장소 컨텍스트 질문): {e}", exc_info=True
            )
            chatbot_ns.abort(500, str(e))
        except Exception as e:
            current_app.logger.error(
                f"저장소 컨텍스트 질문 답변 중 예상치 못한 오류: {e}", exc_info=True
            )
            chatbot_ns.abort(500, "서버 내부 오류가 발생했습니다.")
