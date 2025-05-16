from fastapi import APIRouter
from schema.chat import QuestionInput
from sse_starlette.sse import EventSourceResponse
from util.llm import stream_answer


router = APIRouter()


@router.post("/chat/send")
async def ask(question_input: QuestionInput):
    return EventSourceResponse(stream_answer(question_input.message))