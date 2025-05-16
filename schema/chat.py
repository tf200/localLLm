

from pydantic import BaseModel




# 5. Input schema
class QuestionInput(BaseModel):
    message: str