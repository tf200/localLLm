from typing import List
from pydantic import BaseModel



class FileListResponse(BaseModel):
    path: str
    files: List[str]
    directories: List[str]