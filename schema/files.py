from fileinput import filename
from typing import List
from pydantic import BaseModel



class FileListResponse(BaseModel):
    filename: str
    file_id: str
    path : str



class DeleteFileResponse(BaseModel):
    message: str
    file_id: str