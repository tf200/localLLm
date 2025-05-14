import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import chromadb
import os
from typing import List
from fastapi.exceptions import HTTPException
import json
import uuid
# 1. Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Allow requests from any origin (you can restrict this to specific domains)
    allow_origins=["*"],
    # Allow credentials (cookies, authorization headers)
    allow_credentials=True,
    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    # Allow all headers
    allow_headers=["*"],
)


# 2. Initialize ChromaDB client + collection
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("Micla")

# 3. Define system prompt
system_prompt = """
You are an expert assistant with access to a knowledge base.
Your job is to provide accurate, helpful answers based on the context provided in each query.
Format your response using Markdown for better readability (use headings, bold, italic, lists, code blocks, etc. as appropriate).
If the user's question is not covered by the provided context documents, say "I don't know."

IMPORTANT: At the end of your response, include a section titled "Sources Used" that lists the metadata 
of all the documents you referenced in your answer. This helps with attribution and transparency.
/no_think"""

# 4. Initialize your LLM agent
model = OpenAIModel(
    "Qwen3-4B",
    provider=OpenAIProvider(base_url="http://localhost:8080/v1"),
)
agent = Agent(model=model, system_prompt=system_prompt)

# 5. Input schema
class QuestionInput(BaseModel):
    message: str

# 6. SSE Streaming logic
# Update the stream_answer function to format SSE messages correctly
async def stream_answer(question: str):
    # Retrieve top-5 docs
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    context_with_metadata = []
    for i, (doc, meta) in enumerate(zip(docs, metadata)):
        doc_with_meta = f"DOCUMENT {i+1}:\n{doc}\nMETADATA: {meta}"
        context_with_metadata.append(doc_with_meta)

    context = "\n\n---\n\n".join(context_with_metadata)

    user_prompt = f"""
Use the following retrieved documents to answer my question.

Context:
{context}

Question:
{question}

Remember to include a "Sources Used" section at the end of your response that lists the metadata of documents you referenced.
"""

    full_response = ""
    message_id = str(uuid.uuid4())  # You'll need to import uuid and generate this
    
    async with agent.run_stream(user_prompt) as result:
        async for delta in result.stream_text(delta=True):
            if delta:  # Only add non-empty deltas
                full_response += delta
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "delta": delta,
                            "message_id": message_id,
                            "chat_id": "current_chat_id",  # You'll need to replace this with actual chat_id
                        }
                    ),
                }
    
    # Send end event
    yield {
        "event": "done",
        "data": json.dumps({"text": ""})
    }

@app.post("/chat/send")
async def ask(question_input: QuestionInput):
    return EventSourceResponse(stream_answer(question_input.message))



FOLDER_PATH = "/files"  # Change this to your desired folder

class FileListResponse(BaseModel):
    path: str
    files: List[str]
    directories: List[str]


@app.get("/list-files", response_model=FileListResponse)
async def list_files():
    """
    List all files and directories in the hard-coded folder.
    
    Returns:
    - JSON with path, files list, and directories list
    """
    try:
        print("=== Starting list_files endpoint execution ===")
        print(f"Current working directory: {os.getcwd()}")
        
        # Define FOLDER_PATH correctly for Windows
        # Use a relative path from the current working directory
        FOLDER_PATH = os.path.join(os.getcwd(), "files")
        print(f"FOLDER_PATH set to: {FOLDER_PATH}")
            
        # Verify the folder exists
        print(f"Checking if directory exists: {FOLDER_PATH}")
        if not os.path.exists(FOLDER_PATH):
            print(f"ERROR: Directory does not exist: {FOLDER_PATH}")
            raise HTTPException(status_code=404, detail=f"Directory not found: {FOLDER_PATH}")
            
        print(f"Checking if path is a directory: {FOLDER_PATH}")
        if not os.path.isdir(FOLDER_PATH):
            print(f"ERROR: Path is not a directory: {FOLDER_PATH}")
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {FOLDER_PATH}")
        
        print(f"Directory checks passed. Absolute path: {os.path.abspath(FOLDER_PATH)}")
        
        files = []
        directories = []
        
        try:
            print("Attempting to use os.scandir for directory listing")
            # Try scandir first (more efficient)
            with os.scandir(FOLDER_PATH) as entries:
                for entry in entries:
                    try:
                        if entry.is_file():
                            files.append(entry.name)
                        elif entry.is_dir():
                            directories.append(entry.name)
                    except Exception as entry_error:
                        print(f"Error processing entry {entry.name}: {str(entry_error)}")
        except AttributeError as ae:
            # Fallback to listdir if scandir not available
            print(f"os.scandir not available, falling back to os.listdir: {str(ae)}")
            try:
                for entry in os.listdir(FOLDER_PATH):
                    full_path = os.path.join(FOLDER_PATH, entry)
                    if os.path.isfile(full_path):
                        files.append(entry)
                    elif os.path.isdir(full_path):
                        directories.append(entry)
            except Exception as listdir_error:
                print(f"Error using os.listdir: {str(listdir_error)}")
                raise
        except Exception as se:
            print(f"Error during directory scanning: {str(se)}")
            raise
        
        print(f"Directory scan complete. Found {len(files)} files and {len(directories)} directories")
        
        result = {
            "path": os.path.abspath(FOLDER_PATH),
            "files": sorted(files),
            "directories": sorted(directories)
        }
        
        print("=== list_files endpoint execution completed successfully ===")
        return result
    
    except PermissionError as pe:
        error_msg = f"Permission denied for directory: {FOLDER_PATH}"
        print(f"ERROR: {error_msg}")
        print(f"Exception details: {str(pe)}")
        raise HTTPException(status_code=403, detail=error_msg)
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        error_msg = f"Internal server error in list_files: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Exception type: {type(e)._name_}")
        print(f"Exception args: {e.args}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)