from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
from schema.files import FileListResponse
from typing import List
import shutil
import uuid
from util.vector_store import prepare_document_for_rag, collection






router= APIRouter()



@router.get("/list-files", response_model=FileListResponse)
async def list_files():
    """
    List all files and directories in the hard-coded folder.
    
    Returns:
    - JSON with path, files list, and directories list
    """
    try:
        
        FOLDER_PATH = os.path.join(os.getcwd(), "files")
            
        if not os.path.exists(FOLDER_PATH):
            raise HTTPException(status_code=404, detail=f"Directory not found: {FOLDER_PATH}")
            
        if not os.path.isdir(FOLDER_PATH):
            print(f"ERROR: Path is not a directory: {FOLDER_PATH}")
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {FOLDER_PATH}")
        
        
        files = []
        directories = []
        
        try:
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
        except Exception:
            raise
        
        
        result = {
            "path": os.path.abspath(FOLDER_PATH),
            "files": sorted(files),
            "directories": sorted(directories)
        }
        
        return result
    
    except PermissionError:
        error_msg = f"Permission denied for directory: {FOLDER_PATH}"
        raise HTTPException(status_code=403, detail=error_msg)
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        error_msg = f"Internal server error in list_files: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
    





@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    total_chunks_processed = 0

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Save file to disk
        file_path = os.path.join("files", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append({
            "filename": file.filename,
            "path": file_path,
            "content_type": file.content_type
        })

        # Embed the uploaded file into ChromaDB
        try:
            texts, metadatas = prepare_document_for_rag(file_path)
            ids = [str(uuid.uuid4()) for _ in texts]
            if texts:
                collection.add(documents=texts, metadatas=metadatas, ids=ids) # type: ignore
                total_chunks_processed += len(texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error embedding {file.filename}: {str(e)}")

    return JSONResponse(
        content={
            "message": f"Successfully uploaded and embedded {len(uploaded_files)} file(s)",
            "chunks_embedded": total_chunks_processed,
            "files": uploaded_files
        },
        status_code=200
    )
