import os
import uuid
import asyncio
from typing import List, Dict

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import aiofiles.os as aios

# Import utilities from embedding_utils.py
from util.vector_store import collection, prepare_document_for_rag
from schema.files import FileListResponse







router= APIRouter()

UPLOAD_DIRECTORY = "files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    print(f"Created upload directory: {os.path.abspath(UPLOAD_DIRECTORY)}")



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
        error_msg = f"Permission denied for directory: {FOLDER_PATH}" # type: ignore
        raise HTTPException(status_code=403, detail=error_msg)
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        error_msg = f"Internal server error in list_files: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
    

async def process_and_embed_file(file_path: str, original_filename: str) -> Dict:
    """
    Processes a single file: prepares it for RAG and embeds it into ChromaDB.
    Returns a dictionary with processing results.
    """
    print(f"Processing and embedding: {original_filename}")
    file_result = {
        "filename": original_filename,
        "path": file_path,
        "status": "processing",
        "chunks_embedded": 0,
        "error": None
    }

    try:
        # Run CPU-bound 'prepare_document_for_rag' in a thread pool
        texts, metadatas = await asyncio.to_thread(prepare_document_for_rag, file_path)

        if texts and metadatas:
            ids = [str(uuid.uuid4()) for _ in texts]
            # Run synchronous 'collection.add' in a thread pool
            await asyncio.to_thread(collection.add, documents=texts, metadatas=metadatas, ids=ids) # type: ignore
            file_result["chunks_embedded"] = len(texts)
            file_result["status"] = "successfully_embedded"
            print(f"Successfully embedded {len(texts)} chunks from {original_filename}")
        elif not texts:
            file_result["status"] = "no_text_to_embed"
            print(f"No text to embed from {original_filename} after RAG preparation.")
        else: # Should not happen if texts is empty, metadatas should also be empty
             file_result["status"] = "empty_preparation_result"
             print(f"Empty or inconsistent result from RAG preparation for {original_filename}.")


    except Exception as e:
        error_message = f"Error during embedding process for {original_filename}: {str(e)}"
        print(error_message)
        # import traceback
        # traceback.print_exc() # For more detailed server-side logging
        file_result["status"] = "error_embedding"
        file_result["error"] = error_message
        # Optionally, clean up the saved file if embedding fails and it's desired
        # if await aios.path.exists(file_path):
        #     try:
        #         await aios.remove(file_path)
        #         print(f"Cleaned up failed file: {file_path}")
        #     except Exception as rm_err:
        #         print(f"Error cleaning up file {file_path}: {rm_err}")
    
    return file_result




@router.post("/upload")
async def upload_files_endpoint(files: List[UploadFile] = File(...)):
    """
    Uploads one or more files, saves them, processes them for RAG,
    and embeds them into ChromaDB asynchronously.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    upload_tasks = []
    saved_file_paths_info = [] # To keep track of saved files for processing

    for file_upload in files:
        if not file_upload.filename:
            # This specific file lacks a filename, we can skip it or raise an error for the whole batch
            # For now, let's add a note and continue, or you could raise HTTPException here
            print(f"Skipping a file due to missing filename.")
            # Or: raise HTTPException(status_code=400, detail="One or more files are missing a filename.")
            continue

        # Basic filename sanitization (consider more robust methods for production)
        safe_filename = os.path.basename(file_upload.filename)
        if not safe_filename: # e.g. if filename was just ".."
            print(f"Skipping file due to invalid sanitized filename: {file_upload.filename}")
            continue
        
        file_path = os.path.join(UPLOAD_DIRECTORY, safe_filename)

        # Asynchronously save file to disk
        try:
            async with aiofiles.open(file_path, "wb") as buffer:
                content = await file_upload.read()  # Read content from UploadFile
                await buffer.write(content)         # Write to disk
            print(f"File '{safe_filename}' saved to '{file_path}'")
            saved_file_paths_info.append({"path": file_path, "original_filename": safe_filename, "content_type": file_upload.content_type})
        except Exception as e:
            # If saving fails, we can't process it. Return an error for this file.
            # This is tricky with asyncio.gather; for now, we'll just log and skip.
            # A more robust solution might involve reporting individual file save errors.
            print(f"Error saving file {safe_filename}: {str(e)}. Skipping this file.")
            # Consider adding to a list of failed saves to report back to the client.

    # Create processing tasks for successfully saved files
    for file_info in saved_file_paths_info:
        upload_tasks.append(process_and_embed_file(file_info["path"], file_info["original_filename"]))

    # Wait for all file processing and embedding tasks to complete
    # `return_exceptions=False` by default, `process_and_embed_file` handles its own exceptions and returns a dict.
    results = await asyncio.gather(*upload_tasks)

    total_chunks_globally = 0
    processed_files_details = []
    successful_uploads_count = 0

    for i, result_dict in enumerate(results):
        # Match result with original file info if needed, though result_dict contains filename
        file_info = next((f for f in saved_file_paths_info if f["original_filename"] == result_dict["filename"]), None)
        
        processed_files_details.append({
            "filename": result_dict["filename"],
            "content_type": file_info["content_type"] if file_info else "N/A",
            "status": result_dict["status"],
            "chunks_embedded": result_dict["chunks_embedded"],
            "detail": result_dict["error"] if result_dict["error"] else "Processed successfully."
        })
        if result_dict["status"] == "successfully_embedded":
            total_chunks_globally += result_dict["chunks_embedded"]
            successful_uploads_count += 1

    return JSONResponse(
        content={
            "message": f"Processed {len(saved_file_paths_info)} files. {successful_uploads_count} successfully embedded.",
            "total_chunks_embedded_in_request": total_chunks_globally,
            "files_details": processed_files_details
        },
        status_code=200 # Could be 207 (Multi-Status) if some succeeded and some failed
    )
