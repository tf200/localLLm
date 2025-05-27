import os
import uuid
import asyncio
from typing import List, Dict


from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import aiofiles.os as aios
from sqlalchemy import select

# Import utilities from embedding_utils.py
from logs.logger import LoggerDep
from sqlite_db.query import save_file_metadata_to_db
from sqlite_db.session import get_db
from util.vector_store import collection, prepare_document_for_rag
from schema.files import DeleteFileResponse, FileListResponse

from sqlite_db.models import File

from sqlalchemy.ext.asyncio import AsyncSession


router= APIRouter()

UPLOAD_DIRECTORY = "files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    print(f"Created upload directory: {os.path.abspath(UPLOAD_DIRECTORY)}")



@router.get("/files", response_model=List[FileListResponse])
async def list_files(logger: LoggerDep, db: AsyncSession = Depends(get_db)):
    """
    List all files stored in the vector DB (via tracked metadata in SQL).
    
    Returns:
    - JSON with a list of file names and file IDs.
    """
    try:
        result = await db.execute(select(File))
        files = result.scalars().all()
        
        return [
            FileListResponse(
                filename = file.filename,
                file_id = str(file.id),
                path = file.path
            ) for file in files
        ]
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while listing files.")


@router.delete("/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(file_id: str, logger: LoggerDep, db: AsyncSession = Depends(get_db)):
    """
    Deletes a file by its ID from the vector DB and the filesystem.
    
    Args:
    - file_id: The ID of the file to delete.
    
    Returns:
    - JSON response indicating success or failure.
    """
    try:
        file_uuid = uuid.UUID(file_id)
        result = await db.execute(select(File).where(File.id == file_uuid))
        file = result.scalar_one_or_none()

        if not file:
            raise HTTPException(status_code=404, detail="File not found.")
        
        logger.info(f"Deleting file: {file.filename} (ID: {file.id})")

        try:
            collection.delete(where={"file_id": str(file.id)})  # type: ignore
            logger.info(f"Deleted file from vector store: {file.filename}")
        except Exception as e:
            logger.error(f"Failed to delete from vector store: {e}")
            raise HTTPException(status_code=500, detail="Vector store deletion failed.")

        try:
            if await aios.path.exists(file.path):
                await aios.remove(file.path)
                logger.info(f"Deleted file from filesystem: {file.path}")
        except Exception as e:
            logger.error(f"Failed to delete file from filesystem: {e}")
            raise HTTPException(status_code=500, detail="Filesystem deletion failed.")

        try:
            await db.delete(file)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to delete metadata from DB: {e}")
            await db.rollback()
            raise HTTPException(status_code=500, detail="Database commit failed.")
        
        return DeleteFileResponse(
            message=f"File '{file.filename}' deleted successfully.",
            file_id=file_id
        )

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error deleting file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while deleting file.")



























































# ==================================================

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

    file_id = str(uuid.uuid4())  # Generate a unique ID for the file

    try:
        # Run CPU-bound 'prepare_document_for_rag' in a thread pool
        texts, metadatas = await asyncio.to_thread(prepare_document_for_rag, file_path, file_id)
        if texts and metadatas:
            ids = [str(uuid.uuid4()) for _ in texts]
            # Run synchronous 'collection.add' in a thread pool
            await asyncio.to_thread(collection.add, documents=texts, metadatas=metadatas, ids=ids) # type: ignore
            file_result["chunks_embedded"] = len(texts)
            file_result["status"] = "successfully_embedded"
            print(f"Successfully embedded {len(texts)} chunks from {original_filename}")
            await save_file_metadata_to_db(
                file_id=file_id,
                filename=original_filename,
                path=file_path,
                content_type="application/octet-stream",  # Assuming binary files, adjust as needed
                chunk_count=len(texts)
            )
        elif not texts:
            file_result["status"] = "no_text_to_embed"
            print(f"No text to embed from {original_filename} after RAG preparation.")
        else: # Should not happen if texts is empty, metadatas should also be empty
             file_result["status"] = "empty_preparation_result"
             print(f"Empty or inconsistent result from RAG preparation for {original_filename}.")


    except Exception as e:
        error_message = f"Error during embedding process for {original_filename}: {str(e)}"
        print(error_message)

        file_result["status"] = "error_embedding"
        file_result["error"] = error_message
        # Optionally, clean up the saved file if embedding fails and it's desired
        if await aios.path.exists(file_path):
            try:
                await aios.remove(file_path)
                print(f"Cleaned up failed file: {file_path}")
            except Exception as rm_err:
                print(f"Error cleaning up file {file_path}: {rm_err}")
    
    return file_result



@router.post("/upload")
async def upload_files_endpoint(files: List[UploadFile] ):
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
            print(f"Skipping a file due to missing filename.")
            raise HTTPException(status_code=400, detail="One or more files are missing a filename.")

        safe_filename = os.path.basename(file_upload.filename)
        if not safe_filename: # e.g. if filename was just ".."
            print(f"Skipping file due to invalid sanitized filename: {file_upload.filename}")
            raise HTTPException(status_code=400, detail="One or more files have an invalid filename.")
        
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
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file {safe_filename}: {str(e)}"
            )

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
