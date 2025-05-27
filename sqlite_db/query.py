





from uuid import UUID
from fastapi import HTTPException
from sqlite_db.models import File
from sqlite_db.session import AsyncSessionLocal


async def save_file_metadata_to_db(
    file_id: str,
    filename: str,
    path: str,
    content_type: str,
    chunk_count: int
):
    async with AsyncSessionLocal() as session:
        try:
            new_file = File(
                id=UUID(file_id),
                filename=filename,
                path=path,
                content_type=content_type,
                chunk_count=chunk_count,
            )
            session.add(new_file)
            await session.commit()
        except ValueError as e:
            await session.rollback()
            raise HTTPException(status_code=400, detail=f"Invalid UUID format: {str(e)}")
            
        except Exception as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")