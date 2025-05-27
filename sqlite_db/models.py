from uuid import UUID
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone
from typing import Optional

from sympy import content
from .session import Base



class User(Base):
    __tablename__ = "users"
    
    # Using Mapped[] with mapped_column() for proper type inference
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)



class File(Base):
    __tablename__ = "files"
    id : Mapped[UUID] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), index=True)
    path: Mapped[str] = mapped_column(String(255), index=True)
    content_type: Mapped[str] = mapped_column(String(50), index=True)
    chunk_count: Mapped[int] = mapped_column(default=0)
    upload_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc))
