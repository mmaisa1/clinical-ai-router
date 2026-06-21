# app/core/database.py
"""
SQLAlchemy engine and session management.

One engine is created at import time, shared across the app's lifetime.
Each request gets its own short-lived session via get_db_session(),
following the standard FastAPI dependency pattern.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db_session():
    """
    FastAPI dependency — yields a session, ensures it's closed after
    the request completes, even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()