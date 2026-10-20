"""Database engine, session management, and FastAPI dependency."""

import logging
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantpulse.core.config import DEFAULT_DATABASE_URL
from quantpulse.db.models import Base

logger = logging.getLogger(__name__)


def create_db_engine(database_url: str):
    """Instantiate a SQLAlchemy engine with connection pool parameters."""
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
engine = create_db_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(bind_engine=None) -> None:
    """Initialize database schema tables."""
    target_engine = bind_engine or engine
    logger.info("Initializing database tables on engine: %s", target_engine.url)
    Base.metadata.create_all(bind=target_engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for managing database session lifecycle."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
