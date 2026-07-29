from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def initialize_schema() -> None:
    """Create tables and apply tiny additive dev migrations.

    The project does not yet ship Alembic migrations.  This keeps existing
    local Compose databases usable when the language-selection column is
    added, while remaining harmless on fresh databases.
    """
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE documents "
                    "ADD COLUMN IF NOT EXISTS ocr_languages VARCHAR NOT NULL DEFAULT 'eng'"
                )
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
