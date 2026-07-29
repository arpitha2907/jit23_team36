"""Test fixtures for API-level tests.

Uses a real Postgres database (a separate `ocr_gov_test` database on the
same server as the app) rather than SQLite, since `Document`/`User` use
`sqlalchemy.dialects.postgresql.UUID` columns that don't round-trip
cleanly through string comparisons on SQLite.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app


def _swap_database_name(url: str, name: str) -> str:
    base, _, _ = url.rpartition("/")
    return f"{base}/{name}"


def _ensure_test_database() -> None:
    admin_engine = create_engine(
        _swap_database_name(settings.database_url, "postgres"),
        isolation_level="AUTOCOMMIT",
    )
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = 'ocr_gov_test'")
        ).first()
        if not exists:
            conn.execute(text("CREATE DATABASE ocr_gov_test"))
    admin_engine.dispose()


@pytest.fixture(scope="session")
def engine():
    _ensure_test_database()
    eng = create_engine(_swap_database_name(settings.database_url, "ocr_gov_test"))
    Base.metadata.drop_all(bind=eng)
    Base.metadata.create_all(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def client(db_session):
    def _get_db_override():
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
