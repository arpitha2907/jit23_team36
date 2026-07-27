from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.services import cleanup


def test_retention_cutoff():
    now = datetime(2026, 7, 27, 12, 0, 0)

    cutoff = cleanup.get_retention_cutoff(
        retention_days=30,
        now=now,
    )

    assert cutoff == datetime(2026, 6, 27, 12, 0, 0)


def test_negative_retention_rejected():
    with pytest.raises(ValueError):
        cleanup.get_retention_cutoff(retention_days=-1)


def test_delete_existing_file(tmp_path):
    file_path = tmp_path / "document.enc"
    file_path.write_bytes(b"encrypted-data")

    deleted = cleanup.delete_document_file(str(file_path))

    assert deleted is True
    assert not file_path.exists()


def test_delete_missing_file(tmp_path):
    file_path = tmp_path / "missing.enc"

    deleted = cleanup.delete_document_file(str(file_path))

    assert deleted is False


def test_cleanup_expired_document():
    now = datetime(2026, 7, 27, 12, 0, 0)

    document = MagicMock()
    document.id = "test-document"
    document.created_at = now - timedelta(days=40)
    document.encrypted_image_path = "old-document.enc"

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [document]

    with patch(
        "app.services.cleanup.delete_document_file",
        return_value=True,
    ):
        result = cleanup.cleanup_expired_documents(
            db,
            retention_days=30,
            now=now,
        )

    db.delete.assert_called_once_with(document)
    db.commit.assert_called_once()

    assert result["deleted_documents"] == 1
    assert result["deleted_files"] == 1
    assert result["errors"] == []


def test_cleanup_missing_file_still_deletes_database_record():
    document = MagicMock()
    document.id = "missing-file-document"
    document.encrypted_image_path = "missing.enc"

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [document]

    with patch(
        "app.services.cleanup.delete_document_file",
        return_value=False,
    ):
        result = cleanup.cleanup_expired_documents(
            db,
            retention_days=30,
            now=datetime(2026, 7, 27),
        )

    db.delete.assert_called_once_with(document)

    assert result["deleted_documents"] == 1
    assert result["deleted_files"] == 0
    assert result["errors"] == []