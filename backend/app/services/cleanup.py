"""Automatic cleanup of expired uploaded documents.

Documents older than the configured retention period are removed from
persistent encrypted storage and from the database. Related DocumentField
and AuditLog rows are removed through the Document relationships'
delete-orphan cascades.
"""

import logging
import os
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import models

logger = logging.getLogger(__name__)


def get_retention_cutoff(
    retention_days: int | None = None,
    now: datetime | None = None,
) -> datetime:
    """Return the oldest allowed document creation timestamp."""

    days = (
        settings.document_retention_days
        if retention_days is None
        else retention_days
    )

    if days < 0:
        raise ValueError("retention_days cannot be negative")

    current_time = now or datetime.utcnow()
    return current_time - timedelta(days=days)


def delete_document_file(path: str | None) -> bool:
    """Delete an encrypted document file.

    Returns True when a file was actually deleted.
    Missing files are treated as already cleaned up.
    """

    if not path:
        return False

    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Failed to delete document file: %s", path)
        raise


def cleanup_expired_documents(
    db: Session,
    retention_days: int | None = None,
    now: datetime | None = None,
) -> dict:
    """Delete documents older than the retention cutoff.

    Database rows are deleted only after their encrypted file has either
    been deleted successfully or confirmed to be already absent.
    """

    cutoff = get_retention_cutoff(
        retention_days=retention_days,
        now=now,
    )

    expired_documents = (
        db.query(models.Document)
        .filter(models.Document.created_at < cutoff)
        .all()
    )

    deleted_documents = 0
    deleted_files = 0
    errors = []

    for document in expired_documents:
        try:
            if delete_document_file(document.encrypted_image_path):
                deleted_files += 1

            db.delete(document)
            db.commit()
            deleted_documents += 1

        except Exception as exc:
            db.rollback()

            document_id = str(getattr(document, "id", "unknown"))
            errors.append(
                {
                    "document_id": document_id,
                    "error": str(exc),
                }
            )

            logger.exception(
                "Cleanup failed for document %s",
                document_id,
            )

    return {
        "cutoff": cutoff,
        "deleted_documents": deleted_documents,
        "deleted_files": deleted_files,
        "errors": errors,
    }