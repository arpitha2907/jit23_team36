import json
import os
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import schemas
from app.api.deps import get_current_user, require_document_access
from app.core.config import settings
from app.db import models
from app.db.database import SessionLocal, get_db
from app.services import document_context, languages, ocr_service, pipeline, render_preview, storage

router = APIRouter(prefix="/documents", tags=["documents"])

# Magic-byte signatures for the image formats this app actually processes.
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"BM",  # BMP
    b"II*\x00",  # TIFF (little-endian)
    b"MM\x00*",  # TIFF (big-endian)
)


def _looks_like_image(raw_bytes: bytes) -> bool:
    return any(raw_bytes.startswith(sig) for sig in _IMAGE_SIGNATURES)


def _download_stem(filename: str | None) -> str:
    stem = os.path.splitext(os.path.basename(filename or "document"))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "document"


def _run_pipeline_and_store(
    document_id: uuid.UUID,
    image_path: str,
    doc_type: str,
    ocr_languages: str,
) -> None:
    """Runs in a background task with its own DB session."""
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter(models.Document.id == document_id).first()
        if doc is None:
            return
        try:
            result = pipeline.process_document(
                image_path,
                doc_type=doc_type,
                ocr_languages=ocr_languages,
            )
            doc.raw_ocr_text = result["raw_text"]
            doc.status = models.DocumentStatus.needs_review
            doc.image_width = result["image_width"]
            doc.image_height = result["image_height"]
            for f in result["fields"]:
                db.add(
                    models.DocumentField(
                        document_id=doc.id,
                        token=f["token"],
                        start_offset=f["index"],
                        end_offset=f["index"],
                        ocr_confidence=f["ocr_confidence"],
                        rule_flag=f["rule_flag"],
                        nlp_flag=f["nlp_flag"],
                        ml_score=f["ml_surprisal"],
                        fused_error_score=f["fused_error_score"],
                        suggestions=json.dumps(f["suggestions"]),
                        left=f["left"],
                        top=f["top"],
                        width=f["width"],
                        height=f["height"],
                        field_label=f["field_label"],
                        is_label_token=f["is_label_token"],
                    )
                )
            db.add(
                models.AuditLog(
                    document_id=doc.id,
                    user_id=doc.owner_id,
                    action=models.AuditAction.process,
                    detail="Automated pipeline completed",
                )
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the audit log
            doc.status = models.DocumentStatus.failed
            db.add(
                models.AuditLog(
                    document_id=doc.id,
                    user_id=doc.owner_id,
                    action=models.AuditAction.process,
                    detail=f"Pipeline failed: {exc}",
                )
            )
            db.commit()
    finally:
        db.close()
        if os.path.exists(image_path):
            os.remove(image_path)


@router.post("/upload", response_model=schemas.DocumentOut)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: schemas.DocumentType = Form(schemas.DocumentType.generic),
    ocr_languages: str = Form(settings.ocr_languages),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    doc_id = uuid.uuid4()
    raw_bytes = file.file.read(settings.max_upload_size_bytes + 1)
    if len(raw_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB upload limit",
        )
    if not _looks_like_image(raw_bytes):
        raise HTTPException(status_code=400, detail="File is not a recognized image format")
    try:
        normalized_languages = languages.tesseract_language(ocr_languages)
        ocr_service.validate_languages(normalized_languages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    os.makedirs(settings.upload_dir, exist_ok=True)
    encrypted_path = os.path.join(settings.upload_dir, f"{doc_id}.enc")
    storage.save_encrypted_file(raw_bytes, encrypted_path)

    # OpenCV/Tesseract need a real file path; stage a plaintext temp copy
    # that background processing deletes immediately afterward. The
    # permanent artifact on disk (encrypted_path) is always encrypted.
    # Named purely from the server-generated doc_id -- never the
    # client-supplied filename -- to rule out path traversal.
    temp_path = os.path.join(settings.upload_dir, f"{doc_id}.tmp")
    with open(temp_path, "wb") as f:
        f.write(raw_bytes)

    doc = models.Document(
        id=doc_id,
        owner_id=user.id,
        doc_type=document_context.normalize_document_type(doc_type.value),
        ocr_languages=normalized_languages,
        original_filename=file.filename,
        encrypted_image_path=encrypted_path,
        status=models.DocumentStatus.processing,
    )
    db.add(doc)
    db.add(
        models.AuditLog(
            document_id=doc.id,
            user_id=user.id,
            action=models.AuditAction.upload,
            detail=file.filename,
        )
    )
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(
        _run_pipeline_and_store,
        doc_id,
        temp_path,
        document_context.normalize_document_type(doc_type.value),
        normalized_languages,
    )
    return doc


@router.get("/{document_id}/status")
def get_status(doc: models.Document = Depends(require_document_access())):
    return {"status": doc.status}


@router.get("/{document_id}/result", response_model=schemas.DocumentResult)
def get_result(doc: models.Document = Depends(require_document_access())):
    fields = [
        {
            "id": str(f.id),
            "token": f.token,
            "ocr_confidence": f.ocr_confidence,
            "rule_flag": f.rule_flag,
            "nlp_flag": f.nlp_flag,
            "ml_score": f.ml_score,
            "fused_error_score": f.fused_error_score,
            "suggestions": json.loads(f.suggestions) if f.suggestions else [],
            "accepted_correction": f.accepted_correction,
            "reviewed": f.reviewed,
            "left": f.left,
            "top": f.top,
            "width": f.width,
            "height": f.height,
            "field_label": f.field_label,
            "is_label_token": f.is_label_token,
        }
        for f in doc.fields
    ]
    return {
        "id": str(doc.id),
        "doc_type": doc.doc_type or "generic",
        "ocr_languages": doc.ocr_languages or "eng",
        "status": doc.status,
        "raw_text": doc.raw_ocr_text,
        "corrected_text": doc.corrected_text,
        "image_width": doc.image_width,
        "image_height": doc.image_height,
        "fields": fields,
    }


@router.get("/{document_id}/preview-image")
def get_preview_image(
    doc: models.Document = Depends(require_document_access()),
):
    """Renders the original scan with accepted corrections overlaid at each
    corrected word's original bounding-box position -- everything else is
    left as the real scanned pixels."""
    if not doc.image_width or not doc.image_height:
        raise HTTPException(
            status_code=409, detail="Document hasn't finished processing yet"
        )

    try:
        base_image_bytes = storage.load_decrypted_file(doc.encrypted_image_path)
    except Exception:
        base_image_bytes = None

    words = [
        {
            "text": f.accepted_correction,
            "left": f.left,
            "top": f.top,
            "width": f.width,
            "height": f.height,
            "is_label_token": f.is_label_token,
            "conf": f.ocr_confidence,
        }
        for f in doc.fields
        if f.left is not None and f.accepted_correction and f.accepted_correction != f.token
    ]
    png_bytes = render_preview.render_layout_image(
        doc.image_width, doc.image_height, words, base_image_bytes=base_image_bytes
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/{document_id}/download/image")
def download_preview_image(
    doc: models.Document = Depends(require_document_access()),
):
    """Download the reviewer-edited document as a layout-preserving PNG."""
    if not doc.image_width or not doc.image_height:
        raise HTTPException(
            status_code=409, detail="Document hasn't finished processing yet"
        )

    try:
        base_image_bytes = storage.load_decrypted_file(doc.encrypted_image_path)
    except Exception:
        base_image_bytes = None

    words = [
        {
            "text": f.accepted_correction,
            "left": f.left,
            "top": f.top,
            "width": f.width,
            "height": f.height,
            "is_label_token": f.is_label_token,
            "conf": f.ocr_confidence,
        }
        for f in doc.fields
        if f.left is not None and f.accepted_correction and f.accepted_correction != f.token
    ]
    png_bytes = render_preview.render_layout_image(
        doc.image_width, doc.image_height, words, base_image_bytes=base_image_bytes
    )
    filename = f"{_download_stem(doc.original_filename)}-corrected.png"
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{document_id}/download/text")
def download_corrected_text(
    doc: models.Document = Depends(require_document_access()),
):
    """Download the reviewed text, falling back to raw OCR before review."""
    if doc.status == models.DocumentStatus.processing:
        raise HTTPException(
            status_code=409, detail="Document is still being processed"
        )
    text = doc.corrected_text if doc.corrected_text is not None else doc.raw_ocr_text or ""
    filename = f"{_download_stem(doc.original_filename)}-corrected.txt"
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{document_id}/correct")
def submit_corrections(
    payload: schemas.CorrectionSubmission,
    db: Session = Depends(get_db),
    doc: models.Document = Depends(
        require_document_access(models.UserRole.reviewer, models.UserRole.admin)
    ),
    user: models.User = Depends(get_current_user),
):
    ordered_fields = sorted(doc.fields, key=lambda x: x.start_offset)
    tokens = [f.token for f in ordered_fields]
    fields_by_id = {str(f.id): f for f in doc.fields}

    for decision in payload.decisions:
        field = fields_by_id.get(decision.field_id)
        if field is None:
            continue
        if decision.accept and decision.corrected_value:
            field.accepted_correction = decision.corrected_value
            tokens[field.start_offset] = decision.corrected_value
        field.reviewed = True
        db.add(
            models.AuditLog(
                document_id=doc.id,
                user_id=user.id,
                action=models.AuditAction.approve
                if decision.accept
                else models.AuditAction.reject,
                detail=f"field={field.token} -> {decision.corrected_value}",
            )
        )

    doc.corrected_text = " ".join(tokens)
    if doc.fields and all(f.reviewed for f in doc.fields):
        doc.status = models.DocumentStatus.approved
    db.commit()
    return {"status": "ok", "corrected_text": doc.corrected_text}


@router.get("/{document_id}/audit")
def get_audit(doc: models.Document = Depends(require_document_access())):
    return [
        {
            "action": a.action,
            "detail": a.detail,
            "timestamp": a.timestamp.isoformat(),
            "user_id": str(a.user_id) if a.user_id else None,
        }
        for a in doc.audit_logs
    ]
