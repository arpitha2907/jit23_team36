import json
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import schemas
from app.api.deps import get_current_user, require_role
from app.core.config import settings
from app.db import models
from app.db.database import SessionLocal, get_db
from app.services import pipeline, render_preview, storage

router = APIRouter(prefix="/documents", tags=["documents"])


def _run_pipeline_and_store(document_id: uuid.UUID, image_path: str) -> None:
    """Runs in a background task with its own DB session."""
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter(models.Document.id == document_id).first()
        if doc is None:
            return
        try:
            result = pipeline.process_document(image_path)
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
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    doc_id = uuid.uuid4()
    raw_bytes = file.file.read()

    os.makedirs(settings.upload_dir, exist_ok=True)
    encrypted_path = os.path.join(settings.upload_dir, f"{doc_id}.enc")
    storage.save_encrypted_file(raw_bytes, encrypted_path)

    # OpenCV/Tesseract need a real file path; stage a plaintext temp copy
    # that background processing deletes immediately afterward. The
    # permanent artifact on disk (encrypted_path) is always encrypted.
    temp_path = os.path.join(settings.upload_dir, f"{doc_id}_tmp_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(raw_bytes)

    doc = models.Document(
        id=doc_id,
        owner_id=user.id,
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

    background_tasks.add_task(_run_pipeline_and_store, doc_id, temp_path)
    return doc


@router.get("/{document_id}/status")
def get_status(
    document_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": doc.status}


@router.get("/{document_id}/result", response_model=schemas.DocumentResult)
def get_result(
    document_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
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
        "status": doc.status,
        "raw_text": doc.raw_ocr_text,
        "corrected_text": doc.corrected_text,
        "image_width": doc.image_width,
        "image_height": doc.image_height,
        "fields": fields,
    }


@router.get("/{document_id}/preview-image")
def get_preview_image(
    document_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Renders the (corrected, where accepted) text back onto a blank
    canvas at each word's original bounding-box position -- a layout
    reconstruction that visually mirrors the input document's placement
    and alignment, rather than a flat block of corrected text."""
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.image_width or not doc.image_height:
        raise HTTPException(
            status_code=409, detail="Document hasn't finished processing yet"
        )

    words = [
        {
            "text": f.accepted_correction or f.token,
            "left": f.left,
            "top": f.top,
            "width": f.width,
            "height": f.height,
            "is_label_token": f.is_label_token,
            "conf": f.ocr_confidence,
        }
        for f in doc.fields
        if f.left is not None
    ]
    png_bytes = render_preview.render_layout_image(doc.image_width, doc.image_height, words)
    return Response(content=png_bytes, media_type="image/png")


@router.post("/{document_id}/correct")
def submit_corrections(
    document_id: str,
    payload: schemas.CorrectionSubmission,
    db: Session = Depends(get_db),
    user: models.User = Depends(
        require_role(models.UserRole.reviewer, models.UserRole.admin)
    ),
):
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

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
def get_audit(
    document_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return [
        {
            "action": a.action,
            "detail": a.detail,
            "timestamp": a.timestamp.isoformat(),
            "user_id": str(a.user_id) if a.user_id else None,
        }
        for a in doc.audit_logs
    ]
