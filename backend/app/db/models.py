import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


class UserRole(str, enum.Enum):
    uploader = "uploader"
    reviewer = "reviewer"
    admin = "admin"


class DocumentStatus(str, enum.Enum):
    processing = "processing"
    needs_review = "needs_review"
    approved = "approved"
    failed = "failed"


class AuditAction(str, enum.Enum):
    upload = "upload"
    process = "process"
    approve = "approve"
    reject = "reject"
    edit = "edit"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.uploader, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    doc_type = Column(String, nullable=True)  # aadhaar, passport, marksheet, generic
    ocr_languages = Column(String, nullable=False, default="eng")
    original_filename = Column(String, nullable=False)
    encrypted_image_path = Column(String, nullable=False)
    raw_ocr_text = Column(Text, nullable=True)
    corrected_text = Column(Text, nullable=True)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.processing, nullable=False)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    fields = relationship(
        "DocumentField", back_populates="document", cascade="all, delete-orphan"
    )
    audit_logs = relationship(
        "AuditLog", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentField(Base):
    """One OCR'd token/field and the multi-layer error-detection signals for it."""

    __tablename__ = "document_fields"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    token = Column(String, nullable=False)
    start_offset = Column(Integer, nullable=False)  # word index within the document
    end_offset = Column(Integer, nullable=False)
    ocr_confidence = Column(Float, nullable=True)
    rule_flag = Column(Boolean, default=False)
    nlp_flag = Column(Boolean, default=False)
    ml_score = Column(Float, nullable=True)  # raw contextual surprisal
    fused_error_score = Column(Float, nullable=True)  # 0-1 fused probability
    suggestions = Column(Text, nullable=True)  # JSON-encoded list of candidate corrections
    accepted_correction = Column(String, nullable=True)
    reviewed = Column(Boolean, default=False)

    # Original bounding box from Tesseract -- kept so the document can later
    # be re-rendered with corrected text in the same layout as the input.
    left = Column(Integer, nullable=True)
    top = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    # Field-label vs value structure (see services/field_extraction.py).
    field_label = Column(String, nullable=True)  # e.g. "Date of Birth", if this is a value
    is_label_token = Column(Boolean, default=False)  # True if this token IS a label word

    document = relationship("Document", back_populates="fields")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(Enum(AuditAction), nullable=False)
    detail = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="audit_logs")
