import enum
import uuid
from typing import List, Optional

from pydantic import BaseModel

from app.db.models import DocumentStatus, UserRole


class UserCreate(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: UserRole

    class Config:
        from_attributes = True


class UserRoleUpdate(BaseModel):
    role: UserRole


class Token(BaseModel):
    access_token: str
    token_type: str


class DocumentType(str, enum.Enum):
    generic = "generic"
    aadhaar = "aadhaar"
    pan = "pan"
    passport = "passport"
    marksheet = "marksheet"


class DocumentOut(BaseModel):
    id: uuid.UUID
    original_filename: str
    doc_type: DocumentType
    ocr_languages: str
    status: DocumentStatus

    class Config:
        from_attributes = True


class FieldResult(BaseModel):
    id: str
    token: str
    ocr_confidence: Optional[float] = None
    rule_flag: bool
    nlp_flag: bool
    ml_score: Optional[float] = None
    fused_error_score: Optional[float] = None
    suggestions: List[str] = []
    accepted_correction: Optional[str] = None
    reviewed: bool
    left: Optional[int] = None
    top: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    field_label: Optional[str] = None
    is_label_token: bool = False


class DocumentResult(BaseModel):
    id: str
    doc_type: DocumentType
    ocr_languages: str
    status: DocumentStatus
    raw_text: Optional[str] = None
    corrected_text: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    fields: List[FieldResult]


class CorrectionDecision(BaseModel):
    field_id: str
    accept: bool
    corrected_value: Optional[str] = None


class CorrectionSubmission(BaseModel):
    decisions: List[CorrectionDecision]
