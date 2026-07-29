"""Document-type and field-aware context for government-document OCR.

The language model is useful for ordinary language, but it is not the right
validator for an Aadhaar number, a date, or a PAN.  This module supplies the
missing document context: it maps OCR values to the field they came from and
selects the validator that makes sense for that field.
"""

from __future__ import annotations

import re
from calendar import monthrange

from app.services import rules

KNOWN_DOCUMENT_TYPES = {"generic", "aadhaar", "pan", "passport", "marksheet"}

_FIELD_ALIASES = {
    "date": "date",
    "date of birth": "date",
    "date of issue": "date",
    "date of expiry": "date",
    "valid until": "date",
    "valid from": "date",
    "aadhaar number": "aadhaar",
    "pan number": "pan",
    "passport number": "passport",
    "pin code": "pincode",
    "pincode": "pincode",
    "postal code": "pincode",
    "gender": "gender",
    "name": "name",
    "full name": "name",
    "given name": "name",
    "surname": "name",
    "father's name": "name",
    "address": "address",
}

_PASSPORT_RE = re.compile(r"^[A-Z0-9]{6,9}$", re.IGNORECASE)
_GENDER_VALUES = {"male", "female", "other", "m", "f", "o"}


def normalize_document_type(doc_type: str | None) -> str:
    value = (doc_type or "generic").strip().lower()
    return value if value in KNOWN_DOCUMENT_TYPES else "generic"


def field_kind(field_label: str | None, doc_type: str | None = None) -> str | None:
    """Return the semantic kind of a field value, if it is known."""
    label = " ".join((field_label or "").lower().split())
    kind = _FIELD_ALIASES.get(label)
    document_type = normalize_document_type(doc_type)

    # Generic labels become document-specific only when the user selected a
    # document type; this avoids treating every "ID Number" as Aadhaar.
    if kind is None and label in {"id number", "id no", "identity number"}:
        kind = {
            "aadhaar": "aadhaar",
            "pan": "pan",
            "passport": "passport",
        }.get(document_type)
    return kind


def _valid_date(token: str) -> bool:
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", token.strip())
    if not match:
        return False
    day, month, year = (int(part) for part in match.groups())
    if year < 100:
        year += 2000
    return 1 <= month <= 12 and 1 <= day <= monthrange(year, month)[1]


def field_rule_flag(
    token: str,
    field_label: str | None = None,
    doc_type: str | None = None,
) -> bool:
    """Apply a context-appropriate structural validator to one OCR token."""
    kind = field_kind(field_label, doc_type)
    if kind == "date":
        return not _valid_date(token)
    if kind == "aadhaar":
        return not rules.matches_field(token, "aadhaar") or rules.rule_flag(token)
    if kind == "pan":
        return not rules.matches_field(token.upper(), "pan") or rules.rule_flag(token)
    if kind == "pincode":
        return not rules.matches_field(token, "pincode") or rules.rule_flag(token)
    if kind == "passport":
        return not bool(_PASSPORT_RE.fullmatch(token.strip())) or rules.rule_flag(token)
    if kind == "gender":
        return token.strip().lower() not in _GENDER_VALUES
    return rules.rule_flag(token)


def should_run_dictionary_check(field_label: str | None) -> bool:
    """Dictionary checks are useful for prose, not structured identifiers."""
    return field_kind(field_label) not in {
        "date",
        "aadhaar",
        "pan",
        "pincode",
        "passport",
        "gender",
    }


def should_run_context_model(
    word: dict,
    doc_type: str | None = None,
) -> bool:
    """Use the expensive language model only where language context helps.

    Labels and structured government values are handled by field validators.
    Names, addresses, and free-text values still receive contextual scoring.
    """
    if word.get("is_label_token"):
        return False
    kind = field_kind(word.get("field_label"), doc_type)
    if kind in {"date", "aadhaar", "pan", "pincode", "passport", "gender"}:
        return False
    if kind is None and not any(char.isalpha() for char in word.get("text", "")):
        return False
    return True
