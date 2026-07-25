"""Layer 1 of error detection: format/regex rules for structured government
document fields (ID numbers, dates, PIN codes), plus a cheap heuristic for
detecting classic OCR glyph confusions (O/0, l/1, S/5, ...) inside fields
that look like they should be pure numbers."""
import re

PATTERNS = {
    "aadhaar": re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$"),
    "pan": re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
    "pincode": re.compile(r"^\d{6}$"),
    "date": re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"),
}

# Characters commonly confused by OCR when the true value is a digit.
CONFUSABLES = {"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"}

_MOSTLY_DIGIT_LOOKALIKE = re.compile(r"^[\dOolI|SB/\-]{4,}$")


def matches_field(token: str, field_type: str) -> bool:
    pattern = PATTERNS.get(field_type)
    if not pattern:
        return True
    return bool(pattern.match(token.strip()))


def normalize_confusables(token: str) -> str:
    return "".join(CONFUSABLES.get(ch, ch) for ch in token)


def rule_flag(token: str) -> bool:
    """True if the token looks like it *should* be a numeric/ID/date field
    (only digits and separators like '/', '-') but actually contains a
    letter look-alike (O/o/l/I/|/S/B) mixed in -- a strong signal of an
    OCR digit/letter confusion. A clean date like '15/08/1998' or a plain
    ID number is NOT flagged; only an actual confusable character trips it.
    """
    stripped = token.strip()
    if len(stripped) < 4:
        return False
    if not _MOSTLY_DIGIT_LOOKALIKE.match(stripped):
        return False
    return any(ch in CONFUSABLES for ch in stripped)
