"""Supported OCR languages and shared normalization helpers.

Language selection is deliberately explicit.  Combining every Indic language
for every scan makes Tesseract slower and can reduce recognition quality, so a
reviewer selects one language or a small mixed-language combination.
"""

from __future__ import annotations

from collections.abc import Iterable

SUPPORTED_LANGUAGES = {
    "eng": "English",
    "asm": "Assamese",
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "ori": "Odia",
    "pan": "Punjabi",
    "san": "Sanskrit",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
}

INDIC_LANGUAGE_CODES = frozenset(SUPPORTED_LANGUAGES) - {"eng"}


def normalize_language_codes(value: str | Iterable[str] | None) -> list[str]:
    """Normalize a comma/plus-separated selection and reject unknown codes."""
    if value is None:
        raw_codes = ["eng"]
    elif isinstance(value, str):
        raw_codes = value.replace(",", "+").split("+")
    else:
        raw_codes = list(value)

    codes = []
    for raw in raw_codes:
        code = str(raw).strip().lower()
        if not code:
            continue
        if code not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported OCR language '{code}'. "
                f"Choose from: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        if code not in codes:
            codes.append(code)

    if not codes:
        return ["eng"]
    if len(codes) > 3:
        raise ValueError("Select at most three OCR languages for one document")
    return codes


def tesseract_language(value: str | Iterable[str] | None) -> str:
    return "+".join(normalize_language_codes(value))


def has_indic_language(value: str | Iterable[str] | None) -> bool:
    return bool(set(normalize_language_codes(value)) & INDIC_LANGUAGE_CODES)


def language_label(value: str | Iterable[str] | None) -> str:
    return " + ".join(SUPPORTED_LANGUAGES[code] for code in normalize_language_codes(value))
