"""Layer 2 of error detection: dictionary / edit-distance checks. Catches
non-word errors (tokens that aren't valid words at all) and proposes
spelling-correction candidates."""
from spellchecker import SpellChecker

from app.services.languages import normalize_language_codes

_spell = SpellChecker()


def _clean(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha())


def is_known_word(token: str) -> bool:
    clean = _clean(token)
    if not clean:
        return True  # numbers/symbols aren't handled by this layer
    return clean in _spell


def _contains_non_latin(token: str) -> bool:
    return any(ord(char) > 0x024F for char in token)


def nlp_flag(token: str, languages: str | list[str] | None = None) -> bool:
    """True if the token looks like a word but isn't in the dictionary."""
    codes = normalize_language_codes(languages)
    if _contains_non_latin(token) or "eng" not in codes:
        # pyspellchecker is English-only; never call it an error detector for
        # native-script tokens or a document explicitly marked non-English.
        return False
    if token.isupper() and len(token) <= 8:
        # Government abbreviations and initials are common valid values.
        return False
    if token.isalpha() and len(token) > 2:
        return not is_known_word(token)
    return False


def spelling_candidates(
    token: str,
    k: int = 5,
    languages: str | list[str] | None = None,
) -> list:
    codes = normalize_language_codes(languages)
    if _contains_non_latin(token) or "eng" not in codes:
        return []
    if token.isupper() and len(token) <= 8:
        return []
    clean = _clean(token)
    if not clean:
        return []
    candidates = _spell.candidates(clean)
    if not candidates:
        return []
    return list(candidates)[:k]
