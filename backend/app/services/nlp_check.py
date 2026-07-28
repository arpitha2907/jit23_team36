"""Layer 2 of error detection: dictionary / edit-distance checks. Catches
non-word errors (tokens that aren't valid words at all) and proposes
spelling-correction candidates."""
from spellchecker import SpellChecker

_spell = SpellChecker()


def _clean(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha())


def is_known_word(token: str) -> bool:
    clean = _clean(token)
    if not clean:
        return True  # numbers/symbols aren't handled by this layer
    return clean in _spell


def nlp_flag(token: str) -> bool:
    """True if the token looks like a word but isn't in the dictionary."""
    if token.isalpha() and len(token) > 2:
        return not is_known_word(token)
    return False


def spelling_candidates(token: str, k: int = 5) -> list:
    clean = _clean(token)
    if not clean:
        return []
    candidates = _spell.candidates(clean)
    if not candidates:
        return []
    return list(candidates)[:k]
