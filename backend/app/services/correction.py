"""Error correction: merges deterministic digit/letter-confusable fixes,
dictionary spelling candidates, and context-ranked language-model
predictions into one suggestion list. Always returned as suggestions for
human approval -- never auto-applied -- per the project plan's caution
against silently "fixing" government records.
"""
from app.services import ml_context, nlp_check, rules


def _looks_numeric_field(token: str) -> bool:
    """True if this token is (or should be) a pure-digit field -- an ID
    number, date, or code. For these, a generic word-level language model
    has no real signal (it will happily suggest English words for a masked
    digit string); the deterministic confusable-character fix in rules.py
    is far more reliable."""
    return rules.rule_flag(token)


def _clean_contextual_candidates(candidates: list) -> list:
    """BERT's wordpiece vocabulary includes subword continuations (e.g.
    '##ing') and punctuation, which are meaningless as standalone
    suggestions -- filter those out."""
    cleaned = []
    for c in candidates:
        if not c or c.startswith("##"):
            continue
        if not any(ch.isalnum() for ch in c):
            continue
        cleaned.append(c)
    return cleaned


def suggest_corrections(sentence_tokens: list, index: int, k: int = 5) -> list:
    token = sentence_tokens[index]

    # Numeric/ID-style fields: undo classic OCR digit/letter confusions
    # deterministically instead of asking a word-level language model,
    # which isn't the right tool for arbitrary ID/date numbers.
    if _looks_numeric_field(token):
        normalized = rules.normalize_confusables(token)
        return [normalized] if normalized != token else []

    spell_candidates = nlp_check.spelling_candidates(token, k=k)

    contextual_candidates = []
    try:
        raw = ml_context.top_k_predictions(sentence_tokens, index, k=k)
        contextual_candidates = _clean_contextual_candidates(raw)
    except Exception:
        # ML layer is best-effort; detection still works from rules+NLP if
        # the model isn't available (e.g. no internet to download weights).
        pass

    ranked = []
    seen = set()
    # Dictionary candidates first: they're constrained to real words close
    # in edit distance to the OCR'd token, which is more reliable than a
    # general-purpose masked LM that isn't fine-tuned for OCR correction and
    # can suggest fluent-sounding but unrelated words (see project_plan.md's
    # note on "No Free Lunches" -- generic LLM correction can hallucinate).
    # Contextual candidates are kept as secondary options.
    for candidate in spell_candidates + contextual_candidates:
        key = candidate.lower()
        if not candidate or key == token.lower() or key in seen:
            continue
        seen.add(key)
        ranked.append(candidate)

    return ranked[:k]
