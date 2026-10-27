"""Optional GPT-backed correction suggestions for flagged OCR tokens.

The local dictionary + masked-LM candidates are weak on names, merged
words and noisy scans.  When enabled, one batched chat-completion call per
document asks an OpenAI model for corrections to the flagged free-text
tokens only, given the line each token sits on.

Privacy: this is the only part of the pipeline that sends document text off
the machine, so it is off by default (LLM_SUGGESTIONS_ENABLED) and every
digit is masked before sending -- ID numbers, dates and PIN codes never
leave the server, and tokens containing digits are left to the
deterministic confusable fixes in correction.py.  Suggestions are never
auto-applied; a reviewer still approves each one.
"""
import json
import logging
import re

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_REQUEST = 60
REQUEST_TIMEOUT_SECONDS = 60

_DIGIT_RE = re.compile(r"\d")

_SYSTEM_PROMPT = """You correct OCR misreads in scanned Indian government documents.
You receive JSON: the document type, the OCR language codes, and a list of
flagged tokens, each with an id, the OCR'd token, the full OCR'd line it
appears on, and the field label it belongs to (if known). Digits are masked
as '#'.

For each token, propose up to 3 likely intended values, best first, based on
typical OCR confusions (rn/m, cl/d, li/h, missing or merged spaces, stray
punctuation) and the line's context. Rules:
- Proper names, places and correct words: if the token already looks right,
  return an empty list. Do not "fix" valid names.
- Scanner noise (fragments like "ee", "ae", "oo", or short letter clusters
  with no plausible word): return an empty list. Never invent content.
- A suggestion replaces only that token. Never add words from the rest of
  the line or anything the token's letters don't support.
- Each suggestion is the corrected text alone: no notes, arrows or
  explanations.
- Write each suggestion in the same script as the token (Kannada stays
  Kannada, Tamil stays Tamil). Never transliterate or translate.
- Never output '#' or digits.
Respond with a JSON object only: {"suggestions": {"<id>": ["...", ...]}}."""


def is_enabled() -> bool:
    return settings.llm_suggestions_enabled and bool(settings.openai_api_key)


def _redact(text: str) -> str:
    return _DIGIT_RE.sub("#", text)


def _line_texts(words: list) -> dict:
    """Map each word index to the (redacted) text of its OCR line."""
    lines = {}
    for i, w in enumerate(words):
        key = (w.get("block_num"), w.get("par_num"), w.get("line_num"))
        lines.setdefault(key, []).append(i)
    by_index = {}
    for indices in lines.values():
        text = _redact(" ".join(words[i]["text"] for i in indices))
        for i in indices:
            by_index[i] = text
    return by_index


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )


def _parse(content: str, requested: dict, k: int) -> dict:
    data = json.loads(content or "{}")
    raw = data.get("suggestions", {}) if isinstance(data, dict) else {}
    # Every requested token gets an entry: an empty list means GPT judged it
    # correct or unrecoverable noise, which correction.py treats differently
    # from "GPT wasn't asked".
    results = {index: [] for index in requested}
    for key, candidates in (raw.items() if isinstance(raw, dict) else []):
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if index not in requested or not isinstance(candidates, list):
            continue
        token = requested[index].lower()
        cleaned = []
        for c in candidates:
            if not isinstance(c, str):
                continue
            c = c.strip()
            if not c or c.lower() == token or "#" in c or c in cleaned:
                continue
            cleaned.append(c)
        results[index] = cleaned[:k]
    return results


def suggest_batch(
    words: list,
    indices: list,
    doc_type: str = "generic",
    language_codes=None,
    k: int = 3,
) -> dict:
    """Return {word_index: [suggestion, ...]} for every flagged token sent
    to GPT ([] where it saw nothing to fix).

    Best-effort: returns {} when disabled, when nothing qualifies, or on any
    API/parse failure, so the pipeline falls back to local suggestions.
    """
    if not is_enabled():
        return {}
    requested = {
        i: words[i]["text"]
        for i in indices
        if not _DIGIT_RE.search(words[i]["text"])
    }
    requested = dict(list(requested.items())[:MAX_TOKENS_PER_REQUEST])
    if not requested:
        return {}

    lines = _line_texts(words)
    payload = {
        "document_type": doc_type,
        "languages": list(language_codes or []),
        "tokens": [
            {
                "id": i,
                "token": token,
                "line": lines.get(i, token),
                "field": words[i].get("field_label"),
            }
            for i, token in requested.items()
        ],
    }

    options = {}
    if settings.openai_reasoning_effort:
        options["reasoning_effort"] = settings.openai_reasoning_effort

    try:
        response = _client().chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **options,
        )
        return _parse(response.choices[0].message.content, requested, k)
    except Exception:
        logger.warning("GPT suggestions unavailable; using local candidates", exc_info=True)
        return {}
