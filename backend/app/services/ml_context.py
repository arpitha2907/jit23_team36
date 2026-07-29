"""Contextual ML error detection: a BERT masked-language-model scores how
surprising each OCR token is given its surrounding sentence, and offers
context-ranked replacement candidates.

Token positions are whitespace-split OCR words, not BERT wordpieces. Each
call masks exactly the target word (`is_split_into_words=True` collapses
it to a single [MASK] regardless of how many wordpieces it would normally
split into) and reads the model's prediction at that position.

Model weights are loaded lazily, once per process, as a module-level
singleton. English documents use the configured BERT model; Indic and mixed
documents use the configured Indic model. The first document for a model can
download weights, which is why the frontend polls generously on a first run.
"""
from typing import List, Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from app.core.config import settings

DEFAULT_BATCH_SIZE = 8
CONTEXT_WINDOW = 64  # words kept each side of the masked token
MAX_SEQ_LEN = 512

_tokenizer = None
_model = None
_loaded_model_name = None


def _get_model(language_codes=None):
    global _tokenizer, _model, _loaded_model_name
    model_name = settings.model_name_for_languages(language_codes)
    if _model is None or _loaded_model_name != model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForMaskedLM.from_pretrained(model_name)
        _model.eval()
        _loaded_model_name = model_name
    return _tokenizer, _model


def _window(tokens: List[str], index: int):
    """Slices a fixed-size context window around `index` so sequence
    length -- and therefore cost per position -- doesn't scale with
    document length. Returns (windowed_tokens, local_index)."""
    start = max(index - CONTEXT_WINDOW, 0)
    end = min(index + CONTEXT_WINDOW + 1, len(tokens))
    return tokens[start:end], index - start


def _masked_encoding(tokenizer, word_lists: List[List[str]]):
    return tokenizer(
        word_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
        return_tensors="pt",
    )


def _mask_positions(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Column index of the [MASK] token in each row."""
    return (input_ids == mask_token_id).float().argmax(dim=1)


def token_surprisal(tokens: List[str], index: int, language_codes=None) -> float:
    """-log P(original token | context), via one masked BERT forward pass.
    Returns 0.0 for an empty token list or an out-of-range index, rather
    than raising -- callers treat this as "no ML signal available"."""
    if not tokens or index < 0 or index >= len(tokens):
        return 0.0
    try:
        results = batch_token_surprisal(
            tokens,
            batch_size=1,
            _indices=[index],
            language_codes=language_codes,
        )
        return results[0]
    except Exception:
        return 0.0


def batch_token_surprisal(
    tokens: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    _indices: Optional[List[int]] = None,
    language_codes=None,
) -> List[float]:
    """Surprisal for every token in `tokens`, batching multiple masked
    variants into as few forward passes as possible -- `ceil(n/batch_size)`
    calls total, not one sequential call per token."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not tokens:
        return []

    indices = _indices if _indices is not None else list(range(len(tokens)))
    tokenizer, model = _get_model(language_codes)
    mask_id = tokenizer.mask_token_id

    surprisals = [0.0] * len(indices)
    for chunk_start in range(0, len(indices), batch_size):
        chunk = indices[chunk_start : chunk_start + batch_size]
        windows = []
        target_ids = []
        for idx in chunk:
            windowed, local_idx = _window(tokens, idx)
            masked_words = list(windowed)
            original_word = masked_words[local_idx]
            masked_words[local_idx] = tokenizer.mask_token
            windows.append(masked_words)
            ids = tokenizer.encode(original_word, add_special_tokens=False)
            target_ids.append(ids[0] if ids else tokenizer.unk_token_id)

        encoding = _masked_encoding(tokenizer, windows)
        with torch.no_grad():
            logits = model(**encoding).logits  # (batch, seq_len, vocab)

        mask_cols = _mask_positions(encoding["input_ids"], mask_id)
        log_probs = torch.log_softmax(logits, dim=-1)
        for row, target_id in enumerate(target_ids):
            mask_col = int(mask_cols[row].item())
            # Floor avoids -inf surprisal on a near-zero predicted probability.
            log_p = max(log_probs[row, mask_col, target_id].item(), -30.0)
            surprisals[chunk_start + row] = -log_p

    return surprisals


def top_k_predictions(tokens: List[str], index: int, k: int = 5, language_codes=None) -> List[str]:
    """Top-k predicted tokens (raw BERT vocabulary strings, including
    wordpiece continuations -- callers filter those) for the masked
    position at `index`."""
    if not tokens or index < 0 or index >= len(tokens):
        return []

    return batch_top_k_predictions(
        tokens,
        [index],
        k=k,
        language_codes=language_codes,
    )[index]


def batch_top_k_predictions(
    tokens: List[str],
    indices: List[int],
    k: int = 5,
    language_codes=None,
) -> dict[int, List[str]]:
    """Return contextual candidates for several tokens in batched passes.

    Correction suggestions used to invoke one transformer forward pass per
    flagged token.  Keeping the same masked-LM behavior but batching those
    requests makes the expensive correction stage scale with batches rather
    than with the number of flagged fields.
    """
    if not tokens or not indices:
        return {}
    tokenizer, model = _get_model(language_codes)
    predictions: dict[int, List[str]] = {}

    for chunk_start in range(0, len(indices), DEFAULT_BATCH_SIZE):
        chunk = indices[chunk_start : chunk_start + DEFAULT_BATCH_SIZE]
        windows = []
        for idx in chunk:
            if idx < 0 or idx >= len(tokens):
                continue
            windowed, local_idx = _window(tokens, idx)
            masked_words = list(windowed)
            masked_words[local_idx] = tokenizer.mask_token
            windows.append((idx, masked_words))

        if not windows:
            continue
        encoding = _masked_encoding(tokenizer, [words for _, words in windows])
        with torch.no_grad():
            logits = model(**encoding).logits

        mask_cols = _mask_positions(encoding["input_ids"], tokenizer.mask_token_id)
        for row, (idx, _) in enumerate(windows):
            probs = torch.softmax(logits[row, int(mask_cols[row].item())], dim=-1)
            top_ids = torch.topk(probs, k).indices.tolist()
            predictions[idx] = tokenizer.convert_ids_to_tokens(top_ids)
    return predictions
