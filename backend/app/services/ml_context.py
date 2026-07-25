"""Layer 3 of error detection: the context-driven core. Uses a pre-trained
masked language model (BERT/RoBERTa) to score how surprising each OCR'd
token is given its surrounding sentence -- this is what catches *real-word*
errors (a valid word that's simply wrong in context) that rules and
dictionary checks miss.
"""
import math

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from app.core.config import settings

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.ml_model_name)
        _model = AutoModelForMaskedLM.from_pretrained(settings.ml_model_name)
        _model.eval()
    return _tokenizer, _model


def _mask_and_run(sentence_tokens: list, index: int):
    tokenizer, model = _load()
    masked_tokens = list(sentence_tokens)
    masked_tokens[index] = tokenizer.mask_token
    text = " ".join(masked_tokens)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[0]
    if len(mask_positions) == 0:
        return None, None, None
    mask_pos = mask_positions[0].item()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, mask_pos]
    return tokenizer, logits, mask_pos


def token_surprisal(sentence_tokens: list, index: int) -> float:
    """-log P(observed token | context). Higher = more surprising = more
    likely to be an OCR error."""
    observed = sentence_tokens[index]
    tokenizer, logits, _ = _mask_and_run(sentence_tokens, index)
    if logits is None:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    observed_ids = tokenizer.encode(observed, add_special_tokens=False)
    if not observed_ids:
        return 0.0
    p = max(probs[observed_ids[0]].item(), 1e-12)
    return -math.log(p)


def top_k_predictions(sentence_tokens: list, index: int, k: int = 5) -> list:
    """What the language model thinks *should* be at this position -- used
    as context-aware correction candidates."""
    tokenizer, logits, _ = _mask_and_run(sentence_tokens, index)
    if logits is None:
        return []
    top_ids = torch.topk(logits, k).indices.tolist()
    return [tokenizer.decode([tid]).strip() for tid in top_ids]
