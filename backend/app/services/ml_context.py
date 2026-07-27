"""Layer 3 of error detection: context-driven OCR error scoring.

Uses a pretrained masked language model (BERT/RoBERTa) to measure how
surprising OCR tokens are in context.

The module supports both single-token inference and batched inference.
Batched inference is used by the document pipeline so multiple masked
sentences are processed in one model forward pass.
"""

import math

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from app.core.config import settings


_tokenizer = None
_model = None

DEFAULT_BATCH_SIZE = 16


def _load():
    """Load tokenizer/model once and reuse them across requests."""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.ml_model_name)
        _model = AutoModelForMaskedLM.from_pretrained(settings.ml_model_name)
        _model.eval()

    return _tokenizer, _model


def _mask_and_run(sentence_tokens: list, index: int):
    """Run masked-LM inference for one token.

    Kept for backwards compatibility with existing correction code.
    """
    tokenizer, model = _load()

    if index < 0 or index >= len(sentence_tokens):
        return None, None, None

    masked_tokens = list(sentence_tokens)
    masked_tokens[index] = tokenizer.mask_token

    text = " ".join(masked_tokens)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    mask_positions = (
        inputs["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    if len(mask_positions) == 0:
        return None, None, None

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, mask_pos]

    return tokenizer, logits, mask_pos


def token_surprisal(sentence_tokens: list, index: int) -> float:
    """Return -log P(observed token | context).

    Higher values indicate that the observed OCR token is less likely
    according to the language model.
    """
    if index < 0 or index >= len(sentence_tokens):
        return 0.0

    observed = sentence_tokens[index]

    tokenizer, logits, _ = _mask_and_run(sentence_tokens, index)

    if logits is None:
        return 0.0

    probs = torch.softmax(logits, dim=-1)

    observed_ids = tokenizer.encode(
        observed,
        add_special_tokens=False,
    )

    if not observed_ids:
        return 0.0

    token_id = observed_ids[0]

    if token_id >= probs.shape[-1]:
        return 0.0

    probability = max(probs[token_id].item(), 1e-12)

    return -math.log(probability)


def batch_token_surprisal(
    sentence_tokens: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list:
    """Calculate surprisal for all tokens using batched model inference.

    Instead of running the transformer once for every token independently,
    this function creates masked versions of the sentence and processes
    several of them in each model forward pass.

    Returns one surprisal value for every input token.
    """
    if not sentence_tokens:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    tokenizer, model = _load()

    results = [0.0] * len(sentence_tokens)

    for batch_start in range(0, len(sentence_tokens), batch_size):
        batch_end = min(
            batch_start + batch_size,
            len(sentence_tokens),
        )

        batch_indices = list(range(batch_start, batch_end))

        masked_texts = []

        for index in batch_indices:
            masked_tokens = list(sentence_tokens)
            masked_tokens[index] = tokenizer.mask_token
            masked_texts.append(" ".join(masked_tokens))

        inputs = tokenizer(
            masked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs)

        for row, original_index in enumerate(batch_indices):
            mask_positions = (
                input_ids[row] == tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                results[original_index] = 0.0
                continue

            mask_pos = mask_positions[0].item()

            logits = outputs.logits[row, mask_pos]

            probs = torch.softmax(logits, dim=-1)

            observed = sentence_tokens[original_index]

            observed_ids = tokenizer.encode(
                observed,
                add_special_tokens=False,
            )

            if not observed_ids:
                results[original_index] = 0.0
                continue

            token_id = observed_ids[0]

            if token_id >= probs.shape[-1]:
                results[original_index] = 0.0
                continue

            probability = max(
                probs[token_id].item(),
                1e-12,
            )

            results[original_index] = -math.log(probability)

    return results


def top_k_predictions(
    sentence_tokens: list,
    index: int,
    k: int = 5,
) -> list:
    """Return the language model's top context-aware replacements."""
    tokenizer, logits, _ = _mask_and_run(
        sentence_tokens,
        index,
    )

    if logits is None:
        return []

    top_ids = torch.topk(logits, k).indices.tolist()

    return [
        tokenizer.decode([token_id]).strip()
        for token_id in top_ids
    ]