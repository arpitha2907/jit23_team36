"""Combines OCR confidence + rule flag + NLP flag + ML surprisal into one
error-probability score per token.

Uses a trained classifier (see ml/train_fusion.py) if one has been saved to
backend/models/fusion_model.joblib; otherwise falls back to a transparent
weighted heuristic so the pipeline works out of the box before any model
has been trained on labeled data.
"""
import math
import os

import joblib

MODEL_PATH = os.getenv("FUSION_MODEL_PATH", os.path.join("models", "fusion_model.joblib"))

_model = None
_attempted_load = False

# Weights used by the heuristic fallback. Chosen so OCR confidence (the
# single most reliable per-token signal) dominates, with rules/NLP/ML
# contributing supporting evidence -- see project_plan.md section 6.
_HEURISTIC_WEIGHTS = (0.35, 0.25, 0.15, 0.25)  # norm_conf, rule, nlp, norm_surprisal


def _load_model():
    global _model, _attempted_load
    if not _attempted_load:
        _attempted_load = True
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
    return _model


def _normalize_confidence(ocr_conf: float) -> float:
    clamped = max(0.0, min(ocr_conf, 100.0))
    return 1.0 - clamped / 100.0  # low OCR confidence -> high risk


def _normalize_surprisal(surprisal: float) -> float:
    return 1.0 - math.exp(-max(surprisal, 0.0))


def build_feature_vector(
    ocr_conf: float, rule_flag: bool, nlp_flag: bool, ml_surprisal: float
) -> list:
    return [
        _normalize_confidence(ocr_conf),
        float(bool(rule_flag)),
        float(bool(nlp_flag)),
        _normalize_surprisal(ml_surprisal),
    ]


def fused_error_score(
    ocr_conf: float, rule_flag: bool, nlp_flag: bool, ml_surprisal: float
) -> float:
    features = build_feature_vector(ocr_conf, rule_flag, nlp_flag, ml_surprisal)
    model = _load_model()
    if model is not None:
        proba = model.predict_proba([features])[0][1]
        return float(proba)
    score = sum(w * f for w, f in zip(_HEURISTIC_WEIGHTS, features))
    return float(min(max(score, 0.0), 1.0))
