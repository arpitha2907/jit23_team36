"""Ablation-study utilities for the OCR error-detection system.

Measures the contribution of:
- OCR confidence
- Rule-based detection
- NLP detection
- ML contextual surprisal
- combinations of these signals

The full-system configuration uses the same heuristic weights as fusion.py.
"""

import math

from app.services.evaluation import detection_metrics


DEFAULT_THRESHOLD = 0.4

WEIGHTS = {
    "ocr": 0.35,
    "rules": 0.25,
    "nlp": 0.15,
    "ml": 0.25,
}


def _normalize_confidence(confidence: float) -> float:
    confidence = max(0.0, min(float(confidence), 100.0))
    return 1.0 - confidence / 100.0


def _normalize_surprisal(surprisal: float) -> float:
    return 1.0 - math.exp(-max(float(surprisal), 0.0))


def ablation_score(
    ocr_conf: float,
    rule_flag: bool,
    nlp_flag: bool,
    ml_surprisal: float,
    use_ocr: bool = True,
    use_rules: bool = True,
    use_nlp: bool = True,
    use_ml: bool = True,
) -> float:
    """Calculate error score using only selected detection layers.

    Active weights are renormalized so every ablation configuration
    produces a score in a comparable 0-1 range.
    """

    features = {
        "ocr": _normalize_confidence(ocr_conf),
        "rules": float(bool(rule_flag)),
        "nlp": float(bool(nlp_flag)),
        "ml": _normalize_surprisal(ml_surprisal),
    }

    enabled = {
        "ocr": use_ocr,
        "rules": use_rules,
        "nlp": use_nlp,
        "ml": use_ml,
    }

    active_weight = sum(
        WEIGHTS[name]
        for name in WEIGHTS
        if enabled[name]
    )

    if active_weight == 0:
        return 0.0

    weighted_score = sum(
        WEIGHTS[name] * features[name]
        for name in WEIGHTS
        if enabled[name]
    )

    score = weighted_score / active_weight

    return float(min(max(score, 0.0), 1.0))


CONFIGURATIONS = {
    "ocr_only": {
        "use_ocr": True,
        "use_rules": False,
        "use_nlp": False,
        "use_ml": False,
    },
    "rules_only": {
        "use_ocr": False,
        "use_rules": True,
        "use_nlp": False,
        "use_ml": False,
    },
    "nlp_only": {
        "use_ocr": False,
        "use_rules": False,
        "use_nlp": True,
        "use_ml": False,
    },
    "ml_only": {
        "use_ocr": False,
        "use_rules": False,
        "use_nlp": False,
        "use_ml": True,
    },
    "rules_nlp": {
        "use_ocr": False,
        "use_rules": True,
        "use_nlp": True,
        "use_ml": False,
    },
    "without_ml": {
        "use_ocr": True,
        "use_rules": True,
        "use_nlp": True,
        "use_ml": False,
    },
    "full_system": {
        "use_ocr": True,
        "use_rules": True,
        "use_nlp": True,
        "use_ml": True,
    },
}


def run_ablation(
    samples: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Run all ablation configurations over labeled token samples.

    Each sample must contain:

        ocr_conf
        rule_flag
        nlp_flag
        ml_surprisal
        is_error

    is_error is the ground-truth label.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    results = {}

    ground_truth = [
        bool(sample["is_error"])
        for sample in samples
    ]

    for name, config in CONFIGURATIONS.items():

        predictions = []

        for sample in samples:
            score = ablation_score(
                ocr_conf=sample["ocr_conf"],
                rule_flag=sample["rule_flag"],
                nlp_flag=sample["nlp_flag"],
                ml_surprisal=sample["ml_surprisal"],
                **config,
            )

            predictions.append(score >= threshold)

        metrics = detection_metrics(
            ground_truth,
            predictions,
        )

        results[name] = {
            **metrics,
            "threshold": threshold,
        }

    return results