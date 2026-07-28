<<<<<<< HEAD
"""Orchestrates the full per-document flow described in project_plan.md:
preprocess -> OCR -> line/field-label grouping -> (rules + NLP + ML) error
detection -> fusion -> correction suggestions.
"""
=======
"""Orchestrates the full per-document OCR pipeline:

preprocess -> OCR -> field extraction -> rules/NLP/ML detection
-> fusion -> correction suggestions.
"""

>>>>>>> accb266 (Implement automated tests and batched ML inference)
from app.services import (
    correction,
    field_extraction,
    fusion,
    ml_context,
    nlp_check,
    ocr_service,
    preprocessing,
    rules,
)

<<<<<<< HEAD
=======

>>>>>>> accb266 (Implement automated tests and batched ML inference)
SUGGESTION_THRESHOLD = 0.4


def process_document(image_path: str) -> dict:
    processed_image = preprocessing.preprocess_image(image_path)
<<<<<<< HEAD
    image_height, image_width = processed_image.shape[:2]

    ocr_result = ocr_service.run_ocr(processed_image)
    words = field_extraction.annotate_field_labels(ocr_result["words"])
    tokens = [w["text"] for w in words]

    field_results = []
    for i, w in enumerate(words):
        token = w["text"]
        r_flag = rules.rule_flag(token)
        n_flag = nlp_check.nlp_flag(token)
        try:
            surprisal = ml_context.token_surprisal(tokens, i)
        except Exception:
            # ML layer is best-effort: if model weights aren't available
            # (e.g. offline), fall back to 0 surprisal and let rules/NLP
            # still drive detection.
            surprisal = 0.0

        score = fusion.fused_error_score(w["conf"], r_flag, n_flag, surprisal)

        suggestions = []
        if score >= SUGGESTION_THRESHOLD and not w["is_label_token"]:
            # Only suggest corrections for values -- labels are boilerplate
            # ("Date of Birth") and don't need spellcheck/context repair.
            suggestions = correction.suggest_corrections(tokens, i)
=======

    image_height, image_width = processed_image.shape[:2]

    ocr_result = ocr_service.run_ocr(processed_image)

    words = field_extraction.annotate_field_labels(
        ocr_result["words"]
    )

    tokens = [w["text"] for w in words]

    # ---------------------------------------------------------
    # BATCHED ML INFERENCE
    # ---------------------------------------------------------
    # Previously token_surprisal() was called inside the loop,
    # causing one transformer forward pass per OCR token.
    #
    # Now all token surprisals are calculated in batches.
    # ---------------------------------------------------------

    try:
        surprisals = ml_context.batch_token_surprisal(tokens)
    except Exception:
        # ML is best-effort. The rest of the OCR/error-detection
        # pipeline must continue working if model weights are
        # unavailable or inference fails.
        surprisals = [0.0] * len(tokens)

    field_results = []

    for i, w in enumerate(words):
        token = w["text"]

        r_flag = rules.rule_flag(token)

        n_flag = nlp_check.nlp_flag(token)

        surprisal = surprisals[i]

        score = fusion.fused_error_score(
            w["conf"],
            r_flag,
            n_flag,
            surprisal,
        )

        suggestions = []

        if (
            score >= SUGGESTION_THRESHOLD
            and not w["is_label_token"]
        ):
            suggestions = correction.suggest_corrections(
                tokens,
                i,
            )
>>>>>>> accb266 (Implement automated tests and batched ML inference)

        field_results.append(
            {
                "token": token,
                "index": i,
                "ocr_confidence": w["conf"],
                "rule_flag": r_flag,
                "nlp_flag": n_flag,
                "ml_surprisal": surprisal,
                "fused_error_score": score,
                "suggestions": suggestions,
                "left": w["left"],
                "top": w["top"],
                "width": w["width"],
                "height": w["height"],
                "field_label": w["field_label"],
                "is_label_token": w["is_label_token"],
            }
        )

    return {
        "raw_text": ocr_result["text"],
        "fields": field_results,
        "image_width": image_width,
        "image_height": image_height,
<<<<<<< HEAD
    }
=======
    }
>>>>>>> accb266 (Implement automated tests and batched ML inference)
