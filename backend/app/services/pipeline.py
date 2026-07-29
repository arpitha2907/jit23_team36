"""Orchestrates the full per-document OCR pipeline:

preprocess -> OCR -> field extraction -> rules/NLP/ML detection
-> fusion -> correction suggestions.
"""

from app.services import (
    correction,
    document_context,
    field_extraction,
    fusion,
    languages,
    ml_context,
    nlp_check,
    ocr_service,
    preprocessing,
)

SUGGESTION_THRESHOLD = 0.4


def process_document(
    image_path: str,
    doc_type: str = "generic",
    ocr_languages: str | list[str] | None = None,
) -> dict:
    normalized_languages = languages.normalize_language_codes(ocr_languages)
    processed_image = preprocessing.preprocess_image(image_path)

    image_height, image_width = processed_image.shape[:2]

    ocr_result = ocr_service.run_ocr(processed_image, languages=normalized_languages)

    words = field_extraction.annotate_field_labels(
        ocr_result["words"]
    )

    tokens = [w["text"] for w in words]
    normalized_doc_type = document_context.normalize_document_type(doc_type)

    # Structured fields get deterministic validators.  Only values where
    # language context is meaningful are sent through the expensive model.
    context_indices = [
        i
        for i, word in enumerate(words)
        if document_context.should_run_context_model(word, normalized_doc_type)
    ]
    surprisal_by_index = {}

    # ---------------------------------------------------------
    # BATCHED ML INFERENCE
    # ---------------------------------------------------------
    # Previously token_surprisal() was called inside the loop,
    # causing one transformer forward pass per OCR token.
    #
    # Now all token surprisals are calculated in batches.
    # ---------------------------------------------------------

    try:
        if context_indices:
            context_scores = ml_context.batch_token_surprisal(
                tokens,
                _indices=context_indices,
                language_codes=normalized_languages,
            )
            surprisal_by_index = dict(zip(context_indices, context_scores))
    except Exception:
        # ML is best-effort. The rest of the OCR/error-detection
        # pipeline must continue working if model weights are
        # unavailable or inference fails.
        surprisal_by_index = {}

    # Collect all contextual suggestions in batches before building the
    # result rows.  This avoids one BERT forward pass per flagged token.
    suggestion_indices = []
    for i, word in enumerate(words):
        if not document_context.should_run_context_model(word, normalized_doc_type):
            continue
        rule_flag = document_context.field_rule_flag(
            word["text"], word.get("field_label"), normalized_doc_type
        )
        nlp_flag = (
            nlp_check.nlp_flag(word["text"], languages=normalized_languages)
            if document_context.should_run_dictionary_check(word.get("field_label"))
            else False
        )
        score = fusion.fused_error_score(
            word["conf"],
            rule_flag,
            nlp_flag,
            surprisal_by_index.get(i, 0.0),
        )
        if score >= SUGGESTION_THRESHOLD and not word["is_label_token"]:
            suggestion_indices.append(i)

    try:
        contextual_candidates = ml_context.batch_top_k_predictions(
            tokens,
            suggestion_indices,
            language_codes=normalized_languages,
        )
    except Exception:
        contextual_candidates = {}

    field_results = []

    for i, w in enumerate(words):
        token = w["text"]

        r_flag = document_context.field_rule_flag(
            token,
            w.get("field_label"),
            normalized_doc_type,
        )

        n_flag = (
            nlp_check.nlp_flag(token, languages=normalized_languages)
            if document_context.should_run_dictionary_check(w.get("field_label"))
            else False
        )

        surprisal = surprisal_by_index.get(i, 0.0)

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
                # The batch call above already made the best-effort ML
                # attempt.  Pass an empty list on failure so correction does
                # not retry one model call per flagged token.
                contextual_candidates=contextual_candidates.get(i, []),
                language_codes=normalized_languages,
            )

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
        "doc_type": normalized_doc_type,
        "ocr_languages": "+".join(normalized_languages),
        "fields": field_results,
        "image_width": image_width,
        "image_height": image_height,
    }
