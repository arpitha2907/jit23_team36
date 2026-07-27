"""Evaluation metrics for OCR and OCR-error detection.

Provides:
- Precision
- Recall
- F1 score
- Character Error Rate (CER)
- Word Error Rate (WER)

No external metric library is required.
"""


def _levenshtein(reference: list, hypothesis: list) -> int:
    """Return Levenshtein edit distance between two sequences."""

    previous = list(range(len(hypothesis) + 1))

    for i, ref_item in enumerate(reference, start=1):
        current = [i]

        for j, hyp_item in enumerate(hypothesis, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (
                0 if ref_item == hyp_item else 1
            )

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER).

    CER = character edit distance / number of reference characters.
    """

    reference_chars = list(reference)
    hypothesis_chars = list(hypothesis)

    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0

    distance = _levenshtein(
        reference_chars,
        hypothesis_chars,
    )

    return distance / len(reference_chars)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER)."""

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    distance = _levenshtein(
        reference_words,
        hypothesis_words,
    )

    return distance / len(reference_words)


def detection_metrics(
    ground_truth: list[bool],
    predictions: list[bool],
) -> dict:
    """Calculate precision, recall and F1 for error detection.

    True means that the corresponding token is an OCR error.
    """

    if len(ground_truth) != len(predictions):
        raise ValueError(
            "ground_truth and predictions must have the same length"
        )

    tp = sum(
        truth and prediction
        for truth, prediction in zip(
            ground_truth,
            predictions,
        )
    )

    fp = sum(
        not truth and prediction
        for truth, prediction in zip(
            ground_truth,
            predictions,
        )
    )

    fn = sum(
        truth and not prediction
        for truth, prediction in zip(
            ground_truth,
            predictions,
        )
    )

    tn = sum(
        not truth and not prediction
        for truth, prediction in zip(
            ground_truth,
            predictions,
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(ground_truth)
        if ground_truth
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_ocr(
    reference_text: str,
    ocr_text: str,
) -> dict:
    """Evaluate OCR text against ground-truth text."""

    return {
        "cer": character_error_rate(
            reference_text,
            ocr_text,
        ),
        "wer": word_error_rate(
            reference_text,
            ocr_text,
        ),
    }


def evaluate_system(
    reference_text: str,
    ocr_text: str,
    ground_truth_errors: list[bool],
    predicted_errors: list[bool],
) -> dict:
    """Return OCR and detection metrics together."""

    ocr_metrics = evaluate_ocr(
        reference_text,
        ocr_text,
    )

    detection = detection_metrics(
        ground_truth_errors,
        predicted_errors,
    )

    return {
        **ocr_metrics,
        **detection,
    }