import pytest

from app.services.evaluation import (
    character_error_rate,
    detection_metrics,
    evaluate_ocr,
    evaluate_system,
    word_error_rate,
)


def test_cer_perfect_text():
    assert character_error_rate("AADHAAR", "AADHAAR") == 0.0


def test_cer_single_error():
    result = character_error_rate("ABCDE", "ABXDE")
    assert result == pytest.approx(1 / 5)


def test_wer_perfect_text():
    assert word_error_rate(
        "John Doe India",
        "John Doe India",
    ) == 0.0


def test_wer_single_word_error():
    result = word_error_rate(
        "John Doe India",
        "John X India",
    )
    assert result == pytest.approx(1 / 3)


def test_detection_metrics_perfect():
    truth = [True, False, True, False]
    predictions = [True, False, True, False]

    result = detection_metrics(truth, predictions)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["true_negatives"] == 2

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_detection_metrics_partial():
    truth = [True, True, False, False]
    predictions = [True, False, True, False]

    result = detection_metrics(truth, predictions)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_detection_length_mismatch():
    with pytest.raises(ValueError):
        detection_metrics(
            [True, False],
            [True],
        )


def test_evaluate_ocr():
    result = evaluate_ocr(
        "hello world",
        "hello word",
    )

    assert "cer" in result
    assert "wer" in result
    assert result["cer"] > 0
    assert result["wer"] > 0


def test_complete_system_evaluation():
    result = evaluate_system(
        reference_text="John Doe India",
        ocr_text="John D0e India",
        ground_truth_errors=[False, True, False],
        predicted_errors=[False, True, False],
    )

    assert "cer" in result
    assert "wer" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0