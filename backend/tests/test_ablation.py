import pytest

from app.services.ablation import (
    CONFIGURATIONS,
    ablation_score,
    run_ablation,
)


def test_all_ablation_configurations_exist():
    expected = {
        "ocr_only",
        "rules_only",
        "nlp_only",
        "ml_only",
        "rules_nlp",
        "without_ml",
        "full_system",
    }

    assert set(CONFIGURATIONS.keys()) == expected


def test_ablation_score_between_zero_and_one():
    score = ablation_score(
        ocr_conf=20,
        rule_flag=True,
        nlp_flag=True,
        ml_surprisal=5.0,
    )

    assert 0.0 <= score <= 1.0


def test_no_layers_returns_zero():
    score = ablation_score(
        ocr_conf=20,
        rule_flag=True,
        nlp_flag=True,
        ml_surprisal=5.0,
        use_ocr=False,
        use_rules=False,
        use_nlp=False,
        use_ml=False,
    )

    assert score == 0.0


def test_rules_only_ignores_other_signals():
    score = ablation_score(
        ocr_conf=1,
        rule_flag=False,
        nlp_flag=True,
        ml_surprisal=100,
        use_ocr=False,
        use_rules=True,
        use_nlp=False,
        use_ml=False,
    )

    assert score == 0.0


def test_rules_only_detects_rule_flag():
    score = ablation_score(
        ocr_conf=100,
        rule_flag=True,
        nlp_flag=False,
        ml_surprisal=0,
        use_ocr=False,
        use_rules=True,
        use_nlp=False,
        use_ml=False,
    )

    assert score == 1.0


def test_ocr_only_low_confidence_high_risk():
    low_confidence = ablation_score(
        10, False, False, 0,
        use_ocr=True,
        use_rules=False,
        use_nlp=False,
        use_ml=False,
    )

    high_confidence = ablation_score(
        95, False, False, 0,
        use_ocr=True,
        use_rules=False,
        use_nlp=False,
        use_ml=False,
    )

    assert low_confidence > high_confidence


def test_ml_only_high_surprisal_higher_risk():
    low = ablation_score(
        100, False, False, 0.1,
        use_ocr=False,
        use_rules=False,
        use_nlp=False,
        use_ml=True,
    )

    high = ablation_score(
        100, False, False, 5.0,
        use_ocr=False,
        use_rules=False,
        use_nlp=False,
        use_ml=True,
    )

    assert high > low


def test_run_ablation_returns_all_configurations():

    samples = [
        {
            "ocr_conf": 95,
            "rule_flag": False,
            "nlp_flag": False,
            "ml_surprisal": 0.1,
            "is_error": False,
        },
        {
            "ocr_conf": 30,
            "rule_flag": True,
            "nlp_flag": True,
            "ml_surprisal": 5.0,
            "is_error": True,
        },
    ]

    results = run_ablation(samples)

    assert set(results.keys()) == set(CONFIGURATIONS.keys())


def test_ablation_contains_metrics():

    samples = [
        {
            "ocr_conf": 95,
            "rule_flag": False,
            "nlp_flag": False,
            "ml_surprisal": 0.1,
            "is_error": False,
        },
        {
            "ocr_conf": 20,
            "rule_flag": True,
            "nlp_flag": True,
            "ml_surprisal": 5.0,
            "is_error": True,
        },
    ]

    results = run_ablation(samples)

    for result in results.values():
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "accuracy" in result


def test_invalid_threshold():

    with pytest.raises(ValueError):
        run_ablation([], threshold=1.5)