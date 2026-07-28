from app.services.fusion import build_feature_vector, fused_error_score


def test_feature_vector_length():
    features = build_feature_vector(
        ocr_conf=90,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0.1,
    )

    assert len(features) == 4


def test_features_normalized():
    features = build_feature_vector(
        ocr_conf=50,
        rule_flag=True,
        nlp_flag=True,
        ml_surprisal=1.0,
    )

    for value in features:
        assert 0.0 <= value <= 1.0


def test_high_confidence_has_lower_risk():
    high_conf = fused_error_score(
        ocr_conf=95,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0.1,
    )

    low_conf = fused_error_score(
        ocr_conf=20,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0.1,
    )

    assert high_conf < low_conf


def test_rule_flag_increases_risk():
    normal = fused_error_score(
        ocr_conf=80,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0.2,
    )

    flagged = fused_error_score(
        ocr_conf=80,
        rule_flag=True,
        nlp_flag=False,
        ml_surprisal=0.2,
    )

    assert flagged > normal


def test_nlp_flag_increases_risk():
    normal = fused_error_score(
        ocr_conf=80,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0.2,
    )

    flagged = fused_error_score(
        ocr_conf=80,
        rule_flag=False,
        nlp_flag=True,
        ml_surprisal=0.2,
    )

    assert flagged > normal


def test_surprisal_increases_risk():
    low = fused_error_score(
        ocr_conf=80,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0.1,
    )

    high = fused_error_score(
        ocr_conf=80,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=5.0,
    )

    assert high > low


def test_score_between_zero_and_one():
    score = fused_error_score(
        ocr_conf=0,
        rule_flag=True,
        nlp_flag=True,
        ml_surprisal=100,
    )

    assert 0.0 <= score <= 1.0


def test_perfect_input_low_risk():
    score = fused_error_score(
        ocr_conf=100,
        rule_flag=False,
        nlp_flag=False,
        ml_surprisal=0,
    )

    assert score == 0.0