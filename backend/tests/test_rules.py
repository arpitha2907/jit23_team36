from app.services.rules import matches_field, normalize_confusables, rule_flag


def test_valid_aadhaar():
    assert matches_field("1234 5678 9012", "aadhaar") is True


def test_invalid_aadhaar():
    assert matches_field("1234 5678", "aadhaar") is False


def test_valid_pan():
    assert matches_field("ABCDE1234F", "pan") is True


def test_invalid_pan():
    assert matches_field("123451234F", "pan") is False


def test_valid_pincode():
    assert matches_field("560001", "pincode") is True


def test_invalid_pincode():
    assert matches_field("56001", "pincode") is False


def test_valid_date():
    assert matches_field("15/08/1998", "date") is True


def test_invalid_date():
    assert matches_field("15-Aug-1998", "date") is False


def test_normalize_letter_o():
    assert normalize_confusables("12O45") == "12045"


def test_normalize_multiple_confusables():
    assert normalize_confusables("O1S8") == "0158"


def test_rule_detects_o_confusion():
    assert rule_flag("12O456") is True


def test_rule_detects_s_confusion():
    assert rule_flag("123S56") is True


def test_rule_detects_l_confusion():
    assert rule_flag("123l56") is True


def test_clean_number_not_flagged():
    assert rule_flag("123456") is False


def test_clean_date_not_flagged():
    assert rule_flag("15/08/1998") is False


def test_normal_word_not_flagged():
    assert rule_flag("Government") is False