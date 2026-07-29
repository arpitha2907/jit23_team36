import pytest

from app.services.languages import (
    has_indic_language,
    normalize_language_codes,
    tesseract_language,
)


def test_normalizes_mixed_language_selection():
    assert normalize_language_codes("eng, hin + eng") == ["eng", "hin"]
    assert tesseract_language("eng+hin") == "eng+hin"
    assert has_indic_language("eng+hin") is True


def test_rejects_unknown_language():
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        normalize_language_codes("eng+fra")


def test_limits_language_combinations():
    with pytest.raises(ValueError, match="at most three"):
        normalize_language_codes("eng+hin+tam+tel")
