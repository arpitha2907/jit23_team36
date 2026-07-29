"""Tesseract OCR wrapper that returns per-word text + confidence + position,
which the error-detection layer needs (not just plain text)."""
import numpy as np
import pytesseract
from pytesseract import Output

from app.core.config import settings
from app.services.languages import normalize_language_codes, tesseract_language

pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def validate_languages(languages: str | list[str] | None) -> str:
    """Return the Tesseract language string or raise a useful setup error."""
    codes = normalize_language_codes(languages or settings.ocr_languages)
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception as exc:  # pragma: no cover - depends on host installation
        raise ValueError(f"Tesseract language data could not be inspected: {exc}") from exc
    missing = [code for code in codes if code not in installed]
    if missing:
        raise ValueError(
            "Tesseract language data is not installed for: "
            + ", ".join(missing)
            + ". Install the matching tesseract-ocr language packages."
        )
    return tesseract_language(codes)


def run_ocr(image: np.ndarray, languages: str | list[str] | None = None) -> dict:
    lang = validate_languages(languages)
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--psm 6",
        output_type=Output.DICT,
    )
    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        words.append(
            {
                "text": text,
                "conf": conf,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "block_num": data["block_num"][i],
                "par_num": data["par_num"][i],
                "line_num": data["line_num"][i],
                "word_num": data["word_num"][i],
            }
        )
    full_text = " ".join(w["text"] for w in words)
    return {"text": full_text, "words": words}
