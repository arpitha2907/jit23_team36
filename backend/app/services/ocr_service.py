"""Tesseract OCR wrapper that returns per-word text + confidence + position,
which the error-detection layer needs (not just plain text)."""
import numpy as np
import pytesseract
from pytesseract import Output

from app.core.config import settings

pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def run_ocr(image: np.ndarray) -> dict:
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
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
