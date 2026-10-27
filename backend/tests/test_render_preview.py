import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.render_preview import _FONT_REGULAR as _FONT, render_layout_image


def test_layout_preview_preserves_requested_dimensions_and_png_format():
    base = io.BytesIO()
    Image.new("RGB", (120, 80), (238, 236, 226)).save(base, format="JPEG", quality=92)

    rendered = render_layout_image(
        120,
        80,
        [
            {
                "text": "Corrected",
                "left": 20,
                "top": 25,
                "width": 70,
                "height": 18,
                "conf": 85,
            }
        ],
        base_image_bytes=base.getvalue(),
    )

    output = Image.open(io.BytesIO(rendered))
    assert output.format == "PNG"
    assert output.size == (120, 80)


def _scan_with_words(words, ink=(40, 40, 120), paper=(238, 236, 226)):
    """A flat-paper "scan" of space-separated words; returns its PNG bytes
    and each word's tight ink box (what a perfect OCR box would be)."""
    img = Image.new("RGB", (400, 80), paper)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(_FONT, 32)
    boxes, x = [], 20
    for word in words.split():
        left, top, right, bottom = draw.textbbox((x, 20), word, font=font)
        draw.text((x, 20), word, fill=ink, font=font)
        boxes.append({"left": left, "top": top, "width": right - left, "height": bottom - top})
        x = right + 16
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue(), boxes


def _render(base, box, text):
    rendered = render_layout_image(400, 80, [{"text": text, **box}], base_image_bytes=base)
    return np.array(Image.open(io.BytesIO(rendered)).convert("RGB")).astype(int)


def _crop(out, left, top, right, bottom):
    return out[top:bottom, left:right]


needs_font = pytest.mark.skipif(not os.path.exists(_FONT), reason="DejaVu font not installed")


@needs_font
def test_correction_uses_original_ink_color_and_erases_old_word():
    ink, paper = (40, 40, 120), (238, 236, 226)
    base, (box,) = _scan_with_words("igsued", ink=ink, paper=paper)
    out = _render(base, box, "issued")
    left, top = box["left"], box["top"]
    right, bottom = left + box["width"], top + box["height"]

    dark = _crop(out, left, top, right, bottom).reshape(-1, 3)
    dark = dark[dark.sum(axis=1) < 400]
    assert len(dark) > 0
    # Replacement strokes take the scanned ink's blue tint, not pure black.
    assert np.abs(np.median(dark, axis=0) - ink).max() < 25

    # The "g" descender has no counterpart in "issued": it must be erased
    # back to flat paper, not left as a stale stroke, a block or speckle.
    assert np.abs(_crop(out, left, bottom - 4, right, bottom) - paper).max() < 10


@needs_font
def test_correction_keeps_original_letter_height():
    base, (box,) = _scan_with_words("Ramesh")
    out = _render(base, box, "Rakesh")

    rows = np.where((out.sum(axis=2) < 400).any(axis=1))[0]
    assert abs((rows.max() - rows.min() + 1) - box["height"]) <= 3


@needs_font
def test_clipped_ocr_box_still_erases_whole_word():
    paper = (238, 236, 226)
    base, (box,) = _scan_with_words("monkey", paper=paper)
    # Tesseract on degraded scans often clips a word's last letter and its
    # descenders -- here the "y" and its tail.
    clipped = dict(box, width=int(box["width"] * 0.8), height=int(box["height"] * 0.75))
    out = _render(base, clipped, "m")

    right = box["left"] + box["width"]
    bottom = box["top"] + box["height"]
    tail = _crop(out, box["left"] + box["width"] // 2, box["top"], right + 2, bottom + 2)
    assert np.abs(tail - paper).max() < 10


@needs_font
def test_neighboring_words_are_kept_and_not_mistaken_for_grain():
    paper = (238, 236, 226)
    base, (kumar, sharma) = _scan_with_words("Kumar Sharma", paper=paper)
    original = np.array(Image.open(io.BytesIO(base)).convert("RGB")).astype(int)
    out = _render(base, sharma, "S")

    # The untouched neighbor stays pixel-identical...
    k = (kumar["left"], kumar["top"], kumar["left"] + kumar["width"], kumar["top"] + kumar["height"])
    assert np.array_equal(_crop(out, *k), _crop(original, *k))
    # ...and its ink doesn't inflate the paper "grain" sprayed over the
    # erased letters of the shortened word.
    tail_left = sharma["left"] + sharma["width"] // 2
    tail = _crop(out, tail_left, sharma["top"], sharma["left"] + sharma["width"], sharma["top"] + sharma["height"])
    assert np.abs(tail - paper).max() < 10
