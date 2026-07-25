"""Renders a layout-preserving reconstruction of a document: the
(corrected) text drawn at each word's original bounding-box position, so
the output visually resembles the input image's placement/alignment
instead of collapsing everything into one flat block of text.
"""
import io
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

# Installed via the `fonts-dejavu-core` package in backend/Dockerfile.
_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_font_cache = {}

# If two words' boxes overlap by more than this fraction of the smaller
# box's area, treat it as Tesseract having produced a bogus/duplicate box
# for the same patch of the image (common on heavily degraded text) and
# only draw one of them, instead of stacking text on top of itself.
_OVERLAP_SUPPRESSION_THRESHOLD = 0.3


def _get_font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    path = _FONT_BOLD if bold else _FONT_REGULAR
    try:
        font = ImageFont.truetype(path, size)
    except Exception:
        # Falls back to PIL's built-in bitmap font if the TTF isn't present
        # (e.g. running outside the Docker image) -- fixed size, but keeps
        # the endpoint working rather than erroring.
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def _box(w: dict):
    left = int(w.get("left") or 0)
    top = int(w.get("top") or 0)
    width = max(int(w.get("width") or 0), 1)
    height = max(int(w.get("height") or 0), 1)
    return left, top, left + width, top + height


def _overlap_ratio(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    smaller = min(area_a, area_b) or 1
    return intersection / smaller


def _suppress_overlapping(words: List[dict]) -> List[dict]:
    """Greedy non-max suppression: keep the highest-confidence word for any
    cluster of heavily overlapping boxes, drop the rest. Words without a
    confidence score are treated as least trustworthy (kept last)."""

    def confidence(w: dict) -> float:
        conf = w.get("conf")
        return conf if isinstance(conf, (int, float)) else -1.0

    ordered = sorted(words, key=confidence, reverse=True)
    kept: List[dict] = []
    kept_boxes = []
    for w in ordered:
        box = _box(w)
        if any(
            _overlap_ratio(box, kb) >= _OVERLAP_SUPPRESSION_THRESHOLD
            for kb in kept_boxes
        ):
            continue
        kept.append(w)
        kept_boxes.append(box)
    return kept


def render_layout_image(width: int, height: int, words: List[dict]) -> bytes:
    """`words`: list of dicts with left/top/width/height/text, and
    optionally conf (OCR confidence, used to resolve overlapping boxes)
    and is_label_token (rendered in bold gray to visually distinguish
    labels from values, same as the review UI does)."""
    width = max(int(width or 0), 1)
    height = max(int(height or 0), 1)
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)

    for w in _suppress_overlapping(words):
        text = (w.get("text") or "").strip()
        if not text:
            continue
        left = int(w.get("left") or 0)
        top = int(w.get("top") or 0)
        box_height = max(int(w.get("height") or 16), 10)
        font_size = max(8, int(box_height * 0.85))
        is_label = bool(w.get("is_label_token"))
        font = _get_font(font_size, bold=is_label)
        color = (80, 80, 80) if is_label else (0, 0, 0)
        draw.text((left, top), text, fill=color, font=font)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()
