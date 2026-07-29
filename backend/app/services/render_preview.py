"""Renders the original scanned document with accepted corrections
overlaid at each corrected word's original bounding-box position -- text
that wasn't corrected is left as the real scanned pixels rather than
redrawn, so the preview stays faithful to the input image.
"""
import io
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

# Text is redrawn at most this fraction of the box's width/height -- past
# that we accept overflow rather than shrinking the font past legibility.
_MIN_FONT_SIZE = 8
_FIT_WIDTH_FRACTION = 0.95

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


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int, bold: bool):
    """Shrinks the font size until `text` fits `box_w`, flooring at
    `_MIN_FONT_SIZE` (accepting overflow beyond that rather than
    rendering illegibly small text)."""
    size = max(_MIN_FONT_SIZE, int(box_h * 0.9))
    target_width = box_w * _FIT_WIDTH_FRACTION
    font = _get_font(size, bold=bold)
    while size > _MIN_FONT_SIZE:
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= target_width:
            break
        size -= 1
        font = _get_font(size, bold=bold)
    return font


def _sample_patch_color(base_img: Optional[Image.Image], box):
    """Average pixel color around a box's border, used to paint over the
    old glyph before drawing a replacement -- blends with the scan's paper
    tone instead of a stark-white patch."""
    fallback = (245, 245, 240)
    if base_img is None:
        return fallback
    x0, y0, x1, y1 = box
    margin = 3
    crop_box = (
        max(x0 - margin, 0),
        max(y0 - margin, 0),
        min(x1 + margin, base_img.width),
        min(y1 + margin, base_img.height),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return fallback
    try:
        region = base_img.crop(crop_box)
        pixels = list(region.getdata())
        if not pixels:
            return fallback
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        return (r, g, b)
    except Exception:
        return fallback


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


def render_layout_image(
    width: int,
    height: int,
    words: List[dict],
    base_image_bytes: Optional[bytes] = None,
) -> bytes:
    """`words`: list of dicts with left/top/width/height/text (only
    corrected words should be passed in -- unchanged OCR text is left as
    the real scanned pixels), and optionally conf (OCR confidence, used to
    resolve overlapping boxes) and is_label_token (rendered in bold gray).

    `base_image_bytes`: the original scanned image, used as the
    background so the preview shows the real document. Falls back to a
    blank white canvas if not given or if it fails to decode."""
    width = max(int(width or 0), 1)
    height = max(int(height or 0), 1)

    canvas = None
    if base_image_bytes:
        try:
            canvas = Image.open(io.BytesIO(base_image_bytes)).convert("RGB")
            if canvas.size != (width, height):
                canvas = canvas.resize((width, height), Image.LANCZOS)
        except Exception:
            canvas = None
    if canvas is None:
        canvas = Image.new("RGB", (width, height), color="white")

    render_words = _suppress_overlapping(words)
    draw = ImageDraw.Draw(canvas)

    for w in render_words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        box = _box(w)
        left, top, right, bottom = box
        box_w, box_h = max(right - left, 1), max(bottom - top, 1)
        is_label = bool(w.get("is_label_token"))

        draw.rectangle(box, fill=_sample_patch_color(canvas, box))
        font = _fit_font(draw, text, box_w, box_h, bold=is_label)
        color = (80, 80, 80) if is_label else (0, 0, 0)
        draw.text((left, top), text, fill=color, font=font)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()
