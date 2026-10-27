"""Renders the original scanned document with accepted corrections
overlaid at each corrected word's original bounding-box position -- text
that wasn't corrected is left as the real scanned pixels rather than
redrawn, so the preview stays faithful to the input image.

Each corrected word is replaced entirely locally, measured from the old
word's actual ink rather than its OCR box (which is loose or clipped on
noisy scans):

1. the old glyph strokes are inpainted away and the scan's paper grain is
   added back, so the patch doesn't read as a smooth smudge;
2. the correction is drawn in the old word's ink color, at its measured
   x-height and baseline, and blurred to match how soft the scanned glyphs
   are.
"""
import io
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Corrections are shrunk to fit this multiple of the old word's width --
# a little overflow into the following space reads better than a visibly
# smaller word. Past `_MIN_FONT_SIZE` we accept overflow rather than
# rendering illegibly small text.
_MIN_FONT_SIZE = 8
_FIT_WIDTH_FRACTION = 1.05

# The old word's glyphs are searched this many OCR-box heights above/below
# and beside the box, since Tesseract boxes on degraded scans often clip
# descenders or a word's last letter.
_SEARCH_GROW = 0.5
_SEARCH_GROW_X = 1.0
_ERASE_PADDING = 3
_INPAINT_RADIUS = 3
# Below this grayscale spread a box has no visible ink to separate from the
# paper, and above this ink fraction it is likely dark-background text; in
# both cases the whole OCR box is erased and used for placement.
_MIN_INK_CONTRAST = 30
_MAX_INK_FRACTION = 0.6
# A row/column counts as glyph when it holds at least this fraction of the
# busiest row/column's ink -- drops stray noise specks. Rows use a lower
# bar: an ascender/descender row may hold a single letter's stem.
_INK_COLUMN_FRACTION = 0.08
_INK_ROW_FRACTION = 0.03
# Rows this dense are the word's body (x-height band, or cap height for
# all-caps/digits); sparser rows above/below are ascenders and descenders.
_BODY_PROFILE_FRACTION = 0.35
# Gap (as a fraction of glyph height) still treated as inside one word when
# finding where the old word ends horizontally; word spaces are wider.
_LETTER_GAP_FRACTION = 0.25
# Ink color is the median of this darkest fraction of stroke pixels; blur
# and noise lighten stroke edges, so the full median reads too gray.
_INK_CORE_FRACTION = 0.15
_DEFAULT_INK = (0, 0, 0)
_DEFAULT_PAPER = 255.0
# Candidate Gaussian sigmas for matching the scan's glyph softness, and
# stroke weights -- (bold font, extra dilation px), thinnest first -- for
# matching how heavy the scanned print is.
_BLUR_SIGMAS = (0.0, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0)
_STROKE_WEIGHTS = ((False, 0), (True, 0), (True, 1))
# Match-error handicap per step heavier than regular: body text is almost
# always regular weight, so bold must fit clearly better, not by noise.
_HEAVIER_WEIGHT_PENALTY = 0.05

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


def _fit_font(draw: ImageDraw.ImageDraw, text: str, body_h: int, max_w: int, bold: bool):
    """Font size whose body (x-height, or cap height when `text` has no
    lowercase letters) matches the scanned word's measured `body_h`, then
    shrunk until `text` fits `max_w`, flooring at `_MIN_FONT_SIZE`."""
    probe = "x" if any(ch.islower() for ch in text) else "H"
    size = max(_MIN_FONT_SIZE, body_h * 3)
    while size > _MIN_FONT_SIZE:
        bbox = draw.textbbox((0, 0), probe, font=_get_font(size, bold), anchor="ls")
        if (bbox[3] - bbox[1]) <= body_h:
            break
        size -= 1
    while size > _MIN_FONT_SIZE:
        bbox = draw.textbbox((0, 0), text, font=_get_font(size, bold), anchor="ls")
        if (bbox[2] - bbox[0]) <= max_w:
            break
        size -= 1
    return _get_font(size, bold)


def _ink_mask(gray: np.ndarray) -> Optional[np.ndarray]:
    """Glyph strokes in a region: pixels darker than the region's Otsu
    threshold. None when there's no clear ink/paper split."""
    if int(gray.max()) - int(gray.min()) < _MIN_INK_CONTRAST:
        return None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() / 255 > _MAX_INK_FRACTION:
        return None
    return mask


def _ink_span(profile: np.ndarray, lo: int, hi: int, gap: int, fraction: float):
    """Span of the inked runs of rows/columns (bridging gaps up to `gap`)
    that overlap the OCR box span [lo, hi): the word itself, extended past
    the box where a letter or descender was clipped, but excluding the
    neighboring lines/words that the widened search also sees."""
    if not profile.any():
        return None
    active = profile >= profile.max() * fraction
    runs, start, last = [], None, None
    for i in np.where(active)[0]:
        if start is None:
            start = i
        elif i - last > gap + 1:
            runs.append((start, last + 1))
            start = i
        last = i
    runs.append((start, last + 1))
    inside = [r for r in runs if min(r[1], hi) > max(r[0], lo)]
    if not inside:
        return None
    return int(min(r[0] for r in inside)), int(max(r[1] for r in inside))


def _softness(coverage: np.ndarray) -> float:
    """Share of the glyph area that is anti-aliased/blurred edge rather than
    solid stroke: ~0 for crisp print, rising as the scan gets blurrier.
    `coverage` is 0 (paper) .. 1 (full ink)."""
    glyph = coverage >= 0.2
    if not glyph.any():
        return 0.0
    edge = glyph & (coverage < 0.8)
    return float(edge.sum()) / float(glyph.sum())


def _stroke_width(coverage: np.ndarray) -> float:
    """Mean stroke width in px, ~2 * ink area / outline length. Unlike ink
    density it doesn't depend on which letters the word has."""
    ink = (coverage >= 0.5).astype(np.uint8)
    outline = int(ink.sum()) - int(cv2.erode(ink, np.ones((3, 3), np.uint8)).sum())
    return 2.0 * float(ink.sum()) / outline if outline else 0.0


def _measure_word(pixels: np.ndarray, smooth: np.ndarray, box) -> dict:
    """Locates the old word's real glyphs around its OCR box and samples
    the ink color, paper tone/grain, body height, baseline and glyph
    softness used to redraw it."""
    height, width = smooth.shape
    bx0, by0, bx1, by1 = box
    grow_y = int((by1 - by0) * _SEARCH_GROW) + _ERASE_PADDING
    grow_x = int((by1 - by0) * _SEARCH_GROW_X) + _ERASE_PADDING
    x0, y0 = max(bx0 - grow_x, 0), max(by0 - grow_y, 0)
    x1, y1 = min(bx1 + grow_x, width), min(by1 + grow_y, height)
    measured = {
        "region": (x0, y0, x1, y1),
        "erase": np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), np.uint8),
        "glyph_box": box,
        "body_h": max(int((by1 - by0) * 0.5), 1),
        "baseline": by1,
        "ink": _DEFAULT_INK,
        "paper": _DEFAULT_PAPER,
        "grain": 0.0,
        "softness": 0.0,
        "stroke_width": 0.0,
    }
    if x1 <= x0 or y1 <= y0:
        return measured
    # Local OCR box, padded: the fallback erase area.
    lx0, ly0 = max(bx0 - _ERASE_PADDING - x0, 0), max(by0 - _ERASE_PADDING - y0, 0)
    lx1, ly1 = bx1 + _ERASE_PADDING - x0, by1 + _ERASE_PADDING - y0
    measured["erase"][ly0:ly1, lx0:lx1] = 255

    region_smooth = smooth[y0:y1, x0:x1]
    strokes = _ink_mask(region_smooth)
    if strokes is None:
        return measured
    rows = _ink_span(
        strokes[:, bx0 - x0:bx1 - x0].sum(axis=1), by0 - y0, by1 - y0, 1, _INK_ROW_FRACTION
    )
    if rows is None:
        return measured
    letter_gap = max(2, int((rows[1] - rows[0]) * _LETTER_GAP_FRACTION))
    cols = _ink_span(
        strokes[rows[0]:rows[1]].sum(axis=0), bx0 - x0, bx1 - x0, letter_gap, _INK_COLUMN_FRACTION
    )
    if cols is None:
        return measured
    # Re-measure rows across the word's full width: a clipped last letter's
    # descender lies outside the OCR box's columns.
    rows = _ink_span(
        strokes[:, cols[0]:cols[1]].sum(axis=1), by0 - y0, by1 - y0, 1, _INK_ROW_FRACTION
    ) or rows

    glyph = np.zeros_like(strokes)
    glyph[rows[0]:rows[1], cols[0]:cols[1]] = strokes[rows[0]:rows[1], cols[0]:cols[1]]
    if not glyph.any():
        return measured
    # Dilate so anti-aliased/blurred glyph edges are erased too.
    erase = cv2.dilate(glyph, np.ones((3, 3), np.uint8), iterations=2)
    measured["erase"] = erase
    measured["glyph_box"] = (x0 + cols[0], y0 + rows[0], x0 + cols[1], y0 + rows[1])

    row_profile = glyph.sum(axis=1)
    body = np.where(row_profile >= row_profile.max() * _BODY_PROFILE_FRACTION)[0]
    measured["body_h"] = int(body[-1] - body[0] + 1)
    measured["baseline"] = y0 + int(body[-1]) + 1

    # Stroke cores are picked on the smoothed image (so noise doesn't pick
    # them) but colored from the raw scan (smoothing lightens thin strokes).
    region = pixels[y0:y1, x0:x1]
    stroke_gray = region_smooth[glyph > 0]
    core = stroke_gray <= np.quantile(stroke_gray, _INK_CORE_FRACTION)
    measured["ink"] = tuple(int(c) for c in np.median(region[glyph > 0][core], axis=0))

    # Paper = everything clear of any ink in the search region (neighboring
    # words included); grain is its median-absolute-deviation spread, so a
    # stray inked pixel can't pass for heavy scanner noise.
    paper = cv2.dilate(strokes, np.ones((3, 3), np.uint8), iterations=2) == 0
    if paper.any():
        measured["paper"] = float(np.median(region_smooth[paper]))
        paper_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)[paper].astype(np.float32)
        measured["grain"] = 1.4826 * float(np.median(np.abs(paper_gray - np.median(paper_gray))))

    contrast = measured["paper"] - float(np.median(stroke_gray[core]))
    if contrast > 0:
        gy0, gy1, gx0, gx1 = rows[0], rows[1], cols[0], cols[1]
        coverage = np.clip((measured["paper"] - region_smooth[gy0:gy1, gx0:gx1]) / contrast, 0, 1)
        measured["softness"] = _softness(coverage)
        measured["stroke_width"] = _stroke_width(coverage)
    return measured


def _erase_words(canvas: Image.Image, measures: list) -> None:
    """Inpaints the old strokes of every word, then re-adds paper grain to
    the inpainted pixels so they match the surrounding scan."""
    pixels = np.array(canvas)
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
    grain = np.zeros(pixels.shape[:2], dtype=np.float32)
    for m in measures:
        x0, y0, x1, y1 = m["region"]
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] |= m["erase"]
        grain[y0:y1, x0:x1] = np.where(m["erase"] > 0, m["grain"], grain[y0:y1, x0:x1])
    if not mask.any():
        return
    pixels = cv2.inpaint(pixels, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA).astype(np.float32)
    # Fixed seed: the same document always renders the same image.
    noise = np.random.default_rng(0).standard_normal(grain.shape).astype(np.float32) * grain
    pixels += noise[..., None]
    canvas.paste(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)))


def _blurred(layer: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(layer, (0, 0), sigma) if sigma > 0 else layer


def _as_scanned(alpha: np.ndarray) -> np.ndarray:
    """Normalizes rendered coverage the way _measure_word normalizes the
    scan's (median-filtered, and relative to the darkest stroke cores), so
    candidate and scanned glyphs are compared -- and drawn -- like for like:
    blur thins a stroke's peak, but its core still prints at full ink."""
    glyph = alpha[alpha >= 0.2]
    if not len(glyph):
        return alpha
    core = float(np.quantile(glyph, 1 - _INK_CORE_FRACTION))
    return np.clip(alpha / max(core, 1e-3), 0, 1).astype(np.float32)


def _text_coverage(canvas, draw, text, font, left, baseline, grow):
    """`text` rasterized as 0..1 ink coverage in a margin-padded patch around
    where it will be drawn, plus the patch's canvas box (None if offscreen)."""
    margin = 3 * int(np.ceil(max(_BLUR_SIGMAS))) + 2 + grow
    tx0, ty0, tx1, ty1 = draw.textbbox((left, baseline), text, font=font, anchor="ls")
    x0, y0 = max(int(tx0) - margin, 0), max(int(ty0) - margin, 0)
    x1, y1 = min(int(tx1) + margin, canvas.width), min(int(ty1) + margin, canvas.height)
    if x1 <= x0 or y1 <= y0:
        return None, None
    layer = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(layer).text((left - x0, baseline - y0), text, fill=255, font=font, anchor="ls")
    coverage = np.asarray(layer, dtype=np.float32) / 255.0
    if grow:
        coverage = cv2.dilate(coverage, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return coverage, (x0, y0, x1, y1)


def _draw_word(canvas: Image.Image, draw: ImageDraw.ImageDraw, w: dict, m: dict) -> None:
    """Alpha-composites the correction in the old word's ink color, at its
    measured body height and baseline, with the stroke weight and blur that
    best match the scanned glyphs."""
    text = w["text"].strip()
    left, _, right, _ = m["glyph_box"]
    max_w = max(int((right - left) * _FIT_WIDTH_FRACTION), 1)
    baseline = m["baseline"]
    weights = [wt for wt in _STROKE_WEIGHTS if wt[0] or not w.get("is_label_token")]

    best = None
    for bold, grow in weights:
        font = _fit_font(draw, text, m["body_h"], max_w, bold=bold)
        coverage, patch = _text_coverage(canvas, draw, text, font, left, baseline, grow)
        if coverage is None:
            return
        penalty = _STROKE_WEIGHTS.index((bold, grow)) * _HEAVIER_WEIGHT_PENALTY
        for sigma in _BLUR_SIGMAS:
            alpha = _as_scanned(_blurred(coverage, sigma))
            measured_like = cv2.medianBlur(alpha, 3)
            error = penalty + abs(_softness(measured_like) - m["softness"]) + abs(
                _stroke_width(measured_like) - m["stroke_width"]
            ) / max(m["stroke_width"], 1.0)
            if best is None or error < best[0]:
                best = (error, alpha, patch)
    _, alpha, (x0, y0, x1, y1) = best
    alpha = alpha[..., None]

    pixels = np.array(canvas.crop((x0, y0, x1, y1)), dtype=np.float32)
    ink = np.array(m["ink"], dtype=np.float32)
    composed = pixels * (1 - alpha) + ink * alpha
    # The paper under the text already carries the scan's grain; give the
    # new ink the same grain so strokes aren't conspicuously smooth.
    rng = np.random.default_rng(x0 * 7919 + y0)
    composed += rng.standard_normal(alpha.shape).astype(np.float32) * m["grain"] * alpha
    canvas.paste(Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8)), (x0, y0))


def render_layout_image(
    width: int,
    height: int,
    words: List[dict],
    base_image_bytes: Optional[bytes] = None,
) -> bytes:
    """`words`: list of dicts with left/top/width/height/text (only
    corrected words should be passed in -- unchanged OCR text is left as
    the real scanned pixels), and optionally conf (OCR confidence, used to
    resolve overlapping boxes) and is_label_token (rendered in bold).

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

    render_words = [
        w for w in _suppress_overlapping(words) if (w.get("text") or "").strip()
    ]
    pixels = np.array(canvas)
    # A light median filter keeps scanner speckle from being mistaken for
    # glyph strokes when locating and measuring the old words.
    smooth = cv2.medianBlur(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), 3)
    measures = [_measure_word(pixels, smooth, _box(w)) for w in render_words]

    # Erase every old word before drawing any replacement, so one word's
    # inpainting can't smear a neighbor's freshly drawn text.
    _erase_words(canvas, measures)
    draw = ImageDraw.Draw(canvas)
    for w, m in zip(render_words, measures):
        _draw_word(canvas, draw, w, m)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()
