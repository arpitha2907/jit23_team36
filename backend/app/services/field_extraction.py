"""Groups OCR'd words into lines and matches known government-document
field labels (Name, Date of Birth, ID Number, ...) against each line, so
downstream code can tell a label ("Date of Birth") apart from its value
("15/08/1998") instead of treating every word as an anonymous token.

Detection is layout + fuzzy-match based, using data Tesseract already
provides (block_num/par_num/line_num group words into lines) plus a
configurable list of common field labels found on government ID documents.
"""
import difflib
from typing import List, Optional, Tuple

# Default set of field labels commonly found on government ID documents.
# Extend/override this list per document type as needed.
KNOWN_LABELS = [
    "Name",
    "Full Name",
    "Given Name",
    "Surname",
    "Date of Birth",
    "DOB",
    "Gender",
    "Nationality",
    "Address",
    "ID Number",
    "Aadhaar Number",
    "PAN Number",
    "Passport Number",
    "Issuing Authority",
    "Place of Birth",
    "Date of Issue",
    "Date of Expiry",
    "Valid Until",
    "Valid From",
    "Father's Name",
]

_MATCH_THRESHOLD = 0.6
_MAX_LABEL_TOKENS = 4


def _group_lines(words: List[dict]) -> List[List[int]]:
    """Groups word indices by (block_num, par_num, line_num), preserving
    Tesseract's original reading order within and across lines."""
    lines: List[List[int]] = []
    current_key = None
    current_group: List[int] = []
    for idx, w in enumerate(words):
        key = (w.get("block_num"), w.get("par_num"), w.get("line_num"))
        if key != current_key and current_group:
            lines.append(current_group)
            current_group = []
        current_key = key
        current_group.append(idx)
    if current_group:
        lines.append(current_group)
    return lines


def _find_colon_split(line_indices: List[int], words: List[dict]) -> Optional[int]:
    for pos, idx in enumerate(line_indices):
        if ":" in words[idx]["text"]:
            return pos
    return None


def _best_label_match(
    line_indices: List[int], words: List[dict]
) -> Optional[Tuple[int, str]]:
    """Tries progressively longer token prefixes of the line against
    KNOWN_LABELS and returns (tokens_in_label, matched_label) for whichever
    prefix length scores the *highest similarity ratio* overall -- not
    simply the longest prefix that clears the threshold. Without this, a
    label like "ID Number" followed by digits (no colon) would keep
    matching as the prefix grows ("ID Number 4512...") since a few
    unrelated trailing tokens don't necessarily drop the ratio below
    cutoff; picking the best-scoring length instead avoids swallowing
    value tokens into the label.
    """
    best_ratio = 0.0
    best = None
    max_len = min(_MAX_LABEL_TOKENS, len(line_indices))
    for length in range(1, max_len + 1):
        prefix_text = " ".join(
            words[i]["text"].strip(":") for i in line_indices[:length]
        ).strip()
        if not prefix_text:
            continue
        for label in KNOWN_LABELS:
            ratio = difflib.SequenceMatcher(
                None, prefix_text.lower(), label.lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = (length, label)
    if best_ratio >= _MATCH_THRESHOLD:
        return best
    return None


def annotate_field_labels(words: List[dict]) -> List[dict]:
    """Returns a new list (same order/length as `words`) where each word
    dict has two extra keys:
      - is_label_token: True if this word is part of a detected field label
      - field_label: the matched label string if this word is that
        field's *value*, else None

    Words on lines where no label could be confidently identified are left
    untouched (is_label_token=False, field_label=None) -- they're simply
    not part of any recognized label:value pair.
    """
    annotated = [dict(w) for w in words]
    for w in annotated:
        w["is_label_token"] = False
        w["field_label"] = None

    for line_indices in _group_lines(words):
        colon_pos = _find_colon_split(line_indices, words)
        if colon_pos is not None:
            label_end = colon_pos + 1  # include the colon token itself
            label_text = " ".join(
                words[i]["text"] for i in line_indices[:label_end]
            ).strip(" :")
            if not label_text:
                continue
        else:
            match = _best_label_match(line_indices, words)
            if not match:
                continue
            label_end, label_text = match

        for pos, idx in enumerate(line_indices):
            if pos < label_end:
                annotated[idx]["is_label_token"] = True
            else:
                annotated[idx]["field_label"] = label_text

    return annotated
