# Context-Driven OCR Error Detection for Government Document Digitization
### Progress Report

---

## 1. Description

A full-stack application that digitizes government documents (ID cards, forms, certificates) and catches OCR errors that plain OCR or spellcheck alone would miss. A user uploads a photo/scan through a React frontend; a FastAPI backend preprocesses the image (OpenCV), extracts text (Tesseract), and runs three complementary error-detection layers on every word: regex/format rules for structured fields (ID numbers, dates), a dictionary/edit-distance check for non-words, and a BERT-based contextual model that scores whether a word fits its surrounding sentence — the layer that catches *real-word* errors (a valid word substituted for the wrong one) that the other two miss. A fusion step combines all three signals into one per-token risk score, and flagged tokens get correction suggestions (deterministic digit-confusable fixes for numeric fields, dictionary + context-ranked candidates for words). A human reviewer approves or rejects each suggestion — nothing is auto-corrected silently. Documents and images are encrypted at rest, access is JWT-authenticated with roles, and every decision is audit-logged.

## 2. Tasks Completed

**Planning & research** — Project plan grounded in the ICDAR 2017/2019 Post-OCR Text Correction benchmark and recent OCR-correction research (CLOCR-C, Confidence-Aware OCR Error Detection, and a caution against naive LLM correction from "No Free Lunches").

**Core pipeline** — OpenCV preprocessing (grayscale, denoise, contrast, deskew, binarize); Tesseract OCR with per-word confidence, bounding boxes, and line/block grouping; three-layer error detection (rules, dictionary/NLP, BERT contextual scoring); a fusion step combining all signals into one error probability per token, with a heuristic fallback and a trainable classifier (`ml/train_fusion.py`).

**Correction** — Deterministic digit/letter-confusable fixes (O→0, l→1, S→5, etc.) for numeric/ID/date fields instead of a word-level model; dictionary + BERT-context candidates for ordinary words, with junk (subword pieces, punctuation) filtered out; a manual override box so a reviewer can type the correct value when no automated suggestion is right.

**Field structure** — Groups OCR'd words into lines using Tesseract's layout data and fuzzy-matches known field labels (Name, Date of Birth, ID Number, Address, etc.), so values are now tied to the field they belong to instead of being anonymous tokens.

**Layout-preserving output** — A rendering service redraws (corrected) text at each word's original bounding-box position, reconstructing a version that visually mirrors the input document's placement/alignment, with non-max suppression to avoid double-drawing when OCR produces overlapping/duplicate boxes on badly degraded text.

**App shell** — React frontend (login/register with role picker, upload with resilient polling and a "check again" fallback, review table with accept/reject/custom-correction, corrected-text and audit-trail display, layout preview image); FastAPI backend (JWT auth with roles, encrypted-at-rest document storage via Fernet/AES, full audit logging); Docker Compose setup for Postgres + backend + frontend.

**Bugs found and fixed during testing** — passlib/bcrypt version incompatibility breaking registration; oversized default PyTorch (CUDA) build slowing Docker builds, switched to CPU-only wheel; obsolete `docker-compose.yml` version key; a rules regex that missed confused dates and mis-flagged clean ones; correction suggestions using a generic word model for numeric fields instead of deterministic digit fixes; missing visual feedback on Reject; a silent 403 failure when a non-reviewer tried to save corrections; overlapping/duplicate OCR boxes corrupting the layout preview.

## 3. Tasks Pending

- **Evaluation** — no formal precision/recall/F1 or CER/WER run yet against the ICDAR benchmark; no ablation study (rules-only vs. rules+NLP vs. fully fused) has been performed.
- **Fusion classifier training on real data** — currently only smoke-tested on synthetic data; needs real ICDAR-derived labeled features to actually improve on the heuristic fallback.
- **Automated test suite** — verification so far has been manual/interactive smoke testing during development; no unit or integration tests are committed to the repo.
- **HTTPS/TLS** — the app currently runs over plain HTTP for local development; needs a reverse proxy + certificate for any real deployment.
- **Batched ML inference** — the contextual layer currently runs one BERT forward pass per token sequentially; batching would substantially cut latency on longer documents.
- **Production-scale performance work** — no line-level/seq2seq correction model (recommended for handling merged/split-token OCR errors) or bounding-box-based resegmentation heuristic has been built; these were discussed as the more efficient production approach but are not implemented.
- **Data retention policy** — no automatic expiry/cleanup of stored images or documents.

## 4. Features That Could Be Added

- A fine-tuned line-level or seq2seq correction model (T5/ByT5) to handle OCR segmentation errors (merged/split words) that per-token correction structurally cannot fix.
- Bounding-box-based resegmentation as a cheap pre-filter for suspiciously merged tokens, before any model runs.
- An escalation path to a larger general-purpose LLM only for the small subset of low-confidence, high-stakes documents, rather than as the default path.
- An active-learning loop where reviewer corrections feed back into retraining the fusion classifier over time.
- Document-type-specific field templates (distinct expected label sets for Aadhaar vs. passport vs. mark sheets, rather than one shared list).
- A grouped-by-field review UI (showing "Date of Birth: 15/08/1998" as one row) instead of the current flat per-token table with field labels shown alongside.
- Multilingual support (the ICDAR dataset already covers 10 European languages; extending to Indic languages would matter for local government use).
- Bulk/batch upload and processing for digitizing many documents at once.
- Export of the corrected, reviewed document as a searchable PDF.
- A basic analytics view (error rates, review throughput, most common error types over time).
