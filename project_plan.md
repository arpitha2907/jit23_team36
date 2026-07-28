# Context-Driven OCR Error Detection for Government Document Digitization
### Project Implementation Plan (v2 — full-stack architecture)

---

## 1. Problem Statement

Government digitization pipelines scan large volumes of forms and ID documents — Aadhaar cards, passports, mark sheets, land records — and run them through OCR. OCR output on these documents is frequently degraded by scan quality, stamps/seals, mixed fonts, and dense layouts, producing classic confusions like "O"→"0", "l"→"1", "S"→"5". Dictionary-only checks miss **real-word errors** (a valid word substituted for the wrong one, e.g. a date or ID number that reads as another valid-looking number). Because this data feeds legal, financial, and administrative decisions, wrong-but-plausible text is a real risk, and the corrected document must also be stored securely since it contains personally identifiable information (PII).

The system: upload a document image → read the text (OCR) → find mistakes → fix mistakes automatically (with review) → store the corrected, verified document securely.

## 2. Objectives

1. Provide a web frontend where a user uploads a document image.
2. Preprocess the image (deskew, denoise, binarize) to maximize OCR quality.
3. Extract text via OCR (Tesseract) with per-word confidence.
4. Detect likely OCR errors using three complementary layers: rule-based checks, NLP/dictionary checks, and ML/contextual (transformer) checks.
5. Correct detected errors using spellcheck plus context-aware suggestions, surfaced to a human for approval.
6. Store the final document and corrected text securely (encrypted at rest, access-controlled) in a database.

## 3. Related Work (Grounding)

- **ICDAR 2017 / ICDAR 2019 Post-OCR Text Correction Competitions** — standard public benchmark for OCR error detection + correction; ICDAR 2019 provides ~22M OCR'd characters aligned with ground truth across 10 European languages, split into detection and correction sub-tasks. Top systems used BERT fine-tuned separately for each task.
- **CLOCR-C: Context Leveraging OCR Correction with Pre-trained Language Models (2024)** — shows feeding surrounding document context into a language model materially improves correction over token-only approaches.
- **Confidence-Aware Document OCR Error Detection (2024)** — combines OCR engine confidence with model-based signals to prioritize which spans to flag; directly informs the "Error Detection" layer below.
- **OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches (2025)** — generative LLMs used naively for correction can hallucinate plausible-but-wrong fixes; reinforces treating auto-correction as a suggestion, not a silent overwrite, for sensitive government records.
- **ICDAR 2026 HIPE-OCRepair Competition** — field is moving toward LLM-assisted post-correction with human-in-the-loop review, matching this project's "flag → suggest → approve" flow.

Sources: [ICDAR 2019 Competition on Post-OCR Text Correction](https://ieeexplore.ieee.org/document/8978127/), [ICDAR2017 Competition on Post-OCR Text Correction](https://hal.science/hal-03025499/file/icdar2017-competition-post(3).pdf), [CLOCR-C](https://arxiv.org/html/2408.17428v1), [Confidence-Aware Document OCR Error Detection](https://arxiv.org/pdf/2409.04117), [No Free Lunches](https://arxiv.org/html/2502.01205v1), [ICDAR 2026 HIPE-OCRepair](https://arxiv.org/pdf/2607.08143)

## 4. Datasets (Public)

| Dataset | Why it's used | Notes |
|---|---|---|
| **ICDAR 2017/2019 Post-OCR Text Correction dataset** | Primary benchmark: real OCR output aligned character-by-character with ground truth, with an official detection sub-task | Use English + French/Dutch subsets |
| **RVL-CDIP** | Scanned forms/letters structurally similar to government paperwork | Source of realistic layouts/text for testing |
| **FUNSD** | Noisy scanned forms with OCR + layout annotations | Tests form-style documents (fields, tables) |
| **Synthetic noise-injection set** | Clean public-domain government-style text with injected OCR-like noise (confusable-glyph swaps, deletions, insertions) | Full control over ground truth; used to simulate ID-card/mark-sheet style fields (numbers, names, dates) |

## 5. System Architecture

```
 ┌─────────────────────┐
 │  Frontend (React)    │   Upload image, view results, approve/reject corrections
 │  Upload Image        │
 └──────────┬───────────┘
            │ HTTPS (multipart upload)
            v
 ┌─────────────────────┐
 │  Backend API         │   FastAPI: auth, request validation, orchestration
 │  (Python, FastAPI)    │
 └──────────┬───────────┘
            v
 ┌─────────────────────┐
 │  Image Preprocessing  │   OpenCV: grayscale, denoise, deskew, adaptive
 │  (OpenCV)             │   thresholding, contrast normalization
 └──────────┬───────────┘
            v
 ┌─────────────────────┐
 │  OCR Engine           │   Tesseract (pytesseract): text + per-word
 │  (Tesseract)          │   confidence + bounding boxes
 └──────────┬───────────┘
            v
 ┌─────────────────────┐
 │  Error Detection      │   Layer 1: Rules (regex for ID/date/number formats)
 │  (Rules + NLP + ML)   │   Layer 2: NLP (dictionary + edit distance)
 │                       │   Layer 3: ML (transformer contextual scoring)
 │                       │   → fused error-probability score per token
 └──────────┬───────────┘
            v
 ┌─────────────────────┐
 │  Error Correction     │   Spell-correction candidates (pyspellchecker/
 │  (Spell + Context AI) │   SymSpell) re-ranked by contextual LM fit
 └──────────┬───────────┘
            v
 ┌─────────────────────┐
 │  Secure Storage       │   PostgreSQL: original image (encrypted blob),
 │  (DB + Encryption)    │   OCR text, corrections, audit log; AES-256 at rest,
 │                       │   TLS in transit, JWT-authenticated access
 └───────────────────────┘
```

Design principle: corrections are surfaced to the user for **approve/reject** before being written as "final" text — appropriate given the legal/administrative weight of documents like Aadhaar or passports.

## 6. Methodology

**Step 0 — Frontend upload.** React page accepts an image (JPEG/PNG/PDF page), shows upload progress, and displays results once the backend responds.

**Step 1 — Image preprocessing (OpenCV).** Grayscale conversion, noise removal (median/Gaussian blur), deskew (Hough transform or minAreaRect), adaptive thresholding/binarization, and contrast enhancement (CLAHE) — all shown to measurably improve raw OCR accuracy before text extraction.

**Step 2 — OCR (Tesseract).** Run `pytesseract` in a mode that returns per-word confidence and bounding boxes (`image_to_data`), not just plain text — the confidence scores feed directly into error detection.

**Step 3 — Error detection (three fused layers).**
- *Rules:* regex/format validators for structured fields common in government IDs — Aadhaar's 12-digit format, date patterns, PIN codes, alphanumeric ID patterns. Catches obviously malformed fields immediately.
- *NLP:* dictionary/wordlist lookup + edit-distance to nearest valid word/name, flags out-of-vocabulary tokens.
- *ML (the context-driven core):* a masked language model (BERT/RoBERTa) scores each token's contextual plausibility given the surrounding sentence — this is what catches real-word errors the other two layers miss (e.g., a valid word that's simply the wrong one in context).
- These three signals + OCR confidence are combined by a lightweight fusion classifier (logistic regression/gradient boosting) trained on the ICDAR labeled data into one error-probability score per token/field.

**Step 4 — Error correction.** For flagged spans: generate spelling-correction candidates (SymSpell/pyspellchecker for speed) and re-rank them using the same contextual LM (which candidate best fits the sentence), producing a ranked "did you mean" list rather than a silent overwrite.

**Step 5 — Secure storage.** Once a human approves corrections (or auto-accepts above a confidence threshold, configurable), store: the original image (encrypted blob), raw OCR text, corrected text, per-field confidence, and an audit trail (who approved what, when) in PostgreSQL. Encrypt sensitive fields at the application layer (e.g. Python `cryptography` Fernet/AES-256) in addition to disk-level encryption, since this is government PII.

## 7. Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React (upload form, results viewer, approve/reject UI) |
| Backend API | FastAPI (Python) — async endpoints, request validation via Pydantic |
| Image preprocessing | OpenCV (`opencv-python`) |
| OCR | Tesseract via `pytesseract` |
| Error detection — rules | Python `re`, custom validators per document type |
| Error detection — NLP | `pyspellchecker` / SymSpell, dictionary wordlists |
| Error detection — ML | HuggingFace `transformers` (BERT/RoBERTa for masked-LM scoring) |
| Fusion classifier | scikit-learn / XGBoost |
| Database | PostgreSQL |
| Encryption | TLS (transport), `cryptography` (Fernet/AES-256, at rest), hashed/salted credentials |
| Auth | JWT (FastAPI `OAuth2PasswordBearer` or similar) |
| Containerization (optional) | Docker Compose (frontend, backend, Postgres as services) |

## 8. API Design (Backend)

| Endpoint | Method | Purpose |
|---|---|---|
| `/auth/login` | POST | Authenticate user, issue JWT |
| `/documents/upload` | POST | Accept image, kick off preprocessing → OCR → detection pipeline |
| `/documents/{id}/status` | GET | Poll processing status |
| `/documents/{id}/result` | GET | Return OCR text, flagged errors, suggested corrections |
| `/documents/{id}/correct` | POST | Submit approved/edited corrections |
| `/documents/{id}` | GET | Retrieve final stored (decrypted, permission-checked) document record |
| `/documents/{id}/audit` | GET | Retrieve audit log for that document |

## 9. Security & Storage Details

- **Encryption in transit:** HTTPS/TLS for all frontend↔backend traffic.
- **Encryption at rest:** database-level encryption (Postgres TDE or disk-level) plus application-level encryption of the image blob and any raw PII fields using AES-256 (`cryptography` Fernet), with keys managed outside the codebase (env vars/secrets manager).
- **Access control:** JWT-based auth, role-based permissions (e.g., "uploader" vs "reviewer/approver").
- **Audit logging:** every correction approval/rejection recorded with user ID and timestamp — important for government accountability.
- **Data minimization:** avoid storing raw images longer than needed if policy requires; consider configurable retention.

## 10. Evaluation Plan

- **Detection metrics:** precision, recall, F1 at token/field level vs. ICDAR ground truth, reported separately for real-word errors vs. non-word errors (to show the ML/contextual layer's specific value over rules+NLP alone).
- **Correction metrics:** character/word error rate (CER/WER) before vs. after correction (`jiwer`).
- **Ablation:** rules-only vs. rules+NLP vs. rules+NLP+ML(fused) — quantifies what each layer adds.
- **System-level testing:** end-to-end upload-to-result latency, OCR accuracy improvement from OpenCV preprocessing (with vs. without), and basic security testing (auth bypass attempts, encrypted-field verification).
- **Generalization check:** run the ICDAR-trained model on the synthetic government-style test set and FUNSD/RVL-CDIP samples.

## 11. Suggested Timeline (10 weeks)

| Week | Milestone |
|---|---|
| 1 | Environment setup, download ICDAR dataset, scaffold FastAPI + React skeleton |
| 2 | OpenCV preprocessing pipeline (deskew, denoise, binarize) + Tesseract integration |
| 3 | Rule-based + NLP/dictionary error detection layers |
| 4 | ML contextual scoring (MLM surprisal) + fusion classifier, trained on ICDAR data |
| 5 | Build synthetic government-text test set; run ablation study |
| 6 | Error correction module (spellcheck candidates re-ranked by context) |
| 7 | PostgreSQL schema, encryption at rest, JWT auth, audit logging |
| 8 | React frontend: upload flow, results/approve-reject UI |
| 9 | End-to-end integration testing, security testing |
| 10 | Final evaluation, report, demo prep |

## 12. Deliverables

1. Working full-stack application: React frontend + FastAPI backend + PostgreSQL, containerized with Docker Compose.
2. OCR + error detection + correction pipeline (rules, NLP, ML layers) with trained fusion classifier and evaluation results.
3. Encrypted, audit-logged secure storage layer.
4. Final written report: problem, related work, architecture, methodology, results, limitations.

## 13. Risks & Mitigations

- **Domain gap** (ICDAR data is newspapers/historical text, not ID documents) → mitigate with synthetic government-style test set and FUNSD/RVL-CDIP evaluation.
- **LLM hallucination in correction step** → correction is always advisory with a confidence score and human approve/reject step, never auto-applied silently.
- **Compute limits** for transformer scoring → use a smaller MLM (DistilBERT/RoBERTa-base) rather than a large LLM.
- **Class imbalance** (most tokens are correct) → use precision/recall/F1, class-weighted training for the fusion classifier.
- **PII exposure** (Aadhaar/passport data is sensitive) → application-level encryption, role-based access, audit trail, minimize retention.

## 14. Possible Extensions

- Layout-aware context (bounding boxes/table structure) for tabular government forms.
- Active-learning loop where reviewer corrections retrain the fusion classifier over time.
- Multilingual support (ICDAR 2019 covers 10 European languages; extend to Indic languages for local government use).
- Mobile capture app (photo taken directly from phone) feeding the same backend API.

---

## Implementation status

The plan above has been implemented in this folder (`backend/`, `frontend/`, `ml/`). See `README.md` for setup instructions, required credentials, and how each detection layer works. Verified during development:

- Rules + NLP + fusion layers tested directly on a garbled OCR sample: correctly flagged a heavily corrupted token (fused score 0.44) while a low-confidence real-word substitution ("Affairs" OCR'd as "Affine," a valid but wrong word) scored low on rules+NLP alone — precisely the gap the ML/contextual layer (Step 3) is designed to close.
- `ml/train_fusion.py` runs end-to-end and `fusion.py` correctly auto-loads a trained model when present.
- FastAPI app boots, registers routes, and handles register/login/auth correctly against a test database.
- All backend Python files pass syntax compilation; `docker-compose.yml` and `frontend/package.json` are valid.
