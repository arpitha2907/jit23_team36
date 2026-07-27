# Context-Driven OCR Error Detection for Government Document Digitization

Implementation of the pipeline in `project_plan.md`:

```
Frontend (Upload Image) -> Backend API -> Image Preprocessing (OpenCV)
  -> OCR Engine (Tesseract) -> Error Detection (Rules + NLP + ML)
  -> Error Correction (Spell + Context AI) -> Secure Storage (DB + Encryption)
```

## What's here

```
ocr-error-detection/
  backend/    FastAPI app: preprocessing, OCR, error detection/correction, auth, encrypted storage
  frontend/   React app: login, upload, flagged-field review UI
  ml/         train_fusion.py — trains the error-detection fusion classifier
  docker-compose.yml
  .env.example
```

## Credentials / environment variables needed

Copy `.env.example` to `.env` and fill in:

| Variable | What it is | Where it's used |
|---|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Database credentials | Postgres container + backend `DATABASE_URL` |
| `SECRET_KEY` | Random string used to sign JWT login tokens | Backend auth |
| `ENCRYPTION_KEY` | Fernet (AES) key used to encrypt uploaded documents at rest | Backend secure storage |

Generate `SECRET_KEY` and `ENCRYPTION_KEY` yourself if none were provided:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

No external API keys are required — OCR (Tesseract) and the contextual model (`bert-base-uncased` via HuggingFace) both run locally with no paid API calls.

## Running it (Docker Compose — recommended)

```bash
cd ocr-error-detection
cp .env.example .env   # then fill in real values
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API docs (Swagger): http://localhost:8000/docs

First-time use: register a user via the frontend's "Register" link, or directly:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "reviewer@example.gov", "password": "changeme", "role": "reviewer"}'
```

Use `role: "reviewer"` or `"admin"` for the account that will approve/reject corrections (`/documents/{id}/correct` requires it); `"uploader"` can only upload and view results.

## Running without Docker (local dev)

Backend:
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export $(cat ../.env | xargs)   # or set the vars manually
uvicorn app.main:app --reload
```
Requires Tesseract installed locally (`sudo apt install tesseract-ocr` / `brew install tesseract`) and a reachable Postgres instance.

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## API overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/auth/register` | POST | Create a user |
| `/auth/login` | POST | Get a JWT (form fields: `username`, `password`) |
| `/documents/upload` | POST | Upload an image; runs the full pipeline in the background |
| `/documents/{id}/status` | GET | Poll processing status |
| `/documents/{id}/result` | GET | Get OCR text, flagged fields, per-field scores/suggestions |
| `/documents/{id}/correct` | POST | Submit approve/reject decisions (reviewer/admin only) |
| `/documents/{id}/audit` | GET | Audit trail for a document |

## How error detection actually works

For each OCR'd token, three signals are computed (`backend/app/services/`):

1. **Rules** (`rules.py`) — regex/format checks for structured fields (Aadhaar/PAN/PIN/date patterns) plus a check for digit/letter look-alike confusion (O/0, l/1, S/5, ...).
2. **NLP** (`nlp_check.py`) — dictionary/edit-distance check; flags tokens that look like words but aren't in the dictionary.
3. **ML / context** (`ml_context.py`) — the context-driven core. Masks each token and scores how surprising the *observed* token is given the surrounding sentence, using a pre-trained BERT masked-language model. This is what catches real-word errors (a valid word that's simply wrong in context) that rules and dictionaries miss.

`fusion.py` combines OCR confidence + the three signals above into one 0–1 error probability per token. Out of the box it uses a transparent weighted heuristic; once you've trained a classifier with `ml/train_fusion.py` on labeled data, drop the resulting `fusion_model.joblib` into `backend/models/` and it's used automatically.

`correction.py` generates ranked correction suggestions (dictionary candidates + the language model's own top-k predictions for the masked position). Suggestions are never auto-applied — a reviewer approves or rejects each flagged field via the frontend / `/documents/{id}/correct`, and every decision is written to the audit log.

## Training the fusion classifier on real data

The project plan's benchmark is the ICDAR 2017/2019 Post-OCR Text Correction dataset (OCR output aligned with ground truth, with an official error-detection sub-task). To train on it:

1. Run each aligned OCR token through `rules.rule_flag`, `nlp_check.nlp_flag`, and `ml_context.token_surprisal`, and pull its OCR confidence.
2. Label it 1 if it differs from the ground truth, 0 otherwise.
3. Save as a CSV with columns `ocr_conf,rule_flag,nlp_flag,ml_surprisal,label`.
4. Run:
   ```bash
   cd ml
   pip install -r requirements.txt
   python train_fusion.py --data path/to/labeled_features.csv --out ../backend/models/fusion_model.joblib
   ```

Running `train_fusion.py` with no `--data` trains on a synthetic dataset instead, purely to smoke-test the script and confirm the pipeline picks up a trained model correctly.

## Security notes

- All frontend↔backend traffic should run behind HTTPS/TLS in any real deployment (add a reverse proxy like nginx/Caddy with a cert; docker-compose here is HTTP-only for local dev).
- Uploaded images are encrypted at rest (`ENCRYPTION_KEY`, AES via Fernet) before being written to disk; a plaintext temp copy is created only transiently for OpenCV/Tesseract and deleted immediately after processing.
- JWT-based auth with roles (`uploader`, `reviewer`, `admin`) gates who can approve/reject corrections.
- Every upload, processing run, and correction decision is written to `audit_logs` with user + timestamp.
- Treat `.env`, `SECRET_KEY`, and `ENCRYPTION_KEY` as secrets — `.gitignore` already excludes `.env`.
