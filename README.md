# Context-Driven OCR Error Detection for Government Document Digitization

A full-stack OCR error detection system designed for government document digitization. The project combines **OpenCV**, **Tesseract OCR**, **rule-based validation**, **NLP**, and **BERT-based contextual analysis** to identify and correct OCR errors before documents are securely stored.

---

## Project Workflow

```text
Frontend (Select Document Type + Upload Image)
        │
        ▼
Backend API
        │
        ▼
Image Preprocessing (OpenCV)
        │
        ▼
OCR Engine (Tesseract)
        │
        ▼
Context-Aware Detection
(Field Schema + Rules + NLP + ML)
        │
        ▼
Human Review
(Spell Check + Context AI)
        │
        ▼
Secure Storage
(Database + Encryption)
```

---

## Project Structure

```text
OCR/
├── backend/
│   ├── FastAPI backend
│   ├── OCR processing
│   ├── Error detection & correction
│   ├── Authentication
│   └── Encrypted storage
│
├── frontend/
│   ├── React application
│   ├── Login/Register
│   ├── Upload interface
│   └── Review dashboard
│
├── ml/
│   └── train_fusion.py
│
├── docker-compose.yml
└── .env.example
```

---

# Environment Variables

Copy `.env.example` to `.env` and configure the following values.

| Variable | Description |
|----------|-------------|
| POSTGRES_USER | PostgreSQL username |
| POSTGRES_PASSWORD | PostgreSQL password |
| POSTGRES_DB | PostgreSQL database |
| SECRET_KEY | Secret key used for JWT authentication |
| ENCRYPTION_KEY | Fernet encryption key for secure document storage |

Generate an encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

No external API keys are required.

By default the project runs completely offline using:

- Tesseract OCR
- Hugging Face `bert-base-uncased` for English
- Hugging Face `google/muril-base-cased` for Indic or mixed-language context

### Optional: GPT correction suggestions

Set `LLM_SUGGESTIONS_ENABLED=true` and `OPENAI_API_KEY` (and optionally
`OPENAI_MODEL`, default `gpt-5-mini`) to add GPT suggestions for flagged
free-text words such as names, addresses and remarks. The pipeline makes one
batched call per document and puts GPT's candidates ahead of the local
dictionary/BERT ones; for words GPT judges already correct (or pure scanner
noise) the local candidates are hidden too. Reviewers still approve every
change. `OPENAI_REASONING_EFFORT` (default `minimal`) trades suggestion
quality for speed; leave it empty for models without reasoning.

This is the only feature that sends document text off the machine. Only
flagged words and the OCR line they sit on are sent, with every digit masked
as `#`, so ID numbers, dates and PIN codes never leave the server (they keep
using the deterministic digit fixes). If the call fails or the feature is
off, suggestions fall back to the local models.

## Multilingual OCR

The upload form accepts English, Assamese, Bengali, Gujarati, Hindi, Kannada,
Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu, and common
English + native-language combinations. Tesseract language data for these
languages is included in the backend image. The selected language is passed to
Tesseract, stored with the document, and used to choose the contextual model.

The English dictionary checker is disabled for native-script tokens so Indian
language words are not incorrectly marked as English spelling errors. For
production, fine-tune or evaluate MuRIL/IndicBERT on the target document type
and language; the generic model is a multilingual baseline, not a replacement
for labeled correction data.

---

# Running with Docker

```bash
cp .env.example .env
# Fill in the environment variables

docker compose up --build
```

### Services

| Service | URL |
|----------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Swagger Documentation | http://localhost:8000/docs |

---

# Running Locally

## Backend

```bash
cd backend

python -m venv .venv

# Activate virtual environment

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Requires:

- Tesseract OCR
- PostgreSQL

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# Authentication

Create a new user using the Register page or the API.

```bash
curl -X POST http://localhost:8000/auth/register \
-H "Content-Type: application/json" \
-d '{
"email":"reviewer@example.gov",
"password":"changeme",
"role":"reviewer"
}'
```

Available roles:

- uploader
- reviewer
- admin

---

# API Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/auth/register` | POST | Register a new user |
| `/auth/login` | POST | User login |
| `/documents/upload` | POST | Upload a document image with `doc_type` and `ocr_languages` (for example `eng`, `hin`, or `eng+hin`) |
| `/documents/{id}/status` | GET | Check processing status |
| `/documents/{id}/result` | GET | OCR results and suggestions |
| `/documents/{id}/correct` | POST | Submit corrections |
| `/documents/{id}/download/image` | GET | Download the layout-preserving corrected PNG |
| `/documents/{id}/download/text` | GET | Download reviewed text as UTF-8 plain text |
| `/documents/{id}/audit` | GET | View audit logs |

---

# OCR Error Detection Pipeline

Each OCR token is evaluated using three independent methods.

### 1. Rule-Based Validation

- Aadhaar format validation
- PAN validation
- Date validation
- PIN code validation
- Character confusion detection

Examples:

- O ↔ 0
- I ↔ 1
- S ↔ 5

---

### 2. NLP Validation

Dictionary-based validation with edit-distance analysis to identify misspelled or suspicious words.

---

### 3. Contextual Machine Learning

A BERT masked language model evaluates each token within its surrounding sentence.

This allows the system to detect:

- Contextually incorrect words
- Real-word OCR errors
- Tokens missed by traditional spell checkers

---

# Fusion Classifier

The final error probability combines:

- OCR confidence
- Rule-based validation
- NLP score
- Contextual ML score

The project supports:

- Weighted heuristic scoring
- Trained fusion classifier (`fusion_model.joblib`)

---

# OCR Correction

Suggested corrections are generated using:

- Dictionary candidates
- Edit-distance matching
- BERT contextual predictions

Corrections are **never applied automatically**.

A reviewer must approve or reject every suggestion.

---

# Training the Fusion Model

The project supports training on the ICDAR 2017/2019 Post-OCR Text Correction dataset.

Train the classifier:

```bash
cd ml

pip install -r requirements.txt

python train_fusion.py \
--data path/to/labeled_features.csv \
--out ../backend/models/fusion_model.joblib
```

If no dataset is supplied, a synthetic dataset is used for testing.

---

# Security

The application includes:

- JWT Authentication
- Role-based access control
- AES (Fernet) encrypted document storage
- Audit logging
- Secure upload workflow

For production deployments, HTTPS/TLS should be enabled using a reverse proxy such as Nginx or Caddy.

---

# Technologies Used

### Backend

- FastAPI
- PostgreSQL
- SQLAlchemy
- JWT Authentication

### OCR & AI

- OpenCV
- Tesseract OCR
- Hugging Face Transformers
- BERT
- PyTorch

### Frontend

- React
- Vite

### DevOps

- Docker
- Docker Compose

---

# Project Title

**JIT---OCR**
