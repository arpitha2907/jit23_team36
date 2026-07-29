from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, documents
from app.core.config import settings
from app.db.database import initialize_schema

initialize_schema()

app = FastAPI(
    title="Context-Driven OCR Error Detection API",
    description=(
        "Backend for government document digitization: upload -> preprocess "
        "-> OCR -> rules+NLP+ML error detection -> correction suggestions -> "
        "encrypted, audit-logged storage."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)


@app.get("/health")
def health():
    return {"status": "ok"}
