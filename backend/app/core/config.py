import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration, populated from environment variables / .env.

    See ../../.env.example at the project root for the full list of
    credentials/settings this application expects to be given.
    """

    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://ocr_user:ocr_pass@localhost:5432/ocr_gov"
    )
    secret_key: str = os.getenv("SECRET_KEY", "")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # Fernet key (base64, 32 bytes) used to encrypt uploaded document blobs at
    # rest. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")

    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "tesseract")
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "eng")
    ml_model_name: str = os.getenv("ML_MODEL_NAME", "bert-base-uncased")
    ml_model_name_indic: str = os.getenv(
        "ML_MODEL_NAME_INDIC", "google/muril-base-cased"
    )
    # Optional GPT-backed suggestions for flagged words. Off by default:
    # enabling it sends digit-masked OCR text of flagged lines to OpenAI.
    llm_suggestions_enabled: bool = (
        os.getenv("LLM_SUGGESTIONS_ENABLED", "false").lower() == "true"
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    # Empty = don't send the parameter (for models without reasoning).
    openai_reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "minimal")

    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")

    @property
    def cors_origins(self) -> list:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    def model_name_for_languages(self, languages: str | list[str] | None = None) -> str:
        from app.services.languages import has_indic_language

        return self.ml_model_name_indic if has_indic_language(languages or self.ocr_languages) else self.ml_model_name

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
