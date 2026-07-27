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
    secret_key: str = os.getenv("SECRET_KEY", "insecure-dev-secret-change-me")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # Fernet key (base64, 32 bytes) used to encrypt uploaded document blobs at
    # rest. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")

    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "tesseract")
    ml_model_name: str = os.getenv("ML_MODEL_NAME", "bert-base-uncased")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    document_retention_days: int = int(os.getenv("DOCUMENT_RETENTION_DAYS", "30"))

    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")

    @property
    def cors_origins(self) -> list:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
