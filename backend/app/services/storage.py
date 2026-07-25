"""Encrypted-at-rest file storage for uploaded document images."""
import os

from app.core import security


def save_encrypted_file(data: bytes, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    encrypted = security.encrypt_bytes(data)
    with open(dest_path, "wb") as f:
        f.write(encrypted)


def load_decrypted_file(path: str) -> bytes:
    with open(path, "rb") as f:
        encrypted = f.read()
    return security.decrypt_bytes(encrypted)
