import pytest

from app.core.security import create_access_token
from app.db import models


def test_register_ignores_client_supplied_role(client, db_session):
    resp = client.post(
        "/auth/register",
        json={"email": "escalate@example.com", "password": "hunter2", "role": "admin"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "uploader"

    stored = db_session.query(models.User).filter_by(email="escalate@example.com").first()
    assert stored.role == models.UserRole.uploader


def test_role_update_requires_admin(client):
    client.post(
        "/auth/register",
        json={"email": "plain-uploader@example.com", "password": "hunter2"},
    )
    login = client.post(
        "/auth/login",
        data={"username": "plain-uploader@example.com", "password": "hunter2"},
    )
    token = login.json()["access_token"]

    resp = client.patch(
        "/auth/users/some-id/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_missing_secret_key_raises(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "secret_key", "")
    with pytest.raises(RuntimeError):
        create_access_token({"sub": "whoever"})
