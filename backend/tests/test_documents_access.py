import uuid

from app.db import models


def _register_and_login(client, email: str, password: str = "hunter2") -> str:
    client.post("/auth/register", json={"email": email, "password": password})
    resp = client.post("/auth/login", data={"username": email, "password": password})
    return resp.json()["access_token"]


def _make_document(db_session, owner_id) -> models.Document:
    doc = models.Document(
        id=uuid.uuid4(),
        owner_id=owner_id,
        original_filename="pan.png",
        encrypted_image_path="/tmp/does-not-matter.enc",
        status=models.DocumentStatus.needs_review,
        image_width=800,
        image_height=600,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_uploader_cannot_access_other_uploaders_document(client, db_session):
    token_a = _register_and_login(client, "owner-a@example.com")
    user_a = db_session.query(models.User).filter_by(email="owner-a@example.com").first()
    doc = _make_document(db_session, user_a.id)

    token_b = _register_and_login(client, "owner-b@example.com")

    own_resp = client.get(f"/documents/{doc.id}/result", headers=_auth_headers(token_a))
    assert own_resp.status_code == 200

    other_resp = client.get(f"/documents/{doc.id}/result", headers=_auth_headers(token_b))
    assert other_resp.status_code == 403


def test_reviewer_is_not_ownership_restricted(client, db_session):
    token_a = _register_and_login(client, "owner-c@example.com")
    user_a = db_session.query(models.User).filter_by(email="owner-c@example.com").first()
    doc = _make_document(db_session, user_a.id)

    _register_and_login(client, "reviewer-a@example.com")
    reviewer = db_session.query(models.User).filter_by(email="reviewer-a@example.com").first()
    reviewer.role = models.UserRole.reviewer
    db_session.commit()
    reviewer_token = _register_and_login(client, "reviewer-a@example.com")

    resp = client.get(f"/documents/{doc.id}/result", headers=_auth_headers(reviewer_token))
    assert resp.status_code == 200


def test_upload_temp_file_is_named_from_doc_id_not_client_filename(client, tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    token = _register_and_login(client, "uploader-d@example.com")
    # Minimal 1x1 PNG so the magic-byte check passes.
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c6360000002000100e221bc330000"
        "000049454e44ae426082"
    )
    resp = client.post(
        "/documents/upload",
        files={"file": ("../../../../etc/passwd", png_bytes, "image/png")},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200
    doc_id = resp.json()["id"]

    written = list(tmp_path.iterdir())
    names = [p.name for p in written]
    assert f"{doc_id}.enc" in names
    assert not any("etc" in n or "passwd" in n or ".." in n for n in names)


def test_downloads_require_access_and_return_correct_formats(client, db_session):
    token = _register_and_login(client, "download-owner@example.com")
    user = db_session.query(models.User).filter_by(email="download-owner@example.com").first()
    doc = _make_document(db_session, user.id)
    doc.corrected_text = "Name: Anish Raj"
    db_session.commit()

    headers = _auth_headers(token)
    text_resp = client.get(f"/documents/{doc.id}/download/text", headers=headers)
    assert text_resp.status_code == 200
    assert text_resp.headers["content-type"].startswith("text/plain")
    assert text_resp.content == b"Name: Anish Raj"
    assert "attachment" in text_resp.headers["content-disposition"]

    image_resp = client.get(f"/documents/{doc.id}/download/image", headers=headers)
    assert image_resp.status_code == 200
    assert image_resp.headers["content-type"] == "image/png"
    assert image_resp.content.startswith(b"\x89PNG")
    assert "attachment" in image_resp.headers["content-disposition"]
