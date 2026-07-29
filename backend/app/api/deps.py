from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db import models
from app.db.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> models.User:
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    user_id = payload.get("sub")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    return user


def require_role(*roles: models.UserRole):
    def checker(user: models.User = Depends(get_current_user)) -> models.User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return user

    return checker


def require_document_access(*roles: models.UserRole):
    """Fetches the requested document, 404ing if it doesn't exist.

    If `roles` is given, the caller's role must be one of them (like
    `require_role`). Regardless of `roles`, an `uploader` may only access
    documents they own -- `reviewer`/`admin` are never ownership-restricted,
    since reviewing other users' uploads is their job.
    """

    def checker(
        document_id: str,
        db: Session = Depends(get_db),
        user: models.User = Depends(get_current_user),
    ) -> models.Document:
        if roles and user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        doc = db.query(models.Document).filter(models.Document.id == document_id).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if user.role == models.UserRole.uploader and doc.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return doc

    return checker
