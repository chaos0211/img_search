from __future__ import annotations

import base64
import hmac
import hashlib
import secrets

from backend.app.config import settings

PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 260000


def hash_password(password: str) -> str:
    salt = secrets.token_urlsafe(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        f"{settings.secret_salt}:{salt}".encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    encoded_digest = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt}${encoded_digest}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith(f"{PBKDF2_ALGORITHM}$"):
        try:
            _, iterations_text, salt, expected_digest = password_hash.split("$", 3)
            iterations = int(iterations_text)
        except ValueError:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            f"{settings.secret_salt}:{salt}".encode("utf-8"),
            iterations,
        )
        actual_digest = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return hmac.compare_digest(actual_digest, expected_digest)

    legacy_payload = f"{settings.secret_salt}:{password}".encode("utf-8")
    legacy_hash = hashlib.sha1(legacy_payload).hexdigest()
    return hmac.compare_digest(legacy_hash, password_hash)
