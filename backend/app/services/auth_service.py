from __future__ import annotations

from backend.app.database import execute, fetch_one
from backend.app.utils.security import hash_password, verify_password


class AuthService:
    def login(self, username: str, password: str):
        username = username.strip()
        if not username or not password:
            raise ValueError("请输入账号和密码")
        user = fetch_one("SELECT * FROM users WHERE username = %s LIMIT 1", (username,))
        if not user:
            raise ValueError("用户不存在")
        if user["status"] != "active":
            raise ValueError("用户已停用")
        if not verify_password(password, user["password_hash"]):
            raise ValueError("密码错误")
        return self._serialize_user(user)

    def register(
        self,
        username: str,
        display_name: str,
        phone: str,
        email: str,
        organization: str,
        password: str,
    ):
        username = username.strip()
        display_name = display_name.strip()
        phone = phone.strip()
        email = email.strip()
        organization = organization.strip()
        self._validate_registration(username, display_name, phone, email, organization, password)
        existing = fetch_one("SELECT id FROM users WHERE username = %s LIMIT 1", (username,))
        if existing:
            raise ValueError("用户名已存在")
        execute(
            """
            INSERT INTO users (username, display_name, phone, email, organization, password_hash, role)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (username, display_name, phone, email, organization, hash_password(password), "user"),
        )
        return self.login(username, password)

    def update_profile(self, user_id: int, display_name: str, phone: str, email: str, organization: str):
        execute(
            """
            UPDATE users
            SET display_name = %s, phone = %s, email = %s, organization = %s
            WHERE id = %s
            """,
            (display_name, phone, email, organization, user_id),
        )
        user = fetch_one("SELECT * FROM users WHERE id = %s LIMIT 1", (user_id,))
        return self._serialize_user(user)

    @staticmethod
    def _validate_registration(username: str, display_name: str, phone: str, email: str, organization: str, password: str) -> None:
        if not all((username, display_name, phone, email, organization, password)):
            raise ValueError("请完整填写信息")
        if len(username) < 3 or len(username) > 32:
            raise ValueError("账号长度需为 3-32 位")
        if len(password) < 6:
            raise ValueError("密码长度不能少于 6 位")
        if "@" not in email or "." not in email:
            raise ValueError("邮箱格式不正确")

    @staticmethod
    def _serialize_user(user: dict):
        return {
            "id": int(user["id"]),
            "username": user["username"],
            "displayName": user["display_name"],
            "phone": user.get("phone"),
            "email": user.get("email"),
            "organization": user.get("organization"),
            "role": "user",
            "status": user.get("status", "active"),
            "createdAt": str(user.get("created_at", "")),
            "updatedAt": str(user.get("updated_at", "")),
        }
