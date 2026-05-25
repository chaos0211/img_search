from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import pymysql
from pymysql.cursors import DictCursor

from backend.app.config import settings
from backend.app.utils.security import hash_password


def _connect(database: str | None = None):
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


@contextmanager
def get_connection(database: str | None = None):
    connection = _connect(database or settings.db_name)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def fetch_all(sql: str, params: Iterable[Any] | None = None):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()


def fetch_one(sql: str, params: Iterable[Any] | None = None):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()


def execute(sql: str, params: Iterable[Any] | None = None):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.lastrowid


def init_database() -> None:
    bootstrap_connection = _connect(None)
    try:
        with bootstrap_connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{settings.db_name}` DEFAULT CHARACTER SET utf8mb4")
        bootstrap_connection.commit()
    finally:
        bootstrap_connection.close()

    statements = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(64) NOT NULL UNIQUE,
            display_name VARCHAR(64) NOT NULL,
            phone VARCHAR(32) DEFAULT NULL,
            email VARCHAR(128) DEFAULT NULL,
            organization VARCHAR(128) DEFAULT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(32) NOT NULL,
            status VARCHAR(16) NOT NULL DEFAULT 'active',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS images (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            original_name VARCHAR(255) NOT NULL,
            stored_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(512) NOT NULL,
            thumbnail_path VARCHAR(512) NOT NULL,
            feature_path VARCHAR(512) NOT NULL,
            source VARCHAR(32) NOT NULL,
            label_name VARCHAR(64) DEFAULT NULL,
            split_name VARCHAR(32) DEFAULT NULL,
            width INT NOT NULL,
            height INT NOT NULL,
            mime_type VARCHAR(64) NOT NULL,
            feature_model VARCHAR(64) NOT NULL,
            feature_dim INT NOT NULL,
            created_by BIGINT DEFAULT NULL,
            is_deleted TINYINT(1) NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_images_label (label_name),
            INDEX idx_images_source (source),
            INDEX idx_images_deleted (is_deleted),
            CONSTRAINT fk_images_created_by FOREIGN KEY (created_by) REFERENCES users(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS label_categories (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(64) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS retrieval_logs (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            query_image_id BIGINT DEFAULT NULL,
            query_name VARCHAR(255) NOT NULL,
            query_source VARCHAR(32) NOT NULL,
            index_type VARCHAR(32) NOT NULL,
            top_k INT NOT NULL,
            elapsed_ms DECIMAL(10, 3) NOT NULL,
            rerank_enabled TINYINT(1) NOT NULL DEFAULT 1,
            metrics_json JSON DEFAULT NULL,
            result_ids_json JSON DEFAULT NULL,
            created_by BIGINT DEFAULT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_retrieval_logs_index (index_type),
            CONSTRAINT fk_retrieval_logs_query_image FOREIGN KEY (query_image_id) REFERENCES images(id),
            CONSTRAINT fk_retrieval_logs_created_by FOREIGN KEY (created_by) REFERENCES users(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS duplicate_actions (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            primary_image_id BIGINT NOT NULL,
            duplicate_image_id BIGINT NOT NULL,
            similarity DECIMAL(8, 6) NOT NULL,
            threshold_value DECIMAL(5, 4) NOT NULL,
            action_type VARCHAR(32) NOT NULL,
            acted_by BIGINT DEFAULT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_duplicate_actions_primary FOREIGN KEY (primary_image_id) REFERENCES images(id),
            CONSTRAINT fk_duplicate_actions_duplicate FOREIGN KEY (duplicate_image_id) REFERENCES images(id),
            CONSTRAINT fk_duplicate_actions_user FOREIGN KEY (acted_by) REFERENCES users(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS cluster_runs (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            cluster_count INT NOT NULL,
            inertia_value DOUBLE NOT NULL,
            total_images INT NOT NULL,
            payload_json JSON NOT NULL,
            created_by BIGINT DEFAULT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_cluster_runs_user FOREIGN KEY (created_by) REFERENCES users(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

            cursor.execute("SHOW COLUMNS FROM users")
            user_columns = {row["Field"] for row in cursor.fetchall()}
            user_column_statements = {
                "phone": "ALTER TABLE users ADD COLUMN phone VARCHAR(32) DEFAULT NULL AFTER display_name",
                "email": "ALTER TABLE users ADD COLUMN email VARCHAR(128) DEFAULT NULL AFTER phone",
                "organization": "ALTER TABLE users ADD COLUMN organization VARCHAR(128) DEFAULT NULL AFTER email",
            }
            for column_name, statement in user_column_statements.items():
                if column_name not in user_columns:
                    cursor.execute(statement)
            cursor.execute("ALTER TABLE users MODIFY COLUMN password_hash VARCHAR(255) NOT NULL")

            default_users = [
                ("user", "普通用户", None, None, None, hash_password("123456"), "user"),
            ]
            for username, display_name, phone, email, organization, password_hash, role in default_users:
                cursor.execute(
                    """
                    INSERT INTO users (username, display_name, phone, email, organization, password_hash, role)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        display_name = VALUES(display_name),
                        role = VALUES(role)
                    """,
                    (username, display_name, phone, email, organization, password_hash, role),
                )

            default_labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
            for label_name in default_labels:
                cursor.execute(
                    "INSERT IGNORE INTO label_categories (name) VALUES (%s)",
                    (label_name,),
                )
