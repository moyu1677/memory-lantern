"""忆路明灯的 SQLite 数据访问层。"""

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app, g


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member'
        CHECK (role IN ('member', 'admin')),
    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_session_id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL,
    score INTEGER NOT NULL,
    maximum_score INTEGER NOT NULL,
    percentage INTEGER NOT NULL,
    result_level TEXT NOT NULL,
    domain_scores TEXT NOT NULL,
    input_language TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_assessments_user_completed
ON assessments(user_id, completed_at DESC);

CREATE TABLE IF NOT EXISTS care_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id INTEGER NOT NULL,
    caregiver_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(elder_id, caregiver_id),
    FOREIGN KEY (elder_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (caregiver_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_care_bindings_caregiver
ON care_bindings(caregiver_id, elder_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id INTEGER NOT NULL,
    sender_id INTEGER NOT NULL,
    recipient_id INTEGER NOT NULL,
    body TEXT NOT NULL,
    is_read INTEGER NOT NULL DEFAULT 0
        CHECK (is_read IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (elder_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (sender_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (recipient_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_recipient_created
ON messages(recipient_id, created_at DESC);
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def database_path():
    configured = current_app.config.get("DATABASE_PATH")
    if configured:
        return Path(configured)
    return Path(current_app.root_path) / "data" / "memory_lantern.db"


def get_db():
    if "db" not in g:
        path = database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        g.db = connection
    return g.db


def close_db(_error=None):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db():
    connection = get_db()
    connection.executescript(SCHEMA)
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(users)").fetchall()
    }
    if "account_type" not in columns:
        connection.execute("ALTER TABLE users ADD COLUMN account_type TEXT")
    if "binding_code" not in columns:
        connection.execute("ALTER TABLE users ADD COLUMN binding_code TEXT")

    # 兼容旧数据库：旧管理员仍是后台账号，旧普通用户视为老人账号。
    connection.execute(
        """
        UPDATE users
        SET account_type = CASE
            WHEN role = 'admin' THEN 'admin'
            ELSE 'elder'
        END
        WHERE account_type IS NULL OR account_type = ''
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_binding_code
        ON users(binding_code)
        WHERE binding_code IS NOT NULL
        """
    )
    elder_ids = connection.execute(
        """
        SELECT id FROM users
        WHERE account_type = 'elder'
          AND (binding_code IS NULL OR binding_code = '')
        """
    ).fetchall()
    for row in elder_ids:
        connection.execute(
            "UPDATE users SET binding_code = ? WHERE id = ?",
            (_new_binding_code(connection), row["id"]),
        )
    connection.commit()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()


def count_users():
    return get_db().execute("SELECT COUNT(*) FROM users").fetchone()[0]


def _new_binding_code(connection):
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        code = "YL-" + "".join(secrets.choice(alphabet) for _ in range(8))
        exists = connection.execute(
            "SELECT 1 FROM users WHERE binding_code = ?",
            (code,),
        ).fetchone()
        if exists is None:
            return code


def create_user(
    username,
    password_hash,
    display_name,
    account_type="elder",
    role="member",
):
    connection = get_db()
    binding_code = (
        _new_binding_code(connection)
        if account_type == "elder"
        else None
    )
    cursor = connection.execute(
        """
        INSERT INTO users (
            username, password_hash, display_name, role,
            account_type, binding_code, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            password_hash,
            display_name,
            role,
            account_type,
            binding_code,
            utc_now(),
        ),
    )
    connection.commit()
    return cursor.lastrowid


def get_user_by_id(user_id):
    return get_db().execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()


def get_user_by_username(username):
    return get_db().execute(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
        (username,),
    ).fetchone()


def update_last_login(user_id):
    connection = get_db()
    connection.execute(
        "UPDATE users SET last_login_at = ? WHERE id = ?",
        (utc_now(), user_id),
    )
    connection.commit()


def update_user_active(user_id, is_active):
    connection = get_db()
    connection.execute(
        "UPDATE users SET is_active = ? WHERE id = ?",
        (1 if is_active else 0, user_id),
    )
    connection.commit()


def update_user_password(user_id, password_hash):
    connection = get_db()
    connection.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id),
    )
    connection.commit()


def list_users_with_stats():
    return get_db().execute(
        """
        SELECT
            u.*,
            COUNT(a.id) AS assessment_count,
            MAX(a.completed_at) AS latest_assessment_at
        FROM users AS u
        LEFT JOIN assessments AS a ON a.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
        """
    ).fetchall()


def list_bound_elders(caregiver_id):
    return get_db().execute(
        """
        SELECT
            e.*,
            COUNT(a.id) AS assessment_count,
            MAX(a.completed_at) AS latest_assessment_at
        FROM care_bindings AS b
        JOIN users AS e ON e.id = b.elder_id
        LEFT JOIN assessments AS a ON a.user_id = e.id
        WHERE b.caregiver_id = ? AND e.is_active = 1
        GROUP BY e.id
        ORDER BY e.display_name COLLATE NOCASE
        """,
        (caregiver_id,),
    ).fetchall()


def list_elder_caregivers(elder_id):
    return get_db().execute(
        """
        SELECT u.*, b.created_at AS bound_at
        FROM care_bindings AS b
        JOIN users AS u ON u.id = b.caregiver_id
        WHERE b.elder_id = ? AND u.is_active = 1
        ORDER BY u.account_type, u.display_name COLLATE NOCASE
        """,
        (elder_id,),
    ).fetchall()


def is_elder_bound_to(caregiver_id, elder_id):
    return get_db().execute(
        """
        SELECT 1 FROM care_bindings
        WHERE caregiver_id = ? AND elder_id = ?
        """,
        (caregiver_id, elder_id),
    ).fetchone() is not None


def bind_elder(caregiver_id, binding_code):
    connection = get_db()
    elder = connection.execute(
        """
        SELECT * FROM users
        WHERE binding_code = ? COLLATE NOCASE
          AND account_type = 'elder'
          AND is_active = 1
        """,
        (binding_code,),
    ).fetchone()
    if elder is None:
        return None, False
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO care_bindings (
            elder_id, caregiver_id, created_at
        ) VALUES (?, ?, ?)
        """,
        (elder["id"], caregiver_id, utc_now()),
    )
    connection.commit()
    return elder, cursor.rowcount > 0


def unbind_elder(caregiver_id, elder_id):
    connection = get_db()
    cursor = connection.execute(
        """
        DELETE FROM care_bindings
        WHERE caregiver_id = ? AND elder_id = ?
        """,
        (caregiver_id, elder_id),
    )
    connection.commit()
    return cursor.rowcount > 0


def list_message_contacts(user_id):
    return get_db().execute(
        """
        SELECT DISTINCT
            contact.id,
            contact.display_name,
            contact.account_type,
            elder.id AS elder_id,
            elder.display_name AS elder_name
        FROM care_bindings AS mine
        JOIN care_bindings AS theirs
          ON theirs.elder_id = mine.elder_id
         AND theirs.caregiver_id != mine.caregiver_id
        JOIN users AS contact ON contact.id = theirs.caregiver_id
        JOIN users AS elder ON elder.id = mine.elder_id
        JOIN users AS me ON me.id = mine.caregiver_id
        WHERE mine.caregiver_id = ?
          AND contact.is_active = 1
          AND (
              (me.account_type = 'family' AND contact.account_type = 'community')
              OR
              (me.account_type = 'community' AND contact.account_type = 'family')
          )
        ORDER BY elder.display_name, contact.display_name
        """,
        (user_id,),
    ).fetchall()


def can_message_contact(sender_id, recipient_id, elder_id):
    return get_db().execute(
        """
        SELECT 1
        FROM care_bindings AS sender_binding
        JOIN care_bindings AS recipient_binding
          ON recipient_binding.elder_id = sender_binding.elder_id
        JOIN users AS sender ON sender.id = sender_binding.caregiver_id
        JOIN users AS recipient ON recipient.id = recipient_binding.caregiver_id
        WHERE sender_binding.caregiver_id = ?
          AND recipient_binding.caregiver_id = ?
          AND sender_binding.elder_id = ?
          AND (
              (sender.account_type = 'family' AND recipient.account_type = 'community')
              OR
              (sender.account_type = 'community' AND recipient.account_type = 'family')
          )
        """,
        (sender_id, recipient_id, elder_id),
    ).fetchone() is not None


def create_message(elder_id, sender_id, recipient_id, body):
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO messages (
            elder_id, sender_id, recipient_id, body, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (elder_id, sender_id, recipient_id, body, utc_now()),
    )
    connection.commit()
    return cursor.lastrowid


def list_messages(user_id, limit=200):
    return get_db().execute(
        """
        SELECT
            m.*,
            sender.display_name AS sender_name,
            sender.account_type AS sender_type,
            recipient.display_name AS recipient_name,
            elder.display_name AS elder_name
        FROM messages AS m
        JOIN users AS sender ON sender.id = m.sender_id
        JOIN users AS recipient ON recipient.id = m.recipient_id
        JOIN users AS elder ON elder.id = m.elder_id
        WHERE m.sender_id = ? OR m.recipient_id = ?
        ORDER BY m.created_at DESC
        LIMIT ?
        """,
        (user_id, user_id, limit),
    ).fetchall()


def mark_messages_read(user_id):
    connection = get_db()
    connection.execute(
        "UPDATE messages SET is_read = 1 WHERE recipient_id = ?",
        (user_id,),
    )
    connection.commit()


def count_unread_messages(user_id):
    return get_db().execute(
        """
        SELECT COUNT(*) FROM messages
        WHERE recipient_id = ? AND is_read = 0
        """,
        (user_id,),
    ).fetchone()[0]


def save_assessment(
    test_session_id,
    user_id,
    score,
    maximum_score,
    percentage,
    level,
    domain_scores,
    input_language,
):
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO assessments (
            test_session_id,
            user_id,
            score,
            maximum_score,
            percentage,
            result_level,
            domain_scores,
            input_language,
            completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            test_session_id,
            user_id,
            score,
            maximum_score,
            percentage,
            level,
            json.dumps(domain_scores, ensure_ascii=False),
            input_language,
            utc_now(),
        ),
    )
    connection.commit()
    return cursor.lastrowid


def list_assessments(user_id, limit=100):
    return get_db().execute(
        """
        SELECT *
        FROM assessments
        WHERE user_id = ?
        ORDER BY completed_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()


def count_assessments(user_id):
    return get_db().execute(
        "SELECT COUNT(*) FROM assessments WHERE user_id = ?",
        (user_id,),
    ).fetchone()[0]


def get_assessment(assessment_id):
    return get_db().execute(
        """
        SELECT
            a.*,
            u.username,
            u.display_name
        FROM assessments AS a
        JOIN users AS u ON u.id = a.user_id
        WHERE a.id = ?
        """,
        (assessment_id,),
    ).fetchone()
