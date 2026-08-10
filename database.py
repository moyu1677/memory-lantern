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
    question_bank_version TEXT,
    question_ids TEXT,
    speech_metrics TEXT,
    scoring_audit TEXT,
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

CREATE TABLE IF NOT EXISTS idioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL UNIQUE,
    first_char TEXT NOT NULL,
    last_char TEXT NOT NULL,
    source TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_idioms_first_char
ON idioms(first_char, is_active);

CREATE INDEX IF NOT EXISTS idx_idioms_last_char
ON idioms(last_char, is_active);

CREATE TABLE IF NOT EXISTS knowledge_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    audience TEXT NOT NULL DEFAULT 'all'
        CHECK (audience IN ('all', 'elder', 'family', 'community')),
    keywords TEXT NOT NULL DEFAULT '',
    domains TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'pending', 'published', 'rejected', 'disabled')),
    author_id INTEGER,
    reviewer_id INTEGER,
    review_notes TEXT NOT NULL DEFAULT '',
    is_seed INTEGER NOT NULL DEFAULT 0
        CHECK (is_seed IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_at TEXT,
    published_at TEXT,
    FOREIGN KEY (author_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (reviewer_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_articles_status_audience
ON knowledge_articles(status, audience, category);

CREATE TABLE IF NOT EXISTS knowledge_favorites (
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, article_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES knowledge_articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '健康教育',
    audience TEXT NOT NULL DEFAULT 'all'
        CHECK (audience IN ('all', 'elder', 'family', 'community')),
    source_name TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'disabled')),
    chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_by INTEGER NOT NULL,
    reviewer_id INTEGER,
    review_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    published_at TEXT,
    FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE RESTRICT,
    FOREIGN KEY (reviewer_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status
ON knowledge_documents(status, audience, updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL
        CHECK (source_type IN ('article', 'document')),
    source_id INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_label TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    vector_id TEXT NOT NULL UNIQUE,
    indexed_at TEXT NOT NULL,
    UNIQUE(source_type, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source
ON knowledge_chunks(source_type, source_id, chunk_index);

CREATE TABLE IF NOT EXISTS drawing_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL,
    task_type TEXT NOT NULL
        CHECK (task_type IN ('trail', 'cube', 'clock')),
    source_scale TEXT NOT NULL DEFAULT 'research-drawing-v1',
    canvas_width INTEGER NOT NULL,
    canvas_height INTEGER NOT NULL,
    strokes_json TEXT NOT NULL,
    image_png BLOB NOT NULL,
    features_json TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    machine_score INTEGER,
    machine_confidence REAL,
    machine_status TEXT NOT NULL DEFAULT 'not_available'
        CHECK (machine_status IN ('not_available', 'predicted', 'needs_review')),
    machine_model_version TEXT,
    research_consent INTEGER NOT NULL DEFAULT 0
        CHECK (research_consent IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_drawing_submissions_user_created
ON drawing_submissions(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_drawing_submissions_review_queue
ON drawing_submissions(task_type, machine_status, created_at DESC);

CREATE TABLE IF NOT EXISTS drawing_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL,
    reviewer_id INTEGER NOT NULL,
    review_role TEXT NOT NULL DEFAULT 'rater'
        CHECK (review_role IN ('rater', 'adjudicator')),
    score INTEGER NOT NULL,
    rubric_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(submission_id, reviewer_id, review_role),
    FOREIGN KEY (submission_id) REFERENCES drawing_submissions(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewer_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_drawing_reviews_submission
ON drawing_reviews(submission_id, review_role, updated_at DESC);

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

IDIOM_SEED_VERSION = "china-idiom-0.0.3-four-char-v2"


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

    assessment_columns = {
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(assessments)"
        ).fetchall()
    }
    if "question_bank_version" not in assessment_columns:
        connection.execute(
            "ALTER TABLE assessments ADD COLUMN question_bank_version TEXT"
        )
    if "question_ids" not in assessment_columns:
        connection.execute(
            "ALTER TABLE assessments ADD COLUMN question_ids TEXT"
        )
    if "speech_metrics" not in assessment_columns:
        connection.execute(
            "ALTER TABLE assessments ADD COLUMN speech_metrics TEXT"
        )
    if "scoring_audit" not in assessment_columns:
        connection.execute(
            "ALTER TABLE assessments ADD COLUMN scoring_audit TEXT"
        )

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
    _seed_idioms(connection)
    _seed_knowledge_articles(connection)
    connection.commit()


def _seed_idioms(connection):
    current = connection.execute(
        "SELECT value FROM app_metadata WHERE key = 'idiom_seed_version'"
    ).fetchone()
    if current is not None and current["value"] == IDIOM_SEED_VERSION:
        return

    seed_path = Path(current_app.root_path) / "data" / "idioms_seed.txt"
    if not seed_path.exists():
        raise RuntimeError(f"缺少成语种子文件：{seed_path}")

    words = {
        line.strip()
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if len(line.strip()) == 4
    }
    connection.executemany(
        """
        INSERT OR IGNORE INTO idioms (
            word, first_char, last_char, source
        ) VALUES (?, ?, ?, ?)
        """,
        (
            (word, word[0], word[-1], "china-idiom-0.0.3")
            for word in words
        ),
    )
    connection.execute(
        """
        INSERT INTO app_metadata (key, value)
        VALUES ('idiom_seed_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (IDIOM_SEED_VERSION,),
    )


def _seed_knowledge_articles(connection):
    from knowledge_seed import KNOWLEDGE_ARTICLES, KNOWLEDGE_SEED_VERSION

    current = connection.execute(
        "SELECT value FROM app_metadata WHERE key = 'knowledge_seed_version'"
    ).fetchone()
    if current is not None and current["value"] == KNOWLEDGE_SEED_VERSION:
        return

    now = utc_now()
    for article in KNOWLEDGE_ARTICLES:
        connection.execute(
            """
            INSERT INTO knowledge_articles (
                slug, title, summary, content, category, audience,
                keywords, domains, source_name, source_url, status,
                is_seed, created_at, updated_at, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'published', 1, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                content = excluded.content,
                category = excluded.category,
                audience = excluded.audience,
                keywords = excluded.keywords,
                domains = excluded.domains,
                source_name = excluded.source_name,
                source_url = excluded.source_url,
                updated_at = excluded.updated_at
            """,
            (
                article["slug"],
                article["title"],
                article["summary"],
                article["content"],
                article["category"],
                article["audience"],
                article["keywords"],
                article["domains"],
                article["source_name"],
                article["source_url"],
                now,
                now,
                now,
            ),
        )
    connection.execute(
        """
        INSERT INTO app_metadata (key, value)
        VALUES ('knowledge_seed_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (KNOWLEDGE_SEED_VERSION,),
    )


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


def create_drawing_submission(
    *,
    submission_id,
    user_id,
    task_type,
    source_scale,
    canvas_width,
    canvas_height,
    strokes,
    image_png,
    features,
    machine_prediction,
    research_consent,
):
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO drawing_submissions (
            submission_id, user_id, task_type, source_scale,
            canvas_width, canvas_height, strokes_json, image_png,
            features_json, feature_version, machine_score,
            machine_confidence, machine_status, machine_model_version,
            research_consent, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            submission_id,
            user_id,
            task_type,
            source_scale,
            canvas_width,
            canvas_height,
            json.dumps(strokes, ensure_ascii=False, separators=(",", ":")),
            image_png,
            json.dumps(features, ensure_ascii=False, separators=(",", ":")),
            features["feature_version"],
            machine_prediction.get("score"),
            machine_prediction.get("confidence"),
            machine_prediction.get("status", "not_available"),
            machine_prediction.get("model_version"),
            1 if research_consent else 0,
            utc_now(),
        ),
    )
    connection.commit()
    return cursor.lastrowid


def get_drawing_submission(submission_id):
    return get_db().execute(
        """
        SELECT s.*, u.display_name, u.username
        FROM drawing_submissions AS s
        JOIN users AS u ON u.id = s.user_id
        WHERE s.id = ?
        """,
        (submission_id,),
    ).fetchone()


def list_user_drawing_submissions(user_id, limit=30):
    return get_db().execute(
        """
        SELECT s.*,
               (SELECT COUNT(*) FROM drawing_reviews AS r
                WHERE r.submission_id = s.id AND r.review_role = 'rater')
                   AS rater_count,
               (SELECT COUNT(*) FROM drawing_reviews AS r
                WHERE r.submission_id = s.id AND r.review_role = 'adjudicator')
                   AS adjudicator_count
        FROM drawing_submissions AS s
        WHERE s.user_id = ?
        ORDER BY s.created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()


def list_drawing_submissions_for_review(limit=200):
    return get_db().execute(
        """
        SELECT s.*, u.display_name, u.username,
               (SELECT COUNT(*) FROM drawing_reviews AS r
                WHERE r.submission_id = s.id AND r.review_role = 'rater')
                   AS rater_count,
               (SELECT GROUP_CONCAT(r.score, ',') FROM drawing_reviews AS r
                WHERE r.submission_id = s.id AND r.review_role = 'rater'
                ORDER BY r.updated_at) AS rater_scores,
               (SELECT r.score FROM drawing_reviews AS r
                WHERE r.submission_id = s.id AND r.review_role = 'adjudicator'
                ORDER BY r.updated_at DESC LIMIT 1) AS adjudicated_score
        FROM drawing_submissions AS s
        JOIN users AS u ON u.id = s.user_id
        ORDER BY s.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def list_drawing_reviews(submission_id):
    return get_db().execute(
        """
        SELECT r.*, u.display_name AS reviewer_name
        FROM drawing_reviews AS r
        JOIN users AS u ON u.id = r.reviewer_id
        WHERE r.submission_id = ?
        ORDER BY CASE r.review_role WHEN 'rater' THEN 0 ELSE 1 END,
                 r.updated_at
        """,
        (submission_id,),
    ).fetchall()


def save_drawing_review(
    *, submission_id, reviewer_id, review_role, score, rubric, notes
):
    connection = get_db()
    now = utc_now()
    connection.execute(
        """
        INSERT INTO drawing_reviews (
            submission_id, reviewer_id, review_role, score,
            rubric_json, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(submission_id, reviewer_id, review_role)
        DO UPDATE SET
            score = excluded.score,
            rubric_json = excluded.rubric_json,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (
            submission_id,
            reviewer_id,
            review_role,
            score,
            json.dumps(rubric, ensure_ascii=False, separators=(",", ":")),
            notes,
            now,
            now,
        ),
    )
    connection.commit()


def delete_user_drawing_submission(submission_id, user_id):
    connection = get_db()
    cursor = connection.execute(
        "DELETE FROM drawing_submissions WHERE id = ? AND user_id = ?",
        (submission_id, user_id),
    )
    connection.commit()
    return cursor.rowcount > 0


def _resolved_drawing_label(reviews):
    adjudicated = [
        int(row["score"]) for row in reviews if row["review_role"] == "adjudicator"
    ]
    if adjudicated:
        return adjudicated[-1], "adjudicated"
    rater_scores = [
        int(row["score"]) for row in reviews if row["review_role"] == "rater"
    ]
    if len(rater_scores) >= 2 and len(set(rater_scores)) == 1:
        return rater_scores[0], "rater_consensus"
    return None, "pending_review"


def drawing_review_summary(submission_id):
    reviews = list_drawing_reviews(submission_id)
    label, label_source = _resolved_drawing_label(reviews)
    return {
        "reviews": reviews,
        "resolved_score": label,
        "label_source": label_source,
    }


def list_training_drawing_rows(task_type=None):
    query = """
        SELECT * FROM drawing_submissions
        WHERE research_consent = 1
    """
    parameters = []
    if task_type:
        query += " AND task_type = ?"
        parameters.append(task_type)
    query += " ORDER BY created_at"
    submissions = get_db().execute(query, parameters).fetchall()
    rows = []
    for submission in submissions:
        reviews = list_drawing_reviews(submission["id"])
        score, label_source = _resolved_drawing_label(reviews)
        if score is None:
            continue
        rows.append(
            {
                "submission_id": submission["submission_id"],
                "task_type": submission["task_type"],
                "features": json.loads(submission["features_json"]),
                "score": score,
                "label_source": label_source,
                "created_at": submission["created_at"],
            }
        )
    return rows


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


def list_knowledge_categories(account_type):
    audience_sql = (
        "" if account_type == "admin"
        else "AND audience IN ('all', ?)"
    )
    parameters = () if account_type == "admin" else (account_type,)
    return [
        row["category"]
        for row in get_db().execute(
            f"""
            SELECT DISTINCT category
            FROM knowledge_articles
            WHERE status = 'published'
              {audience_sql}
            ORDER BY category COLLATE NOCASE
            """,
            parameters,
        ).fetchall()
    ]


def list_knowledge_articles(
    account_type,
    user_id,
    query="",
    category="",
    favorites_only=False,
    limit=100,
):
    conditions = ["a.status = 'published'"]
    parameters = []
    if account_type != "admin":
        conditions.append("a.audience IN ('all', ?)")
        parameters.append(account_type)
    normalized_query = str(query or "").strip()
    if normalized_query:
        conditions.append(
            """
            (
                a.title LIKE ? OR a.summary LIKE ? OR
                a.content LIKE ? OR a.keywords LIKE ?
            )
            """
        )
        pattern = f"%{normalized_query}%"
        parameters.extend([pattern, pattern, pattern, pattern])
    if category:
        conditions.append("a.category = ?")
        parameters.append(category)
    if favorites_only:
        conditions.append("f.user_id IS NOT NULL")
    return get_db().execute(
        f"""
        SELECT
            a.*,
            CASE WHEN f.user_id IS NULL THEN 0 ELSE 1 END AS is_favorite
        FROM knowledge_articles AS a
        LEFT JOIN knowledge_favorites AS f
          ON f.article_id = a.id AND f.user_id = ?
        WHERE {" AND ".join(conditions)}
        ORDER BY a.published_at DESC, a.id DESC
        LIMIT ?
        """,
        [user_id, *parameters, limit],
    ).fetchall()


def get_knowledge_article(article_id, user_id=None):
    return get_db().execute(
        """
        SELECT
            a.*,
            author.display_name AS author_name,
            reviewer.display_name AS reviewer_name,
            CASE WHEN f.user_id IS NULL THEN 0 ELSE 1 END AS is_favorite
        FROM knowledge_articles AS a
        LEFT JOIN users AS author ON author.id = a.author_id
        LEFT JOIN users AS reviewer ON reviewer.id = a.reviewer_id
        LEFT JOIN knowledge_favorites AS f
          ON f.article_id = a.id AND f.user_id = ?
        WHERE a.id = ?
        """,
        (user_id, article_id),
    ).fetchone()


def set_knowledge_favorite(user_id, article_id, favorite):
    connection = get_db()
    if favorite:
        connection.execute(
            """
            INSERT OR IGNORE INTO knowledge_favorites (
                user_id, article_id, created_at
            ) VALUES (?, ?, ?)
            """,
            (user_id, article_id, utc_now()),
        )
    else:
        connection.execute(
            """
            DELETE FROM knowledge_favorites
            WHERE user_id = ? AND article_id = ?
            """,
            (user_id, article_id),
        )
    connection.commit()


def create_knowledge_article(data, author_id, status="draft"):
    now = utc_now()
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO knowledge_articles (
            title, summary, content, category, audience, keywords,
            domains, source_name, source_url, status, author_id,
            created_at, updated_at, published_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["title"],
            data["summary"],
            data["content"],
            data["category"],
            data["audience"],
            data["keywords"],
            data["domains"],
            data["source_name"],
            data["source_url"],
            status,
            author_id,
            now,
            now,
            now if status == "published" else None,
        ),
    )
    connection.commit()
    return cursor.lastrowid


def update_knowledge_article(article_id, data, status=None):
    connection = get_db()
    status_sql = ""
    parameters = [
        data["title"],
        data["summary"],
        data["content"],
        data["category"],
        data["audience"],
        data["keywords"],
        data["domains"],
        data["source_name"],
        data["source_url"],
        utc_now(),
    ]
    if status is not None:
        status_sql = ", status = ?, published_at = ?"
        parameters.extend([
            status,
            utc_now() if status == "published" else None,
        ])
    parameters.append(article_id)
    connection.execute(
        f"""
        UPDATE knowledge_articles
        SET title = ?, summary = ?, content = ?, category = ?,
            audience = ?, keywords = ?, domains = ?, source_name = ?,
            source_url = ?, updated_at = ?
            {status_sql}
        WHERE id = ?
        """,
        parameters,
    )
    connection.commit()


def submit_knowledge_article(article_id, author_id):
    connection = get_db()
    cursor = connection.execute(
        """
        UPDATE knowledge_articles
        SET status = 'pending', review_notes = '', updated_at = ?
        WHERE id = ? AND author_id = ? AND status IN ('draft', 'rejected')
        """,
        (utc_now(), article_id, author_id),
    )
    connection.commit()
    return cursor.rowcount > 0


def list_user_knowledge_submissions(author_id):
    return get_db().execute(
        """
        SELECT * FROM knowledge_articles
        WHERE author_id = ? AND is_seed = 0
        ORDER BY updated_at DESC
        """,
        (author_id,),
    ).fetchall()


def list_admin_knowledge_articles(status=""):
    parameters = []
    condition = ""
    if status:
        condition = "WHERE a.status = ?"
        parameters.append(status)
    return get_db().execute(
        f"""
        SELECT
            a.*,
            author.display_name AS author_name,
            reviewer.display_name AS reviewer_name
        FROM knowledge_articles AS a
        LEFT JOIN users AS author ON author.id = a.author_id
        LEFT JOIN users AS reviewer ON reviewer.id = a.reviewer_id
        {condition}
        ORDER BY
            CASE a.status WHEN 'pending' THEN 0 ELSE 1 END,
            a.updated_at DESC
        """,
        parameters,
    ).fetchall()


def review_knowledge_article(article_id, reviewer_id, decision, notes=""):
    if decision not in {"published", "rejected"}:
        raise ValueError("不支持的审核结果")
    now = utc_now()
    connection = get_db()
    cursor = connection.execute(
        """
        UPDATE knowledge_articles
        SET status = ?, reviewer_id = ?, review_notes = ?,
            reviewed_at = ?, updated_at = ?, published_at = ?
        WHERE id = ? AND status = 'pending'
        """,
        (
            decision,
            reviewer_id,
            notes,
            now,
            now,
            now if decision == "published" else None,
            article_id,
        ),
    )
    connection.commit()
    return cursor.rowcount > 0


def list_recommended_knowledge(account_type, user_id, domains, limit=4):
    domain_names = [str(item).strip() for item in domains if str(item).strip()]
    parameters = [user_id]
    audience_sql = ""
    if account_type != "admin":
        audience_sql = "AND a.audience IN ('all', ?)"
        parameters.append(account_type)
    domain_sql = ""
    if domain_names:
        domain_sql = (
            " AND (" +
            " OR ".join("a.domains LIKE ?" for _ in domain_names) +
            ")"
        )
        parameters.extend(f"%{name}%" for name in domain_names)
    parameters.append(limit)
    return get_db().execute(
        f"""
        SELECT
            a.*,
            CASE WHEN f.user_id IS NULL THEN 0 ELSE 1 END AS is_favorite
        FROM knowledge_articles AS a
        LEFT JOIN knowledge_favorites AS f
          ON f.article_id = a.id AND f.user_id = ?
        WHERE a.status = 'published'
          {audience_sql}
          {domain_sql}
        ORDER BY
          CASE WHEN a.category = '结果解读' THEN 1 ELSE 0 END,
          a.published_at DESC,
          a.id DESC
        LIMIT ?
        """,
        parameters,
    ).fetchall()


def create_knowledge_document(data, uploaded_by):
    now = utc_now()
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO knowledge_documents (
            document_id, title, filename, content, category, audience,
            source_name, source_url, content_hash, status, uploaded_by,
            reviewer_id, created_at, updated_at, published_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["document_id"],
            data["title"],
            data["filename"],
            data["content"],
            data["category"],
            data["audience"],
            data["source_name"],
            data["source_url"],
            data["content_hash"],
            data["status"],
            uploaded_by,
            uploaded_by if data["status"] == "published" else None,
            now,
            now,
            now if data["status"] == "published" else None,
        ),
    )
    connection.commit()
    return cursor.lastrowid


def get_knowledge_document(document_id):
    return get_db().execute(
        """
        SELECT
            d.*,
            uploader.display_name AS uploaded_by_name,
            reviewer.display_name AS reviewer_name
        FROM knowledge_documents AS d
        JOIN users AS uploader ON uploader.id = d.uploaded_by
        LEFT JOIN users AS reviewer ON reviewer.id = d.reviewer_id
        WHERE d.document_id = ?
        """,
        (document_id,),
    ).fetchone()


def list_knowledge_documents(status=""):
    parameters = []
    condition = ""
    if status:
        condition = "WHERE d.status = ?"
        parameters.append(status)
    return get_db().execute(
        f"""
        SELECT
            d.*,
            uploader.display_name AS uploaded_by_name,
            reviewer.display_name AS reviewer_name
        FROM knowledge_documents AS d
        JOIN users AS uploader ON uploader.id = d.uploaded_by
        LEFT JOIN users AS reviewer ON reviewer.id = d.reviewer_id
        {condition}
        ORDER BY d.updated_at DESC
        """,
        parameters,
    ).fetchall()


def update_knowledge_document_status(
    document_id,
    status,
    reviewer_id,
    review_notes="",
):
    if status not in {"draft", "published", "disabled"}:
        raise ValueError("不支持的文档状态")
    now = utc_now()
    connection = get_db()
    cursor = connection.execute(
        """
        UPDATE knowledge_documents
        SET status = ?, reviewer_id = ?, review_notes = ?,
            updated_at = ?, published_at = ?
        WHERE document_id = ?
        """,
        (
            status,
            reviewer_id,
            review_notes,
            now,
            now if status == "published" else None,
            document_id,
        ),
    )
    connection.commit()
    return cursor.rowcount > 0


def update_knowledge_document_chunk_count(document_id, chunk_count):
    connection = get_db()
    connection.execute(
        """
        UPDATE knowledge_documents
        SET chunk_count = ?, updated_at = updated_at
        WHERE document_id = ?
        """,
        (chunk_count, document_id),
    )
    connection.commit()


def delete_knowledge_document(document_id):
    connection = get_db()
    document = connection.execute(
        "SELECT id FROM knowledge_documents WHERE document_id = ?",
        (document_id,),
    ).fetchone()
    if document is None:
        return False
    connection.execute(
        """
        DELETE FROM knowledge_chunks
        WHERE source_type = 'document' AND source_id = ?
        """,
        (document["id"],),
    )
    connection.execute(
        "DELETE FROM knowledge_documents WHERE id = ?",
        (document["id"],),
    )
    connection.commit()
    return True


def knowledge_document_hash_exists(content_hash):
    return get_db().execute(
        """
        SELECT 1 FROM knowledge_documents
        WHERE content_hash = ?
        """,
        (content_hash,),
    ).fetchone() is not None


def count_idioms():
    return get_db().execute(
        "SELECT COUNT(*) FROM idioms WHERE is_active = 1"
    ).fetchone()[0]


def idiom_exists(word):
    return get_db().execute(
        """
        SELECT 1 FROM idioms
        WHERE word = ? AND is_active = 1
        """,
        (word,),
    ).fetchone() is not None


def list_idioms_starting(first_character, excluded=(), limit=100):
    excluded = tuple(excluded)
    parameters = [first_character]
    exclusion_sql = ""
    if excluded:
        placeholders = ",".join("?" for _ in excluded)
        exclusion_sql = f"AND word NOT IN ({placeholders})"
        parameters.extend(excluded)
    parameters.append(limit)
    return [
        row["word"]
        for row in get_db().execute(
            f"""
            SELECT word FROM idioms
            WHERE first_char = ? AND is_active = 1
              {exclusion_sql}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    ]


def list_random_idioms(limit=200):
    return [
        row["word"]
        for row in get_db().execute(
            """
            SELECT word FROM idioms
            WHERE is_active = 1
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    ]


def random_chainable_idiom():
    row = get_db().execute(
        """
        SELECT current.word
        FROM idioms AS current
        WHERE current.is_active = 1
          AND EXISTS (
              SELECT 1
              FROM idioms AS next
              WHERE next.first_char = current.last_char
                AND next.is_active = 1
                AND next.word != current.word
          )
        ORDER BY RANDOM()
        LIMIT 1
        """
    ).fetchone()
    return row["word"] if row else None


def save_assessment(
    test_session_id,
    user_id,
    score,
    maximum_score,
    percentage,
    level,
    domain_scores,
    input_language,
    question_bank_version=None,
    question_ids=None,
    speech_metrics=None,
    scoring_audit=None,
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
            question_bank_version,
            question_ids,
            speech_metrics,
            scoring_audit,
            completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            question_bank_version,
            json.dumps(question_ids, ensure_ascii=False)
            if question_ids is not None
            else None,
            json.dumps(speech_metrics, ensure_ascii=False)
            if speech_metrics is not None
            else None,
            json.dumps(scoring_audit, ensure_ascii=False)
            if scoring_audit is not None
            else None,
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
