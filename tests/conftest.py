"""忆路明灯自动化测试的公共夹具。

所有功能测试使用独立的临时 SQLite 数据库，不读取或修改正式数据库。
DeepSeek 和腾讯云网络调用应在测试中显式 mock；真实网络性能测试放在
tests/benchmarks 中单独执行。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import pytest
from werkzeug.security import generate_password_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PERMISSION_ATTEMPTS: list[dict[str, object]] = []


@pytest.fixture(scope="session")
def app_module():
    """首次使用时再导入应用，确保应用初始化到临时数据库。"""
    runtime_directory = Path(__file__).resolve().parent / ".runtime"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    database_path = runtime_directory / f"test-{os.getpid()}.db"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{database_path}{suffix}")
        if candidate.exists():
            candidate.unlink()

    os.environ["DATABASE_PATH"] = str(database_path)
    os.environ["FLASK_SECRET_KEY"] = "test-only-secret-key"
    os.environ["SESSION_COOKIE_SECURE"] = "false"
    # 真实模型属于单独的模型测试；常规功能测试禁止下载外部模型。
    os.environ["SEMANTIC_SCORING_ENABLED"] = "false"

    import app as imported_app

    imported_app.app.config.update(
        TESTING=True,
        DATABASE_PATH=str(database_path),
        SECRET_KEY="test-only-secret-key",
        SESSION_COOKIE_SECURE=False,
    )
    yield imported_app

    imported_app.TEST_SESSIONS.clear()
    imported_app.IDIOM_GAME_SESSIONS.clear()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{database_path}{suffix}")
        if candidate.exists():
            candidate.unlink()


@pytest.fixture(autouse=True)
def clean_mutable_state(app_module):
    """每个用例前清理用户数据，保留内置知识文章和成语种子。"""
    with app_module.app.app_context():
        connection = app_module.database.get_db()
        connection.executescript(
            """
            DELETE FROM drawing_reviews;
            DELETE FROM drawing_submissions;
            DELETE FROM messages;
            DELETE FROM care_bindings;
            DELETE FROM knowledge_favorites;
            DELETE FROM knowledge_chunks WHERE source_type = 'document';
            DELETE FROM knowledge_documents;
            DELETE FROM knowledge_articles WHERE is_seed = 0;
            DELETE FROM assessments;
            DELETE FROM users;
            DELETE FROM sqlite_sequence
            WHERE name IN (
                'messages',
                'drawing_reviews',
                'drawing_submissions',
                'care_bindings',
                'knowledge_favorites',
                'knowledge_chunks',
                'knowledge_documents',
                'knowledge_articles',
                'assessments',
                'users'
            );
            """
        )
        connection.commit()
    app_module.TEST_SESSIONS.clear()
    app_module.IDIOM_GAME_SESSIONS.clear()
    yield
    app_module.TEST_SESSIONS.clear()
    app_module.IDIOM_GAME_SESSIONS.clear()


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture
def user_factory(app_module):
    counter = {"value": 0}

    def create(
        *,
        username: str | None = None,
        display_name: str | None = None,
        account_type: str = "elder",
        role: str = "member",
        password: str = "Test12345",
    ):
        counter["value"] += 1
        suffix = counter["value"]
        username_value = username or f"{account_type}_{suffix}"
        display_value = display_name or f"测试{account_type}{suffix}"
        with app_module.app.app_context():
            user_id = app_module.database.create_user(
                username=username_value,
                password_hash=generate_password_hash(password),
                display_name=display_value,
                account_type=account_type,
                role=role,
            )
            return dict(app_module.database.get_user_by_id(user_id))

    return create


@pytest.fixture
def login_as():
    """直接建立测试会话；返回可用于 API 和表单的 CSRF token。"""

    def login(client, user_id: int):
        token = f"csrf-test-{user_id}"
        with client.session_transaction() as session:
            session.clear()
            session["user_id"] = user_id
            session["_csrf_token"] = token
        return token

    return login


@pytest.fixture
def csrf_for_anonymous():
    def prepare(client):
        client.get("/login")
        with client.session_transaction() as session:
            return session["_csrf_token"]

    return prepare


@pytest.fixture
def permission_probe():
    """记录一次越权访问是否被正确拦截，并同时执行断言。"""

    def probe(name: str, response, accepted=(403, 404)):
        blocked = response.status_code in accepted
        PERMISSION_ATTEMPTS.append(
            {
                "case": name,
                "status_code": response.status_code,
                "blocked": int(blocked),
            }
        )
        assert blocked, (
            f"{name} 未被正确拦截：HTTP {response.status_code}，"
            f"期望 {accepted}"
        )

    return probe


def pytest_sessionfinish(session, exitstatus):
    """将权限探测原始结果导出，供论文统计脚本读取。"""
    if not PERMISSION_ATTEMPTS:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "permission_results.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["case", "status_code", "blocked"],
        )
        writer.writeheader()
        writer.writerows(PERMISSION_ATTEMPTS)
