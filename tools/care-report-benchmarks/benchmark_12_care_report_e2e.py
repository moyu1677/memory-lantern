"""Real end-to-end verification for the care-report web flow.

This script uses a disposable SQLite database and synthetic dialogue.  It
exercises the actual Flask routes and DeepSeek report call, but never records
model wording, chat text, user credentials, or audio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

from werkzeug.security import generate_password_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def login(client, user_id: int) -> str:
    """Create an isolated test session without printing user credentials."""

    token = f"e2e-csrf-{user_id}"
    with client.session_transaction() as session:
        session.clear()
        session["user_id"] = user_id
        session["_csrf_token"] = token
    return token


def main():
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法进行真实端到端验证。")

    with tempfile.TemporaryDirectory(prefix="care-report-e2e-") as temporary_dir:
        os.environ["DATABASE_PATH"] = str(Path(temporary_dir) / "e2e.db")
        os.environ["FLASK_SECRET_KEY"] = "care-report-e2e-secret"
        os.environ["SESSION_COOKIE_SECURE"] = "false"

        import app as app_module

        app_module.app.config.update(
            TESTING=True,
            DATABASE_PATH=os.environ["DATABASE_PATH"],
            SECRET_KEY="care-report-e2e-secret",
            SESSION_COOKIE_SECURE=False,
        )
        with app_module.app.app_context():
            elder_id = app_module.database.create_user(
                username="e2e_elder",
                password_hash=generate_password_hash("E2eTest123"),
                display_name="验收老人",
                account_type="elder",
            )
            family_id = app_module.database.create_user(
                username="e2e_family",
                password_hash=generate_password_hash("E2eTest123"),
                display_name="验收家人",
                account_type="family",
            )
            admin_id = app_module.database.create_user(
                username="e2e_admin",
                password_hash=generate_password_hash("E2eTest123"),
                display_name="验收管理员",
                account_type="admin",
                role="admin",
            )
            stranger_id = app_module.database.create_user(
                username="e2e_stranger",
                password_hash=generate_password_hash("E2eTest123"),
                display_name="未绑定家人",
                account_type="family",
            )
            elder = app_module.database.get_user_by_id(elder_id)
            app_module.database.bind_elder(family_id, elder["binding_code"])

        elder_client = app_module.app.test_client()
        elder_token = login(elder_client, elder_id)
        chat_statuses = []
        for message in (
            "这两天家里很安静，电视开着也没什么劲。",
            "孩子平时工作忙，我也不想总去打扰他们。",
            "要是周末有人一起坐坐，我会很高兴。",
        ):
            response = elder_client.post(
                "/chat",
                json={"message": message, "use_knowledge": False},
                headers={"X-CSRF-Token": elder_token},
            )
            if response.status_code != 200 or not response.get_json().get("reply"):
                raise RuntimeError(f"老人聊天请求失败：HTTP {response.status_code}")
            chat_statuses.append(response.status_code)

        family_client = app_module.app.test_client()
        family_token = login(family_client, family_id)
        before = family_client.get("/family-briefs")
        if before.status_code != 200:
            raise RuntimeError(f"家人报告页无法打开：HTTP {before.status_code}")
        generated = family_client.post(
            f"/family-briefs/{elder_id}/generate",
            data={"_csrf_token": family_token},
            follow_redirects=True,
        )
        generated_html = generated.get_data(as_text=True)
        if generated.status_code != 200 or "家庭关怀报告" not in generated_html:
            raise RuntimeError(f"家人报告生成失败：HTTP {generated.status_code}")
        if "这两天家里很安静" in generated_html:
            raise RuntimeError("家庭页出现了测试聊天原话。")
        with app_module.app.app_context():
            reports = app_module.database.list_care_reports(family_id)
        if not reports or reports[0]["status"] != "signals_present":
            raise RuntimeError("未生成可用的家庭关怀报告。")

        admin_client = app_module.app.test_client()
        login(admin_client, admin_id)
        validation_page = admin_client.get("/admin/care-report-validation")
        if validation_page.status_code != 200 or "27/27" not in validation_page.get_data(as_text=True):
            raise RuntimeError("管理员验证页没有显示已发布的真实评测汇总。")

        stranger_client = app_module.app.test_client()
        stranger_token = login(stranger_client, stranger_id)
        forbidden = stranger_client.post(
            f"/family-briefs/{elder_id}/generate",
            data={"_csrf_token": stranger_token},
        )
        if forbidden.status_code != 403:
            raise RuntimeError("未绑定家人没有被正确拦截。")

        print(
            json.dumps(
                {
                    "elder_chat_http_statuses": chat_statuses,
                    "family_report_generated": True,
                    "family_page_has_no_chat_quote": True,
                    "admin_validation_page_available": True,
                    "unbound_family_blocked": True,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
