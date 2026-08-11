"""Four-role entry points must retain the logged-in product flow."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("account_type", "role", "path", "destination"),
    [
        ("elder", "member", "/elder", "/assistant"),
        ("family", "member", "/family", "/dashboard"),
        ("community", "member", "/community", "/dashboard"),
        ("admin", "admin", "/admin", "/admin/users"),
    ],
)
def test_each_role_has_a_protected_portal(
    client, login_as, user_factory, account_type, role, path, destination
):
    user = user_factory(account_type=account_type, role=role)
    login_as(client, user["id"])

    response = client.get(path)

    assert response.status_code == 302
    assert response.headers["Location"].endswith(destination)


def test_login_page_lists_the_four_role_entries(client):
    response = client.get("/login")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    for label in ("老人端", "家人端", "社区端", "管理员端"):
        assert label in page


def test_retired_public_voice_demo_routes_are_not_exposed(app_module):
    routes = {rule.rule for rule in app_module.app.url_map.iter_rules()}

    assert "/family/voices" not in routes
    assert "/audio/chat" not in routes


def test_login_rejects_a_role_entry_that_does_not_match_the_account(
    client, csrf_for_anonymous, user_factory
):
    user = user_factory(account_type="elder", password="Test12345")
    csrf = csrf_for_anonymous(client)

    response = client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": user["username"],
            "password": "Test12345",
            "account_type": "family",
        },
        follow_redirects=True,
    )

    assert "请选择与该账号一致的登录入口。" in response.get_data(as_text=True)
