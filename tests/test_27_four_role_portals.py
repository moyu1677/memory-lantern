"""Four-role entry points must retain the logged-in product flow."""

from __future__ import annotations

import pytest

from tests.helpers import api_headers


@pytest.mark.parametrize(
    ("account_type", "role", "path", "destination", "status_code"),
    [
        ("elder", "member", "/elder", None, 200),
        ("family", "member", "/family", "/dashboard", 302),
        ("community", "member", "/community", "/dashboard", 302),
        ("admin", "admin", "/admin", "/admin/users", 302),
    ],
)
def test_each_role_has_a_protected_portal(
    client, login_as, user_factory, account_type, role, path, destination, status_code
):
    user = user_factory(account_type=account_type, role=role)
    login_as(client, user["id"])

    response = client.get(path)

    assert response.status_code == status_code
    if destination:
        assert response.headers["Location"].endswith(destination)
    else:
        page = response.get_data(as_text=True)
        assert "从一次轻松的练习开始" in page
        assert "智慧陪伴聊天" in page


@pytest.mark.parametrize(
    ("account_type", "role", "destination"),
    [
        ("elder", "member", "/assistant"),
        ("family", "member", "/dashboard"),
        ("community", "member", "/dashboard"),
        ("admin", "admin", "/admin/users"),
    ],
)
def test_every_portal_has_a_visible_logout_that_ends_the_session(
    client, login_as, user_factory, account_type, role, destination
):
    user = user_factory(account_type=account_type, role=role)
    token = login_as(client, user["id"])

    page = client.get(destination)
    assert page.status_code == 200
    assert 'action="/logout"' in page.get_data(as_text=True)
    assert "退出登录" in page.get_data(as_text=True)

    logout = client.post("/logout", headers=api_headers(token))
    assert logout.status_code == 302
    assert logout.headers["Location"].endswith("/login")

    protected_page = client.get(destination)
    assert protected_page.status_code == 302
    assert "/login" in protected_page.headers["Location"]


def test_login_page_requires_a_portal_choice_before_showing_that_portals_login_form(client):
    response = client.get("/login")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'name="account_type"' not in page
    assert "请选择您的服务端" in page
    assert "点击一个服务端后，再使用相应账号登录。" in page
    for account_type, label in (
        ("elder", "老人端"),
        ("family", "家人端"),
        ("community", "社区端"),
        ("admin", "管理员端"),
    ):
        assert label in page
        assert f"/login?account_type={account_type}" in page
    assert 'class="auth-language-toggle"' in page
    assert "/language/en" in page
    assert "语音陪伴与智能转写" in page
    assert "已授权家人声音复刻" in page
    assert "聊天关怀提取" in page

    protected_return = client.get("/login?next=/assistant")
    assert (
        "/login?account_type=elder&amp;next=/assistant"
        in protected_return.get_data(as_text=True)
    )

    selected_response = client.get("/login?account_type=family")
    selected_page = selected_response.get_data(as_text=True)
    assert selected_response.status_code == 200
    assert 'name="account_type" value="family"' in selected_page
    assert "家人端登录" in selected_page
    assert "/register?account_type=family" in selected_page


def test_retired_public_voice_demo_routes_are_not_exposed(app_module):
    routes = {rule.rule for rule in app_module.app.url_map.iter_rules()}

    assert "/family/voices" not in routes
    assert "/audio/chat" not in routes


def test_login_rejects_a_account_when_a_different_portal_was_chosen(
    client, csrf_for_anonymous, user_factory
):
    user = user_factory(account_type="elder", password="Test12345")
    csrf = csrf_for_anonymous(client)

    response = client.post(
        "/login?account_type=family",
        data={
            "_csrf_token": csrf,
            "username": user["username"],
            "password": "Test12345",
            "account_type": "family",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "该账号不属于当前服务端，请返回后选择正确入口。" in response.get_data(as_text=True)


def test_register_portal_is_preselected_from_the_chosen_login_portal(client):
    response = client.get("/register?account_type=community")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'name="account_type" value="community"' in page
    assert "当前注册端：" in page
    assert "社区端" in page
    assert "更换服务端" in page


def test_role_navigation_keeps_reports_visible_and_elder_tools_scoped(
    app_module, client, login_as, user_factory
):
    family = user_factory(account_type="family")
    elder = user_factory(account_type="elder")
    with app_module.app.app_context():
        app_module.database.bind_elder(family["id"], elder["binding_code"])
    login_as(client, family["id"])
    family_dashboard = client.get("/dashboard").get_data(as_text=True)
    assert "老人关怀报告" in family_dashboard
    assert 'href="/family-briefs"' in family_dashboard
    assert "查看测试结果与报告" in family_dashboard
    assert 'href="/assistant"' not in family_dashboard
    assert 'href="/idiom-game"' not in family_dashboard

    community = user_factory(account_type="community")
    login_as(client, community["id"])
    community_dashboard = client.get("/dashboard").get_data(as_text=True)
    assert 'href="/messages"' in community_dashboard
    assert 'href="/family-briefs"' not in community_dashboard
    assert 'href="/assistant"' not in community_dashboard

    admin = user_factory(account_type="admin", role="admin")
    login_as(client, admin["id"])
    admin_page = client.get("/admin/users").get_data(as_text=True)
    assert 'href="/admin/users"' in admin_page
    assert 'href="/assistant"' not in admin_page
    assert 'href="/research/datasets"' not in admin_page
