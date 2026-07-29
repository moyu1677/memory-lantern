"""论文表3：绑定前、绑定后、解绑后的角色与越权访问测试。"""

import pytest

from tests.helpers import api_headers, create_assessment


pytestmark = pytest.mark.permission


def test_family_access_changes_with_binding_state(
    client,
    user_factory,
    login_as,
    permission_probe,
):
    elder = user_factory(account_type="elder")
    family = user_factory(account_type="family")
    token = login_as(client, family["id"])

    permission_probe(
        "家属绑定前访问老人历史",
        client.get(f"/elders/{elder['id']}/history"),
    )

    bind_response = client.post(
        "/bindings",
        data={
            "_csrf_token": token,
            "binding_code": elder["binding_code"],
        },
    )
    assert bind_response.status_code == 302
    assert client.get(f"/elders/{elder['id']}/history").status_code == 200

    remove_response = client.post(
        f"/bindings/{elder['id']}/remove",
        data={"_csrf_token": token},
    )
    assert remove_response.status_code == 302
    permission_probe(
        "家属解绑后访问老人历史",
        client.get(f"/elders/{elder['id']}/history"),
    )


def test_unbound_family_cannot_read_assessment_or_report(
    app_module,
    client,
    user_factory,
    login_as,
    permission_probe,
):
    elder = user_factory(account_type="elder")
    family = user_factory(account_type="family")
    assessment_id = create_assessment(app_module, elder["id"])
    login_as(client, family["id"])

    permission_probe(
        "未绑定家属访问测试详情",
        client.get(f"/history/{assessment_id}"),
    )
    permission_probe(
        "未绑定家属访问专属报告",
        client.get(f"/history/{assessment_id}/report"),
    )


def test_member_cannot_open_admin_pages(
    client,
    user_factory,
    login_as,
    permission_probe,
):
    elder = user_factory(account_type="elder")
    login_as(client, elder["id"])

    permission_probe("普通用户访问用户管理", client.get("/admin/users"))
    permission_probe("普通用户访问知识审核", client.get("/admin/knowledge"))
    permission_probe(
        "普通用户访问RAG文档管理",
        client.get("/admin/rag-documents"),
    )


def test_admin_can_open_admin_pages(
    client,
    user_factory,
    login_as,
):
    admin = user_factory(account_type="admin", role="admin")
    login_as(client, admin["id"])

    assert client.get("/admin/users").status_code == 200
    assert client.get("/admin/knowledge").status_code == 200
    assert client.get("/admin/rag-documents").status_code == 200


def test_family_and_community_need_common_elder_to_message(
    app_module,
    client,
    user_factory,
    login_as,
    permission_probe,
):
    elder = user_factory(account_type="elder")
    family = user_factory(account_type="family")
    community = user_factory(account_type="community")
    token = login_as(client, family["id"])

    permission_probe(
        "未共同绑定老人时发送留言",
        client.post(
            "/messages",
            data={
                "_csrf_token": token,
                "contact": f"{community['id']}:{elder['id']}",
                "body": "测试留言",
            },
        ),
    )

    with app_module.app.app_context():
        app_module.database.bind_elder(
            family["id"],
            elder["binding_code"],
        )
        app_module.database.bind_elder(
            community["id"],
            elder["binding_code"],
        )

    allowed = client.post(
        "/messages",
        data={
            "_csrf_token": token,
            "contact": f"{community['id']}:{elder['id']}",
            "body": "共同绑定后的测试留言",
        },
    )
    assert allowed.status_code == 302


def test_family_cannot_start_cognitive_test(
    client,
    user_factory,
    login_as,
    permission_probe,
):
    family = user_factory(account_type="family")
    token = login_as(client, family["id"])
    permission_probe(
        "家属端启动老人认知测试",
        client.post(
            "/test/start",
            json={"input_language": "browser"},
            headers=api_headers(token),
        ),
        accepted=(403,),
    )


def test_elder_cannot_open_community_submission_page(
    client,
    user_factory,
    login_as,
    permission_probe,
):
    elder = user_factory(account_type="elder")
    login_as(client, elder["id"])
    permission_probe(
        "老人端访问社区投稿页面",
        client.get("/knowledge/submissions"),
        accepted=(403,),
    )

