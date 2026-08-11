"""论文表3：注册登录、三类角色和认知测试等核心业务功能。"""

from tests.helpers import api_headers, correct_answer, drawing_answer_payload


def test_public_registration_creates_member_account(
    app_module,
    client,
    csrf_for_anonymous,
):
    token = csrf_for_anonymous(client)
    response = client.post(
        "/register",
        data={
            "_csrf_token": token,
            "username": "new_elder",
            "display_name": "新老人",
            "password": "Test12345",
            "password_confirm": "Test12345",
            "account_type": "elder",
        },
    )
    assert response.status_code == 302

    with app_module.app.app_context():
        user = app_module.database.get_user_by_username("new_elder")
        assert user is not None
        assert user["role"] == "member"
        assert user["account_type"] == "elder"
        assert user["binding_code"].startswith("YL-")


def test_duplicate_username_and_password_confirmation(
    client,
    user_factory,
    csrf_for_anonymous,
):
    user_factory(username="duplicate")
    token = csrf_for_anonymous(client)

    duplicate = client.post(
        "/register",
        data={
            "_csrf_token": token,
            "username": "duplicate",
            "display_name": "重复账号",
            "password": "Test12345",
            "password_confirm": "Test12345",
            "account_type": "elder",
        },
    )
    assert duplicate.status_code == 200
    assert "这个账号名称已经被使用" in duplicate.get_data(as_text=True)

    mismatch = client.post(
        "/register",
        data={
            "_csrf_token": token,
            "username": "different",
            "display_name": "密码不一致",
            "password": "Test12345",
            "password_confirm": "Test54321",
            "account_type": "elder",
        },
    )
    assert mismatch.status_code == 200
    assert "两次输入的密码不一致" in mismatch.get_data(as_text=True)


def test_login_success_and_wrong_password(
    client,
    user_factory,
    csrf_for_anonymous,
):
    user_factory(username="login_user", password="Test12345")
    token = csrf_for_anonymous(client)

    wrong = client.post(
        "/login?account_type=elder",
        data={
            "_csrf_token": token,
            "username": "login_user",
            "password": "Wrong12345",
            "account_type": "elder",
        },
    )
    assert wrong.status_code == 200
    assert "账号或密码不正确" in wrong.get_data(as_text=True)

    correct = client.post(
        "/login?account_type=elder",
        data={
            "_csrf_token": token,
            "username": "login_user",
            "password": "Test12345",
            "account_type": "elder",
        },
    )
    assert correct.status_code == 302
    assert correct.headers["Location"].endswith("/elder")


def test_all_account_types_can_open_their_dashboard(
    client,
    user_factory,
    login_as,
):
    for account_type in ("elder", "family", "community"):
        user = user_factory(account_type=account_type)
        login_as(client, user["id"])
        response = client.get("/dashboard")
        assert response.status_code == 200


def test_complete_assessment_with_clock_observation_and_full_score(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])

    start = client.post(
        "/test/start",
        json={"input_language": "browser"},
        headers=api_headers(token),
    )
    assert start.status_code == 200
    assert start.get_json()["progress"] == "1/11"

    with client.session_transaction() as session:
        test_id = session["test_id"]

    for index in range(10):
        state = app_module.TEST_SESSIONS[test_id]
        question = state["questions"][state["question_index"]]
        response = client.post(
            "/test/answer",
            json={"message": correct_answer(question)},
            headers=api_headers(token),
        )
        assert response.status_code == 200
        if index < 9:
            assert response.get_json()["testing"] is True

    state = app_module.TEST_SESSIONS[test_id]
    assert state["questions"][state["question_index"]]["response_type"] == "drawing"
    response = client.post(
        "/test/drawing-answer",
        json=drawing_answer_payload(),
        headers=api_headers(token),
    )
    assert response.status_code == 201

    result = response.get_json()
    assert result["completed"] is True
    assert result["progress"] == "11/11"
    assert "总分：14/14" in result["reply"]
    assert "不构成认知障碍或其他疾病诊断" in result["reply"]

    with app_module.app.app_context():
        rows = app_module.database.list_assessments(elder["id"])
        assert len(rows) == 1
        assert rows[0]["score"] == 14
        assert rows[0]["maximum_score"] == 14
        assert rows[0]["question_bank_version"] == app_module.QUESTION_BANK_VERSION


def test_assessment_rejects_missing_state_and_empty_answer(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])

    missing = client.post(
        "/test/answer",
        json={"message": "答案"},
        headers=api_headers(token),
    )
    assert missing.status_code == 409

    client.post(
        "/test/start",
        json={"input_language": "browser"},
        headers=api_headers(token),
    )
    empty = client.post(
        "/test/answer",
        json={"message": "   "},
        headers=api_headers(token),
    )
    assert empty.status_code == 400


def test_chat_modes_do_not_call_real_network(
    app_module,
    client,
    user_factory,
    login_as,
    monkeypatch,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])

    monkeypatch.setattr(
        app_module,
        "get_ai_reply",
        lambda message, system_prompt: "普通陪伴回复",
    )
    monkeypatch.setattr(
        app_module,
        "generate_knowledge_answer",
        lambda query, account_type, client, model: (
            "知识库回复",
            [{
                "number": 1,
                "title": "测试资料",
                "source_name": "测试来源",
                "source_url": "",
                "section_label": "",
            }],
            False,
        ),
    )

    greeting = client.post(
        "/chat",
        json={"message": "你好", "use_knowledge": True},
        headers=api_headers(token),
    )
    assert greeting.status_code == 200
    assert greeting.get_json()["reply"] == "普通陪伴回复"
    assert greeting.get_json()["knowledge_used"] is False

    knowledge = client.post(
        "/chat",
        json={"message": "老人晚上睡不好怎么办", "use_knowledge": True},
        headers=api_headers(token),
    )
    assert knowledge.status_code == 200
    assert knowledge.get_json()["reply"] == "知识库回复"
    assert knowledge.get_json()["knowledge_used"] is True
