"""Phase-one Chinese/English interface and assessment regression tests."""

from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import random
import re

from memory_lantern.assessment.questions import build_test_questions, score_answer
from tests.helpers import api_headers, create_assessment


class VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        if not self.hidden_depth:
            attributes = dict(attrs)
            for name in ("placeholder", "aria-label", "title"):
                if attributes.get(name):
                    self.parts.append(attributes[name])

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if not self.hidden_depth and data.strip():
            self.parts.append(data.strip())


def visible_text(html):
    parser = VisibleTextParser()
    parser.feed(html)
    return " ".join(parser.parts)


def assert_no_unexpected_chinese(html):
    content = visible_text(html).replace("切换到中文", "").replace("中文", "")
    chinese = "".join(character for character in content if "\u4e00" <= character <= "\u9fff")
    assert not chinese, f"Unexpected Chinese UI text: {content}"


def switch_to_english(client, token, next_path="/login"):
    return client.post(
        "/language/en",
        data={"_csrf_token": token, "next": next_path},
        follow_redirects=True,
    )


def test_dashboard_language_button_submits_the_rendered_csrf_token(
    client,
    user_factory,
    login_as,
):
    elder = user_factory(display_name="Button Test", account_type="elder")
    login_as(client, elder["id"])
    dashboard = client.get("/dashboard").get_data(as_text=True)

    form = re.search(
        r'<form class="dashboard-language-form".*?</form>',
        dashboard,
        flags=re.DOTALL,
    )
    assert form is not None
    token = re.search(r'name="_csrf_token" value="([^"]+)"', form.group(0))
    assert token is not None

    response = client.post(
        "/language/en",
        data={"_csrf_token": token.group(1), "next": "/dashboard"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b'lang="en"' in response.data
    assert b"Start cognitive assessment" in response.data
    assert "页面验证已失效" not in response.get_data(as_text=True)


def test_public_language_switch_persists_on_login(client, csrf_for_anonymous):
    token = csrf_for_anonymous(client)
    response = switch_to_english(client, token)

    assert response.status_code == 200
    assert b'lang="en"' in response.data
    assert b"Welcome back" in response.data
    assert b"Memory Lantern" in response.data
    assert "中文" in response.get_data(as_text=True)
    assert_no_unexpected_chinese(response.get_data(as_text=True))

    second_response = client.get("/register")
    assert b'lang="en"' in second_response.data
    assert b"Create an account" in second_response.data
    assert_no_unexpected_chinese(second_response.get_data(as_text=True))

    chinese_response = client.post(
        "/language/zh",
        data={"_csrf_token": token, "next": "/login"},
        follow_redirects=True,
    )
    assert b'lang="zh-CN"' in chinese_response.data
    assert "欢迎回来" in chinese_response.get_data(as_text=True)


def test_dashboard_history_and_report_render_in_english(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(display_name="Alex", account_type="elder")
    assessment_id = create_assessment(app_module, elder["id"], score=14, percentage=100)
    token = login_as(client, elder["id"])

    switch_to_english(client, token, "/dashboard")
    dashboard = client.get("/dashboard").get_data(as_text=True)
    history = client.get("/history").get_data(as_text=True)
    detail = client.get(f"/history/{assessment_id}").get_data(as_text=True)
    report = client.get(f"/history/{assessment_id}/report").get_data(as_text=True)

    assert "Start cognitive assessment" in dashboard
    assert "Important notice" in dashboard
    assert 'action="/language/zh"' in dashboard
    assert ">中文<" in dashboard
    assert "Alex's assessment history" in history
    assert "Personalised support report" in report
    assert "Everyday brain-health guidance" in report
    assert "一次练习5至10分钟" not in report
    assert_no_unexpected_chinese(dashboard)
    assert_no_unexpected_chinese(history)
    assert_no_unexpected_chinese(detail)
    assert_no_unexpected_chinese(report)

    assistant = client.get("/assistant").get_data(as_text=True)
    assert "English (browser)" in assistant
    assert "Start cognitive assessment" in assistant
    assert_no_unexpected_chinese(assistant)


def test_english_question_bank_accepts_english_answers():
    questions = build_test_questions(random.Random(20260808), language="en")
    assert len(questions) == 11
    assert all(question.get("domain_label") for question in questions)
    assert all(
        not any("\u4e00" <= character <= "\u9fff" for character in question["prompt"])
        for question in questions
    )

    now = datetime(2026, 8, 8)
    for question in questions:
        if question["type"] == "drawing_clock":
            continue
        if question["type"] == "keywords":
            answer = ", ".join(aliases[0] for aliases in question["concepts"])
        elif question["type"] == "year":
            answer = "2026"
        elif question["type"] == "month":
            answer = "August"
        elif question["type"] == "day":
            answer = "8"
        elif question["type"] == "weekday":
            answer = "Saturday"
        else:
            answer = question["answers"][0]
        assert score_answer(question, answer, now=now) == question["points"]


def test_english_assessment_api_returns_english_question(
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    with client.session_transaction() as session:
        session["ui_language"] = "en"

    response = client.post(
        "/test/start",
        json={"input_language": "mandarin"},
        headers=api_headers(token),
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["testing"] is True
    assert "Task 1 of 11" in payload["reply"]
    assert payload["question"]["domain"] == "Immediate memory"
    assert not any(
        "\u4e00" <= character <= "\u9fff"
        for character in payload["question"]["prompt"]
    )


def test_english_companion_chat_requests_an_english_reply(
    app_module,
    client,
    user_factory,
    login_as,
    monkeypatch,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    with client.session_transaction() as session:
        session["ui_language"] = "en"

    captured = {}

    def fake_reply(message, system_prompt):
        captured["message"] = message
        captured["system_prompt"] = system_prompt
        return "It is lovely to hear from you."

    monkeypatch.setattr(app_module, "get_ai_reply", fake_reply)
    response = client.post(
        "/chat",
        json={"message": "Hello", "use_knowledge": False},
        headers=api_headers(token),
    )

    assert response.status_code == 200
    assert response.get_json()["reply"] == "It is lovely to hear from you."
    assert "easy-to-understand English" in captured["system_prompt"]
