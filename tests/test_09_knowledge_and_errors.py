"""知识库审核边界、异常输入和接口故障处理测试。"""

from io import BytesIO

from rag import retrieve_knowledge
from tests.helpers import api_headers


def test_admin_upload_publish_and_disable_rag_document(
    app_module,
    client,
    user_factory,
    login_as,
):
    admin = user_factory(account_type="admin", role="admin")
    token = login_as(client, admin["id"])
    unique_phrase = "蓝花雨伞记忆训练法"
    content = (
        f"{unique_phrase}仅作为自动化测试资料。"
        "进行训练时应保持环境安静，内容不构成医疗诊断或处方建议。"
    )

    upload = client.post(
        "/admin/rag-documents/upload",
        data={
            "_csrf_token": token,
            "title": "RAG自动化测试资料",
            "category": "自动化测试",
            "audience": "family",
            "source_name": "测试来源",
            "source_url": "https://example.com/test-source",
            "status": "published",
            "file": (
                BytesIO(content.encode("utf-8")),
                "rag-test.txt",
            ),
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 302

    with app_module.app.app_context():
        documents = app_module.database.list_knowledge_documents("published")
        assert len(documents) == 1
        document_id = documents[0]["document_id"]
        family_results = retrieve_knowledge(unique_phrase, "family")
        elder_results = retrieve_knowledge(unique_phrase, "elder")
        assert any(
            item["title"] == "RAG自动化测试资料"
            for item in family_results
        )
        assert all(
            item["title"] != "RAG自动化测试资料"
            for item in elder_results
        )

    disabled = client.post(
        f"/admin/rag-documents/{document_id}/status",
        data={
            "_csrf_token": token,
            "status": "disabled",
            "review_notes": "自动化测试停用",
        },
    )
    assert disabled.status_code == 302
    with app_module.app.app_context():
        assert all(
            item["title"] != "RAG自动化测试资料"
            for item in retrieve_knowledge(unique_phrase, "family")
        )


def test_unsupported_rag_file_is_rejected(
    app_module,
    client,
    user_factory,
    login_as,
):
    admin = user_factory(account_type="admin", role="admin")
    token = login_as(client, admin["id"])

    response = client.post(
        "/admin/rag-documents/upload",
        data={
            "_csrf_token": token,
            "title": "错误格式",
            "file": (BytesIO(b"not supported"), "unsafe.exe"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "只支持TXT" in response.get_data(as_text=True)
    with app_module.app.app_context():
        assert app_module.database.list_knowledge_documents() == []


def test_api_rejects_missing_csrf(
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    login_as(client, elder["id"])
    response = client.post(
        "/chat",
        json={"message": "你好", "use_knowledge": False},
    )
    assert response.status_code == 400
    assert "页面验证已失效" in response.get_json()["error"]


def test_api_rejects_unauthenticated_user_with_valid_csrf(client):
    with client.session_transaction() as session:
        session["_csrf_token"] = "anonymous-token"
    response = client.post(
        "/chat",
        json={"message": "你好", "use_knowledge": False},
        headers=api_headers("anonymous-token"),
    )
    assert response.status_code == 401


def test_speech_endpoint_reports_missing_whisper_configuration(
    app_module,
    client,
    user_factory,
    login_as,
    monkeypatch,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    class UnavailableAsr:
        @staticmethod
        def transcribe(*_args, **_kwargs):
            raise app_module.AsrServiceError("语音识别服务尚未连接。")

    monkeypatch.setattr(app_module, "asr_service", UnavailableAsr())

    response = client.post(
        "/speech/recognize",
        data={"audio": (BytesIO(b"audio"), "sample.webm")},
        content_type="multipart/form-data",
        headers=api_headers(token),
    )
    assert response.status_code == 503
    assert "尚未连接" in response.get_json()["error"]


def test_speech_endpoint_rejects_unknown_language_before_network(
    app_module,
    client,
    user_factory,
    login_as,
    monkeypatch,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    response = client.post(
        "/speech/recognize",
        data={
            "language": "unknown",
            "audio": (BytesIO(b"fake audio"), "sample.webm"),
        },
        content_type="multipart/form-data",
        headers=api_headers(token),
    )
    assert response.status_code == 400
    assert "不支持" in response.get_json()["error"]


def test_mandarin_speech_endpoint_uses_whisper_without_browser_fallback(
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
        "asr_service",
        type("Whisper", (), {"transcribe": staticmethod(lambda *_args, **_kwargs: "您好")})(),
    )

    response = client.post(
        "/speech/recognize",
        data={
            "language": "mandarin",
            "audio": (BytesIO(b"audio"), "sample.webm"),
        },
        content_type="multipart/form-data",
        headers=api_headers(token),
    )

    assert response.status_code == 200
    assert response.get_json()["text"] == "您好"
    assert response.get_json()["language"] == "mandarin"
