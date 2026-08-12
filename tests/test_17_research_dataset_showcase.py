"""普通话与粤语研究数据展示页。"""


def test_research_dataset_page_requires_login(client):
    response = client.get("/research/datasets")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_research_dataset_page_shows_only_mandarin_and_cantonese(
    client,
    user_factory,
    login_as,
):
    user = user_factory(account_type="elder")
    login_as(client, user["id"])

    response = client.get("/research/datasets")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "测试集与验证记录" in html
    assert "Common Voice 北京普通话子集" in html
    assert "远程 Whisper large-v3" in html
    assert "浏览器仅录音/上传" in html
    assert "Google FLEURS 粤语测试集" in html
    assert "待 Whisper V3 复测" in html
    assert "当前没有可展示的有效识别指标" in html
    assert "Teochew-Wild" not in html
    assert "原始录音" in html


def test_dataset_navigation_is_not_shown_on_the_family_care_dashboard(
    client,
    user_factory,
    login_as,
):
    user = user_factory(account_type="family")
    login_as(client, user["id"])
    response = client.get("/dashboard")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'href="/research/datasets"' not in html
    assert "测试数据" not in html
