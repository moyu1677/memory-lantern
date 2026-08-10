"""绘图采集、双人复核和机器辅助评分基础链路。"""

from __future__ import annotations

import base64
import json

from drawing_scoring import extract_features, predict_with_model, train_centroid_model


PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(
    b"\x89PNG\r\n\x1a\nminimal-test-payload"
).decode("ascii")


def sample_strokes(offset=0):
    return [
        {
            "points": [
                {"x": 20 + offset, "y": 20, "t": 0, "p": 0.5},
                {"x": 100 + offset, "y": 20, "t": 200, "p": 0.5},
                {"x": 100 + offset, "y": 100, "t": 400, "p": 0.5},
                {"x": 20 + offset, "y": 100, "t": 600, "p": 0.5},
                {"x": 20 + offset, "y": 20, "t": 800, "p": 0.5},
            ]
        }
    ]


def submission_payload(task_type="clock"):
    return {
        "task_type": task_type,
        "canvas_width": 900,
        "canvas_height": 600,
        "strokes": sample_strokes(),
        "image_png": PNG_DATA_URL,
        "research_consent": True,
    }


def test_feature_extraction_and_model_abstention_are_reproducible():
    low = extract_features(sample_strokes(0), 900, 600)
    high = extract_features(sample_strokes(300), 900, 600)
    samples = []
    for _ in range(5):
        samples.append({"features": low, "score": 0})
        samples.append({"features": {**high, "stroke_count": 5}, "score": 1})
    model = train_centroid_model(samples, "cube", minimum_per_class=5)
    prediction = predict_with_model(low, model, confidence_threshold=0.99)

    assert low["feature_version"] == "drawing-features-v1"
    assert low["closed_stroke_ratio"] == 1.0
    assert model["training_sample_count"] == 10
    assert prediction["score"] in {0, 1}
    assert prediction["status"] in {"predicted", "needs_review"}


def test_drawing_submission_requires_consent_and_does_not_change_assessment_score(
    app_module, client, user_factory, login_as
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    page = client.get("/drawing-lab")
    assert page.status_code == 200
    assert "绘图评分实验" in page.get_data(as_text=True)
    payload = submission_payload()
    payload["research_consent"] = False
    rejected = client.post(
        "/drawing/api/submit",
        json=payload,
        headers={"X-CSRF-Token": token},
    )
    assert rejected.status_code == 422

    accepted = client.post(
        "/drawing/api/submit",
        json=submission_payload(),
        headers={"X-CSRF-Token": token},
    )
    assert accepted.status_code == 201
    assert accepted.get_json()["machine_status"] in {
        "not_available", "needs_review", "predicted"
    }

    with app_module.app.app_context():
        drawing = app_module.database.get_drawing_submission(
            accepted.get_json()["submission_id"]
        )
        assert drawing["research_consent"] == 1
        assert json.loads(drawing["features_json"])["point_count"] == 5
        assert app_module.database.count_assessments(elder["id"]) == 0


def test_two_raters_and_third_adjudicator_create_anonymous_training_row(
    app_module, client, user_factory, login_as
):
    elder = user_factory(account_type="elder", display_name="不应出现在导出中")
    raters = [
        user_factory(account_type="admin", role="admin", username=f"reviewer_{index}")
        for index in range(3)
    ]
    token = login_as(client, elder["id"])
    submitted = client.post(
        "/drawing/api/submit",
        json=submission_payload("cube"),
        headers={"X-CSRF-Token": token},
    )
    submission_id = submitted.get_json()["submission_id"]

    for reviewer, score in zip(raters[:2], (0, 1)):
        token = login_as(client, reviewer["id"])
        response = client.post(
            f"/admin/drawings/{submission_id}/review",
            data={"_csrf_token": token, "review_role": "rater", "score": score},
        )
        assert response.status_code == 302

    with app_module.app.app_context():
        pending = app_module.database.drawing_review_summary(submission_id)
        assert pending["resolved_score"] is None

    token = login_as(client, raters[2]["id"])
    queue = client.get("/admin/drawings")
    detail = client.get(f"/admin/drawings/{submission_id}/review")
    assert queue.status_code == 200
    assert detail.status_code == 200
    adjudicated = client.post(
        f"/admin/drawings/{submission_id}/review",
        data={
            "_csrf_token": token,
            "review_role": "adjudicator",
            "score": 1,
            "notes": "第三方仲裁",
        },
    )
    assert adjudicated.status_code == 302

    exported = client.get("/admin/drawings/dataset.csv")
    assert exported.status_code == 200
    text = exported.get_data(as_text=True)
    assert "adjudicated" in text
    assert "不应出现在导出中" not in text

    with app_module.app.app_context():
        rows = app_module.database.list_training_drawing_rows("cube")
        assert len(rows) == 1
        assert rows[0]["score"] == 1
