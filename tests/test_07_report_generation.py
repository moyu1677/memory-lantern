"""论文表3：专属报告的内容正确性与访问接口测试。"""

import json
from time import perf_counter

from tests.helpers import create_assessment


def test_report_contains_score_advice_and_disclaimer(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    assessment_id = create_assessment(
        app_module,
        elder["id"],
        score=14,
        percentage=100,
    )
    login_as(client, elder["id"])

    response = client.get(f"/history/{assessment_id}/report")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "专属" in body
    assert "100" in body
    assert "能力维度概览" in body
    assert "不能用于疾病诊断、治疗或替代医生意见" in body


def test_report_rules_cover_high_middle_and_low_results(
    app_module,
    user_factory,
):
    elder = user_factory(account_type="elder")

    with app_module.app.app_context():
        maximums = app_module.domain_max_points()
        for index, percentage in enumerate((90, 70, 40), start=1):
            assessment = {
                "id": index,
                "percentage": percentage,
                "score": round(14 * percentage / 100),
                "maximum_score": 14,
                "result_level": app_module.result_level(percentage),
            }
            scores = {
                domain: round(maximum * percentage / 100)
                for domain, maximum in maximums.items()
            }
            report = app_module.build_assessment_report(
                assessment,
                scores,
                maximums,
            )
            assert report["overall"]
            assert report["training_cards"]
            assert report["weekly_plan"]
            assert report["health_actions"]


def test_report_endpoint_responds_repeatedly(
    app_module,
    client,
    user_factory,
    login_as,
):
    """快速本地回归；论文正式时延应使用 benchmarks/benchmark_report.py。"""
    elder = user_factory(account_type="elder")
    assessment_id = create_assessment(
        app_module,
        elder["id"],
        score=9,
        percentage=64,
    )
    login_as(client, elder["id"])

    durations = []
    for _ in range(30):
        started = perf_counter()
        response = client.get(f"/history/{assessment_id}/report")
        durations.append(perf_counter() - started)
        assert response.status_code == 200

    assert len(durations) == 30
    assert all(duration >= 0 for duration in durations)
