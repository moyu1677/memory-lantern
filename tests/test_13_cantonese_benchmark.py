"""粤语公开数据集基准的规范化与汇总回归测试。"""

from tests.benchmarks.benchmark_common import character_error_rate, normalize_chinese
from tests.benchmarks.benchmark_14_summarize_cantonese_fleurs import summarize


def test_traditional_and_simplified_characters_share_one_cer_space():
    traditional = "兩週內美軍轉向德國"
    simplified = "两周内美军转向德国"
    assert normalize_chinese(traditional) == simplified
    assert character_error_rate(traditional, simplified) == 0


def test_cantonese_summary_uses_normalized_cer():
    rows = [
        {
            "success": "1",
            "category": "medium",
            "normalized_cer": "0",
            "latency_s": "2.0",
        },
        {
            "success": "1",
            "category": "medium",
            "normalized_cer": "0.1",
            "latency_s": "3.0",
        },
    ]
    overall = summarize(rows)["overall"]
    assert overall["success_rate_percent"] == 100.0
    assert overall["character_accuracy_percent"] == 95.0
    assert overall["normalized_cer_percent"] == 5.0
