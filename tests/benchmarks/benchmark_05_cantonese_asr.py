"""表3和表4：粤语腾讯云端到端时延、成功率与CER。"""

from benchmark_cloud_asr_common import parse_args, run_language_benchmark


if __name__ == "__main__":
    run_language_benchmark(
        "cantonese",
        "asr_cantonese_results.csv",
        parse_args("粤语真实ASR测试"),
    )

