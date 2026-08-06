"""使用固定普通话公开数据集调用项目真实 ASR 接口。"""

from benchmark_cloud_asr_common import parse_args, run_language_benchmark


if __name__ == "__main__":
    run_language_benchmark(
        language="mandarin",
        output_filename="asr_mandarin_dataset_results.csv",
        args=parse_args("普通话公开数据集真实识别性能测试"),
    )
