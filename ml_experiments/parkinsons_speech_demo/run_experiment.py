"""一键下载数据、训练模型并生成可视化。"""

from download_data import ensure_dataset
from train_model import run_experiment


def main() -> None:
    ensure_dataset()
    run_experiment()


if __name__ == "__main__":
    main()
