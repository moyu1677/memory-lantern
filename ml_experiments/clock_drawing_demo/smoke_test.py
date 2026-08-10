"""不进行正式训练的快速结构与数据自检。"""

from pathlib import Path

import torch

try:
    from .dataset import create_split_manifest, discover_samples, load_image_tensor, manifest_statistics
    from .model import ClockScoreNet, model_summary
except ImportError:  # 兼容在实验目录内直接运行脚本
    from dataset import create_split_manifest, discover_samples, load_image_tensor, manifest_statistics
    from model import ClockScoreNet, model_summary


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "raw" / "clock_shulman"
MANIFEST = ROOT / "data" / "splits" / "split_manifest.csv"


def main():
    samples = discover_samples(DATA_ROOT)
    assert len(samples) == 3108, len(samples)
    rows = create_split_manifest(DATA_ROOT, MANIFEST, seed=20260808)
    statistics = manifest_statistics(rows)
    assert statistics["all"]["total"] == 3108
    tensor = load_image_tensor(samples[0].path, image_size=160)
    assert tensor.shape == (1, 160, 160)
    assert torch.isfinite(tensor).all()
    model = ClockScoreNet()
    output = model(torch.stack([tensor, tensor]))
    assert output.shape == (2, 6)
    print("自检通过：数据3108张、分层划分、图像预处理和模型前向传播均正常。")
    print(model_summary(model))
    print(statistics)


if __name__ == "__main__":
    main()
