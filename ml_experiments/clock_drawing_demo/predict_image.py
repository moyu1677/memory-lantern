"""用已训练模型对单张画钟图进行研究型六级评分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .dataset import CLASS_NAMES, load_image_tensor
    from .model import ClockScoreNet
except ImportError:  # 兼容在实验目录内直接运行脚本
    from dataset import CLASS_NAMES, load_image_tensor
    from model import ClockScoreNet


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="预测单张画钟图的Shulman 0～5级分数")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "best_model.pt")
    parser.add_argument("--review-threshold", type=float, default=0.75)
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.model, map_location="cpu", weights_only=False)
    model = ClockScoreNet(num_classes=6)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    inputs = load_image_tensor(arguments.image, image_size=int(checkpoint["image_size"])).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    prediction = int(probabilities.argmax())
    confidence = float(probabilities[prediction])
    result = {
        "image": str(arguments.image.resolve()),
        "score": prediction,
        "class_name": CLASS_NAMES[prediction],
        "confidence": round(confidence, 6),
        "status": "predicted" if confidence >= arguments.review_threshold else "needs_review",
        "probabilities": {
            str(score): round(float(probabilities[score]), 6) for score in range(6)
        },
        "warning": "研究型Shulman六级评分，不是MoCA三分项评分，不用于医学诊断。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
