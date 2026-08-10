"""在 Flask 应用中安全、惰性地调用画钟实验模型。

模型使用公开画钟图片的 Shulman 0～5 标签训练，不等同于 MoCA
钟表题的 0～3 分规则，也不能用于疾病诊断。应用仅展示辅助预测，并把
低置信度或无法推理的样本交给人工复核。
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "ml_experiments"
    / "clock_drawing_demo"
    / "outputs"
    / "best_model.pt"
)

SCORE_LABELS = {
    0: "未形成可辨认钟表",
    1: "存在明显视空间错误",
    2: "存在中度视空间或指针错误",
    3: "存在指针或时间错误",
    4: "存在轻微视空间错误",
    5: "接近完整钟表",
}


def _unavailable(reason, *, status="not_available"):
    return {
        "score": None,
        "confidence": None,
        "status": status,
        "model_version": None,
        "source_label_system": "Shulman 0-5 research baseline",
        "score_label": None,
        "probabilities": {},
        "reason": reason,
    }


@lru_cache(maxsize=4)
def _load_runtime(checkpoint_text):
    """只在首次画钟提交时加载大依赖与权重，避免拖慢普通聊天。"""
    import torch

    from ml_experiments.clock_drawing_demo.model import ClockScoreNet

    checkpoint_path = Path(checkpoint_text).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("未找到画钟模型权重")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = ClockScoreNet(num_classes=6)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        "image_size": int(checkpoint.get("image_size", 160)),
        "model_version": (
            f"{checkpoint.get('model_name', 'ClockScoreNet')}"
            f"-epoch{checkpoint.get('epoch', 'unknown')}"
        ),
        "source_label_system": checkpoint.get(
            "source_label_system", "Shulman 0-5"
        ),
    }
    return torch, model, metadata


def score_clock_image(image_png, checkpoint_path=None, confidence_threshold=0.75):
    """返回六级实验预测；任何加载/图片异常都转为人工复核状态。"""
    path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    if not path.is_file():
        return _unavailable("模型权重尚未安装")

    try:
        from PIL import Image
        from ml_experiments.clock_drawing_demo.dataset import (
            crop_and_pad,
            image_to_tensor,
        )
        torch, model, metadata = _load_runtime(str(path))
        with Image.open(BytesIO(image_png)) as image:
            prepared = crop_and_pad(
                image.convert("RGB"), image_size=metadata["image_size"]
            )
        tensor = image_to_tensor(prepared).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        score = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[score].item())
        return {
            "score": score,
            "confidence": round(confidence, 4),
            "status": (
                "predicted"
                if confidence >= float(confidence_threshold)
                else "needs_review"
            ),
            "model_version": metadata["model_version"],
            "source_label_system": metadata["source_label_system"],
            "score_label": SCORE_LABELS[score],
            "probabilities": {
                str(index): round(float(value), 4)
                for index, value in enumerate(probabilities.tolist())
            },
            "reason": (
                ""
                if confidence >= float(confidence_threshold)
                else "模型置信度不足，必须人工复核"
            ),
        }
    except (ImportError, OSError, RuntimeError, KeyError, ValueError) as error:
        return _unavailable(
            f"模型暂时无法完成推理：{type(error).__name__}",
            status="needs_review",
        )
