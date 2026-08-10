"""训练、验证并生成画钟六级评分基线的全部可视化结果。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import (
        CLASS_NAMES,
        ClockDrawingDataset,
        create_split_manifest,
        manifest_statistics,
        read_manifest,
    )
    from .model import ClockScoreNet, model_summary
except ImportError:  # 兼容在实验目录内直接运行脚本
    from dataset import (
        CLASS_NAMES,
        ClockDrawingDataset,
        create_split_manifest,
        manifest_statistics,
        read_manifest,
    )
    from model import ClockScoreNet, model_summary


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data" / "raw" / "clock_shulman"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "split_manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前PyTorch未检测到CUDA，请使用 --device cpu 或安装CUDA版PyTorch。")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def compute_class_weights(rows):
    counts = np.array([
        sum(int(row["label"]) == label for row in rows)
        for label in range(6)
    ], dtype=np.float64)
    weights = np.sqrt(counts.max() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int).tolist()


def run_epoch(model, loader, criterion, device, optimizer=None, ordinal_weight=0.08):
    training = optimizer is not None
    model.train(training)
    losses = []
    labels = []
    predictions = []
    probabilities = []
    paths = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets, batch_paths in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            classification_loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)
            score_axis = torch.arange(6, device=device, dtype=torch.float32)
            expected_scores = (probs * score_axis.unsqueeze(0)).sum(dim=1)
            ordinal_loss = nn.functional.smooth_l1_loss(expected_scores, targets.float())
            loss = classification_loss + ordinal_weight * ordinal_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().tolist())
            paths.extend(batch_paths)

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": accuracy_score(labels, predictions) if labels else math.nan,
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0) if labels else math.nan,
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0) if labels else math.nan,
        "quadratic_kappa": cohen_kappa_score(labels, predictions, weights="quadratic") if labels else math.nan,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "paths": paths,
    }


def save_history(history, output_dir):
    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="Train")
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history], label="Validation")
    axes[1].set_title("Macro-F1")
    axes[2].plot(epochs, [row["validation_kappa"] for row in history], label="Validation")
    axes[2].set_title("Quadratic weighted kappa")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(figure)


def save_distribution(statistics, output_dir):
    labels = np.arange(6)
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, split in enumerate(("train", "validation", "test")):
        values = [statistics[split]["by_class"][str(label)] for label in labels]
        axis.bar(labels + (index - 1) * width, values, width=width, label=split.title())
    axis.set_xticks(labels)
    axis.set_xlabel("Shulman score")
    axis.set_ylabel("Images")
    axis.set_title("Class distribution (note the severe imbalance)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(figure)


def save_confusion(labels, predictions, output_dir):
    matrix = confusion_matrix(labels, predictions, labels=list(range(6)))
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(6):
        for column in range(6):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{normalized[row, column]:.0%}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.55 else "black",
                fontsize=9,
            )
    axis.set_xticks(range(6), labels=range(6))
    axis.set_yticks(range(6), labels=range(6))
    axis.set_xlabel("Predicted score")
    axis.set_ylabel("True score")
    axis.set_title("Test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    return matrix.tolist()


def save_predictions(result, output_dir):
    rows = []
    for path, label, prediction, probabilities in zip(
        result["paths"], result["labels"], result["predictions"], result["probabilities"]
    ):
        row = {
            "image_path": path,
            "true_score": label,
            "predicted_score": prediction,
            "confidence": round(max(probabilities), 6),
            "absolute_error": abs(label - prediction),
        }
        row.update({f"probability_{score}": round(probabilities[score], 6) for score in range(6)})
        rows.append(row)
    path = output_dir / "test_predictions.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_error_gallery(prediction_rows, output_dir, limit=16):
    errors = [row for row in prediction_rows if row["true_score"] != row["predicted_score"]]
    errors.sort(key=lambda row: (row["absolute_error"], row["confidence"]), reverse=True)
    errors = errors[:limit]
    if not errors:
        return None
    tile_size = 220
    caption_height = 42
    columns = 4
    rows_count = math.ceil(len(errors) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows_count * (tile_size + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, row in enumerate(errors):
        with Image.open(row["image_path"]) as source:
            image = ImageOpsContain(source.convert("RGB"), (tile_size - 12, tile_size - 12))
        x = (index % columns) * tile_size + (tile_size - image.width) // 2
        y = (index // columns) * (tile_size + caption_height) + (tile_size - image.height) // 2
        canvas.paste(image, (x, y))
        caption_y = (index // columns) * (tile_size + caption_height) + tile_size
        draw.text(
            ((index % columns) * tile_size + 6, caption_y + 4),
            f"true={row['true_score']} pred={row['predicted_score']} conf={row['confidence']:.2f}",
            fill="black",
            font=font,
        )
    path = output_dir / "error_examples.png"
    canvas.save(path)
    return path


def ImageOpsContain(image, size):
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def build_report(metrics, statistics, model_info, arguments):
    per_class = metrics["per_class"]
    lines = [
        "# 画钟六级评分基线实验结果",
        "",
        "> 本实验只验证图像评分训练流程，不等同于MoCA三分项评分，也不能用于医学诊断。",
        "",
        "## 配置",
        "",
        f"- 设备：`{metrics['device']}`",
        f"- 模型参数量：{model_info['parameters']:,}",
        f"- 训练轮数：{metrics['epochs_trained']}",
        f"- 最佳验证 Macro-F1：{metrics['best_validation_macro_f1']:.4f}",
        f"- 数据划分：训练 {statistics['train']['total']} / 验证 {statistics['validation']['total']} / 测试 {statistics['test']['total']}",
        "",
        "## 测试集结果",
        "",
        f"- Accuracy：{metrics['accuracy']:.4f}",
        f"- Macro-F1：{metrics['macro_f1']:.4f}",
        f"- Weighted-F1：{metrics['weighted_f1']:.4f}",
        f"- Quadratic weighted kappa：{metrics['quadratic_kappa']:.4f}",
        f"- 平均绝对分数误差：{metrics['mean_absolute_score_error']:.4f}",
        "",
        "| 分数 | Precision | Recall | F1 | 测试样本 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in range(6):
        row = per_class[str(label)]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']} |"
        )
    lines.extend([
        "",
        "## 使用限制",
        "",
        "- 原始标签是Shulman 0～5级，不是MoCA的轮廓、数字、指针三个分项。",
        "- 低分样本极少，0分与1分测试集指标方差很大，必须谨慎解释。",
        "- 源数据来自纸质量表扫描图，部署到浏览器画布前还需要本项目真实画布数据微调。",
        "- 源仓库没有展示明确的数据许可证；公开模型或商业使用前应取得作者许可。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="训练轻量画钟六级评分基线")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rebuild-split", action="store_true")
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs 必须大于0")
    set_seed(arguments.seed)
    device = choose_device(arguments.device)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = arguments.data_root.resolve()
    manifest = arguments.manifest.resolve()
    if arguments.rebuild_split or not manifest.exists():
        rows = create_split_manifest(data_root, manifest, seed=arguments.seed)
    else:
        rows = read_manifest(manifest)
    statistics = manifest_statistics(rows)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_distribution(statistics, output_dir)

    train_dataset = ClockDrawingDataset(
        data_root, manifest, "train", image_size=arguments.image_size, augment=True, seed=arguments.seed
    )
    validation_dataset = ClockDrawingDataset(
        data_root, manifest, "validation", image_size=arguments.image_size, augment=False
    )
    test_dataset = ClockDrawingDataset(
        data_root, manifest, "test", image_size=arguments.image_size, augment=False
    )
    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    weights, train_counts = compute_class_weights(read_manifest(manifest, split="train"))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    model = ClockScoreNet(num_classes=6).to(device)
    info = model_summary(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, arguments.epochs))

    print(f"设备：{device}；模型参数：{info['parameters']:,}；训练图片：{len(train_dataset)}")
    history = []
    best_f1 = -1.0
    no_improvement = 0
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, arguments.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_result = run_epoch(model, validation_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "train_macro_f1": train_result["macro_f1"],
            "validation_loss": validation_result["loss"],
            "validation_accuracy": validation_result["accuracy"],
            "validation_macro_f1": validation_result["macro_f1"],
            "validation_kappa": validation_result["quadratic_kappa"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_macro_f1={row['validation_macro_f1']:.4f} "
            f"val_kappa={row['validation_kappa']:.4f}"
        )
        if validation_result["macro_f1"] > best_f1 + 1e-5:
            best_f1 = validation_result["macro_f1"]
            no_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": "ClockScoreNet",
                "model_summary": info,
                "image_size": arguments.image_size,
                "class_names": CLASS_NAMES,
                "best_validation_macro_f1": best_f1,
                "epoch": epoch,
                "seed": arguments.seed,
                "source_label_system": "Shulman 0-5",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, checkpoint_path)
        else:
            no_improvement += 1
        if no_improvement >= arguments.patience:
            print(f"验证集连续{arguments.patience}轮没有改善，提前停止。")
            break

    save_history(history, output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = run_epoch(model, test_loader, criterion, device)
    prediction_rows = save_predictions(test_result, output_dir)
    matrix = save_confusion(test_result["labels"], test_result["predictions"], output_dir)
    save_error_gallery(prediction_rows, output_dir)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_result["labels"], test_result["predictions"], labels=list(range(6)), zero_division=0
    )
    per_class = {
        str(label): {
            "name": CLASS_NAMES[label],
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
            "support": int(support[label]),
        }
        for label in range(6)
    }
    metrics = {
        "device": str(device),
        "epochs_trained": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_validation_macro_f1"]),
        "accuracy": float(test_result["accuracy"]),
        "macro_f1": float(test_result["macro_f1"]),
        "weighted_f1": float(test_result["weighted_f1"]),
        "quadratic_kappa": float(test_result["quadratic_kappa"]),
        "mean_absolute_score_error": float(np.mean([
            abs(truth - prediction)
            for truth, prediction in zip(test_result["labels"], test_result["predictions"])
        ])),
        "class_weights": weights.tolist(),
        "train_class_counts": train_counts,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            test_result["labels"],
            test_result["predictions"],
            labels=list(range(6)),
            target_names=[CLASS_NAMES[label] for label in range(6)],
            zero_division=0,
            output_dict=True,
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "result_report.md").write_text(
        build_report(metrics, statistics, info, arguments), encoding="utf-8"
    )
    print(f"完成。模型与图表已写入：{output_dir}")
    print(
        f"测试集 Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
        f"quadratic kappa={metrics['quadratic_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
