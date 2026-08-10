"""绘图任务的可解释特征、模型训练与机器辅助评分。

本模块只负责研究型辅助评分。最终量表分数必须由已授权的量表规则和
经过培训的评分人员确认，模型低置信度时必须拒绝自动给分。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


FEATURE_VERSION = "drawing-features-v1"
MODEL_SCHEMA_VERSION = "nearest-centroid-v1"
FEATURE_NAMES = (
    "stroke_count",
    "point_count",
    "duration_s",
    "ink_length_norm",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "coverage_ratio",
    "closed_stroke_ratio",
    "pen_lift_rate",
)

TASKS = {
    "trail": {"label": "交替连线", "maximum_score": 1},
    "cube": {"label": "立方体临摹", "maximum_score": 1},
    "clock": {"label": "钟表绘制", "maximum_score": 3},
}


class DrawingScoringError(ValueError):
    """绘图数据或模型格式不正确。"""


def _finite_number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_strokes(strokes, width, height):
    """清洗浏览器轨迹，拒绝超量数据并限制坐标范围。"""
    width = int(width)
    height = int(height)
    if width < 200 or height < 200 or width > 2400 or height > 2400:
        raise DrawingScoringError("画布尺寸不在允许范围内。")
    if not isinstance(strokes, list) or not strokes:
        raise DrawingScoringError("请先在画布上完成绘图。")
    if len(strokes) > 1000:
        raise DrawingScoringError("笔画数量过多，请清空后重新绘制。")

    cleaned = []
    total_points = 0
    for stroke in strokes:
        points = stroke.get("points") if isinstance(stroke, dict) else None
        if not isinstance(points, list) or not points:
            continue
        clean_points = []
        for point in points:
            if not isinstance(point, dict):
                continue
            x = min(width, max(0.0, _finite_number(point.get("x"))))
            y = min(height, max(0.0, _finite_number(point.get("y"))))
            timestamp = max(0.0, _finite_number(point.get("t")))
            pressure = min(1.0, max(0.0, _finite_number(point.get("p"), 0.5)))
            clean_points.append(
                {"x": round(x, 2), "y": round(y, 2), "t": round(timestamp, 1), "p": round(pressure, 3)}
            )
        if clean_points:
            cleaned.append({"points": clean_points})
            total_points += len(clean_points)
        if total_points > 20000:
            raise DrawingScoringError("轨迹采样点过多，请清空后重新绘制。")
    if not cleaned:
        raise DrawingScoringError("没有检测到有效绘图轨迹。")
    return cleaned


def extract_features(strokes, width, height):
    """从笔画轨迹提取不含个人身份信息的可解释数值特征。"""
    cleaned = normalize_strokes(strokes, width, height)
    width = float(width)
    height = float(height)
    diagonal = math.hypot(width, height) or 1.0
    all_points = [point for stroke in cleaned for point in stroke["points"]]

    xs = [point["x"] for point in all_points]
    ys = [point["y"] for point in all_points]
    times = [point["t"] for point in all_points]
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)

    total_length = 0.0
    closed_count = 0
    for stroke in cleaned:
        points = stroke["points"]
        for previous, current in zip(points, points[1:]):
            total_length += math.hypot(
                current["x"] - previous["x"],
                current["y"] - previous["y"],
            )
        if len(points) >= 4:
            endpoint_distance = math.hypot(
                points[-1]["x"] - points[0]["x"],
                points[-1]["y"] - points[0]["y"],
            )
            if endpoint_distance <= diagonal * 0.035:
                closed_count += 1

    stroke_count = len(cleaned)
    duration_s = max(0.0, (max(times) - min(times)) / 1000.0)
    features = {
        "stroke_count": float(stroke_count),
        "point_count": float(len(all_points)),
        "duration_s": duration_s,
        "ink_length_norm": total_length / diagonal,
        "bbox_width_ratio": bbox_width / width,
        "bbox_height_ratio": bbox_height / height,
        "coverage_ratio": (bbox_width * bbox_height) / (width * height),
        "closed_stroke_ratio": closed_count / stroke_count,
        "pen_lift_rate": max(0, stroke_count - 1) / max(duration_s, 0.1),
    }
    return {
        "feature_version": FEATURE_VERSION,
        **{name: round(features[name], 6) for name in FEATURE_NAMES},
    }


def train_centroid_model(samples, task_type, minimum_per_class=5):
    """训练小样本可解释基线模型；只接收已形成共识的人工标签。"""
    if task_type not in TASKS:
        raise DrawingScoringError("未知绘图任务。")
    grouped = defaultdict(list)
    for sample in samples:
        score = int(sample["score"])
        if not 0 <= score <= TASKS[task_type]["maximum_score"]:
            continue
        vector = [_finite_number(sample["features"].get(name)) for name in FEATURE_NAMES]
        grouped[score].append(vector)
    if len(grouped) < 2:
        raise DrawingScoringError("至少需要两个得分类别的已复核样本。")
    too_small = [score for score, rows in grouped.items() if len(rows) < minimum_per_class]
    if too_small:
        raise DrawingScoringError(
            "以下得分类别样本不足：" + "、".join(map(str, sorted(too_small)))
        )

    all_vectors = [vector for rows in grouped.values() for vector in rows]
    feature_means = []
    feature_scales = []
    for index in range(len(FEATURE_NAMES)):
        values = [vector[index] for vector in all_vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        feature_means.append(mean)
        feature_scales.append(max(math.sqrt(variance), 1e-6))

    centroids = {}
    class_counts = {}
    for score, vectors in sorted(grouped.items()):
        centroids[str(score)] = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(FEATURE_NAMES))
        ]
        class_counts[str(score)] = len(vectors)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fingerprint = hashlib.sha256(
        json.dumps(
            {"task": task_type, "counts": class_counts, "centroids": centroids},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": f"{task_type}-{trained_at[:10]}-{fingerprint}",
        "task_type": task_type,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_means": feature_means,
        "feature_scales": feature_scales,
        "centroids": centroids,
        "class_counts": class_counts,
        "trained_at": trained_at,
        "training_sample_count": len(all_vectors),
        "minimum_per_class": minimum_per_class,
    }


def predict_with_model(features, model, confidence_threshold=0.75):
    """返回预测和置信度；不足阈值时明确拒绝自动给分。"""
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise DrawingScoringError("模型格式版本不受支持。")
    if model.get("feature_version") != features.get("feature_version"):
        raise DrawingScoringError("模型与特征版本不匹配。")
    feature_names = model.get("feature_names")
    if feature_names != list(FEATURE_NAMES):
        raise DrawingScoringError("模型特征顺序不匹配。")

    vector = [_finite_number(features.get(name)) for name in FEATURE_NAMES]
    scales = [max(_finite_number(value, 1.0), 1e-6) for value in model["feature_scales"]]
    distances = {}
    for score, centroid in model["centroids"].items():
        distances[score] = sum(
            ((value - _finite_number(center)) / scale) ** 2
            for value, center, scale in zip(vector, centroid, scales)
        )
    ordered = sorted(distances.items(), key=lambda item: item[1])
    weights = {score: math.exp(-min(distance, 700.0) / 2.0) for score, distance in ordered}
    total_weight = sum(weights.values()) or 1.0
    best_score = ordered[0][0]
    confidence = weights[best_score] / total_weight
    return {
        "score": int(best_score),
        "confidence": round(confidence, 4),
        "status": "predicted" if confidence >= confidence_threshold else "needs_review",
        "model_version": model["model_version"],
        "distances": {key: round(value, 6) for key, value in distances.items()},
    }


def load_task_model(task_type, model_directory):
    path = Path(model_directory) / f"drawing_{task_type}.json"
    if not path.exists():
        return None
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DrawingScoringError(f"无法读取绘图模型：{error}") from error
    if model.get("task_type") != task_type:
        raise DrawingScoringError("模型任务类型与文件名不匹配。")
    return model


def score_drawing_assist(features, task_type, model_directory, threshold=0.75):
    model = load_task_model(task_type, model_directory)
    if model is None:
        return {
            "score": None,
            "confidence": None,
            "status": "not_available",
            "model_version": None,
            "distances": {},
        }
    return predict_with_model(features, model, confidence_threshold=threshold)
