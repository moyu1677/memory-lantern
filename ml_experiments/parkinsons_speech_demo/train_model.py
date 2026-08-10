"""受试者分组的临床语音特征分类与可视化实验。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 10

FEATURE_NAMES = [
    "jitter_local",
    "jitter_local_absolute",
    "jitter_rap",
    "jitter_ppq5",
    "jitter_ddp",
    "shimmer_local",
    "shimmer_local_db",
    "shimmer_apq3",
    "shimmer_apq5",
    "shimmer_apq11",
    "shimmer_dda",
    "autocorrelation",
    "noise_to_harmonics",
    "harmonics_to_noise",
    "median_pitch",
    "mean_pitch",
    "pitch_std",
    "min_pitch",
    "max_pitch",
    "number_of_pulses",
    "number_of_periods",
    "mean_period",
    "period_std",
    "fraction_unvoiced_frames",
    "voice_break_count",
    "degree_voice_breaks",
]

FEATURE_LABELS = {
    "jitter_local": "Jitter (local)",
    "jitter_local_absolute": "Jitter (absolute)",
    "jitter_rap": "Jitter (RAP)",
    "jitter_ppq5": "Jitter (PPQ5)",
    "jitter_ddp": "Jitter (DDP)",
    "shimmer_local": "Shimmer (local)",
    "shimmer_local_db": "Shimmer (dB)",
    "shimmer_apq3": "Shimmer (APQ3)",
    "shimmer_apq5": "Shimmer (APQ5)",
    "shimmer_apq11": "Shimmer (APQ11)",
    "shimmer_dda": "Shimmer (DDA)",
    "autocorrelation": "Autocorrelation",
    "noise_to_harmonics": "Noise/Harmonics",
    "harmonics_to_noise": "Harmonics/Noise",
    "median_pitch": "Median pitch",
    "mean_pitch": "Mean pitch",
    "pitch_std": "Pitch SD",
    "min_pitch": "Minimum pitch",
    "max_pitch": "Maximum pitch",
    "number_of_pulses": "Number of pulses",
    "number_of_periods": "Number of periods",
    "mean_period": "Mean period",
    "period_std": "Period SD",
    "fraction_unvoiced_frames": "Unvoiced-frame fraction",
    "voice_break_count": "Voice-break count",
    "degree_voice_breaks": "Degree of voice breaks",
}

TRAIN_COLUMNS = ["subject_id", *FEATURE_NAMES, "updrs", "target"]
TEST_COLUMNS = ["subject_id", *FEATURE_NAMES, "target"]

MODEL_FACTORIES: dict[str, Callable[[int], BaseEstimator]] = {
    "Logistic Regression": lambda seed: Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    ),
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    ),
    "Gradient Boosting": lambda seed: GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        random_state=seed,
    ),
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train_data.txt"
    test_path = DATA_DIR / "test_data.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "未找到数据文件。请先运行 python download_data.py 或 python run_experiment.py。"
        )

    train = pd.read_csv(train_path, header=None, names=TRAIN_COLUMNS)
    test = pd.read_csv(test_path, header=None, names=TEST_COLUMNS)
    train["subject_id"] = train["subject_id"].astype(int)
    test["subject_id"] = test["subject_id"].astype(int)
    train["target"] = train["target"].astype(int)
    test["target"] = test["target"].astype(int)

    if train[FEATURE_NAMES].isna().any().any():
        raise ValueError("训练数据存在缺失的语音特征。")
    if train.groupby("subject_id")["target"].nunique().max() != 1:
        raise ValueError("同一受试者出现了不同标签，无法进行受试者级评估。")
    return train, test


def _subject_predictions(
    subject_ids: pd.Series,
    targets: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_id": subject_ids.to_numpy(),
            "target": targets.to_numpy(),
            "probability": probabilities,
        }
    )
    return (
        frame.groupby("subject_id", as_index=False)
        .agg(target=("target", "first"), probability=("probability", "mean"))
        .sort_values("subject_id")
    )


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(target, prediction),
        "balanced_accuracy": balanced_accuracy_score(target, prediction),
        "precision": precision_score(target, prediction, zero_division=0),
        "recall_sensitivity": recall_score(target, prediction, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(target, prediction, zero_division=0),
        "roc_auc": roc_auc_score(target, probability),
    }


def repeated_group_cv(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = train[FEATURE_NAMES]
    y = train["target"]
    groups = train["subject_id"]
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for repeat in range(N_REPEATS):
        seed = RANDOM_SEED + repeat
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        fold_subject_predictions: list[pd.DataFrame] = []

        for model_name, factory in MODEL_FACTORIES.items():
            fold_subject_predictions.clear()
            for fold, (train_index, valid_index) in enumerate(
                splitter.split(x, y, groups), start=1
            ):
                model = factory(seed + fold)
                model.fit(x.iloc[train_index], y.iloc[train_index])
                probability = model.predict_proba(x.iloc[valid_index])[:, 1]
                subject_frame = _subject_predictions(
                    groups.iloc[valid_index],
                    y.iloc[valid_index],
                    probability,
                )
                subject_frame["fold"] = fold
                fold_subject_predictions.append(subject_frame)

            repeat_predictions = pd.concat(fold_subject_predictions, ignore_index=True)
            if repeat_predictions["subject_id"].duplicated().any():
                raise RuntimeError("交叉验证中同一受试者被重复预测。")
            scores = _metrics(
                repeat_predictions["target"].to_numpy(),
                repeat_predictions["probability"].to_numpy(),
            )
            metric_rows.append({"model": model_name, "repeat": repeat, **scores})
            repeat_predictions.insert(0, "repeat", repeat)
            repeat_predictions.insert(0, "model", model_name)
            prediction_rows.append(repeat_predictions)

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    ]
    summary = repeat_metrics.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("roc_auc_mean", ascending=False)


def feature_importance(model: BaseEstimator) -> pd.DataFrame:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)[0])
    else:
        raise TypeError("当前模型不支持直接提取特征重要性。")
    return pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "label": [FEATURE_LABELS[name] for name in FEATURE_NAMES],
            "importance": values,
        }
    ).sort_values("importance", ascending=False)


def save_visualizations(
    summary: pd.DataFrame,
    best_predictions: pd.DataFrame,
    importance: pd.DataFrame,
    positive_only: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    green = "#39746d"
    blue = "#4169a9"
    orange = "#d18145"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(summary))
    width = 0.34
    ax.bar(
        positions - width / 2,
        summary["roc_auc_mean"],
        width,
        yerr=summary["roc_auc_std"],
        label="ROC-AUC",
        color=green,
        capsize=4,
    )
    ax.bar(
        positions + width / 2,
        summary["f1_mean"],
        width,
        yerr=summary["f1_std"],
        label="F1",
        color=blue,
        capsize=4,
    )
    ax.set_xticks(positions, summary["model"], rotation=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Subject-level score")
    ax.set_title("Repeated Grouped Cross-validation: Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    target = best_predictions["target"].to_numpy()
    probability = best_predictions["probability"].to_numpy()
    prediction = (probability >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(target, prediction, labels=[0, 1]),
        display_labels=["Healthy", "Parkinson's"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Subject-level Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
    auc = roc_auc_score(target, probability)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.plot(false_positive_rate, true_positive_rate, color=green, linewidth=3, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Subject-level ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["label"], top["importance"], color=orange)
    ax.set_xlabel("Model importance")
    ax.set_title("Top 12 Voice Features in the Final Model")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.hist(positive_only["probability"], bins=10, color=blue, edgecolor="white")
    ax.axvline(0.5, color=orange, linestyle="--", linewidth=2, label="Decision threshold = 0.5")
    ax.set_xlabel("Predicted Parkinson's probability")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Official Positive-only Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "positive_only_probabilities.png", dpi=180)
    plt.close(fig)


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    summary: pd.DataFrame,
    best_name: str,
    best_predictions: pd.DataFrame,
    positive_only: pd.DataFrame,
) -> None:
    best_summary = summary.loc[summary["model"] == best_name].iloc[0]
    oof_scores = _metrics(
        best_predictions["target"].to_numpy(),
        best_predictions["probability"].to_numpy(),
    )
    positive_sensitivity = float((positive_only["probability"] >= 0.5).mean())

    report = f"""# UCI帕金森语音特征分类实验结果

## 实验范围

本实验使用UCI公开的帕金森语音特征数据验证机器学习分类流程。
该数据集不是认知障碍数据，也不是普通话数据，因此实验结果不能解释为认知障碍筛查准确率或医学诊断能力。

## 数据与验证设计

- 训练记录：{len(train)}条；
- 训练受试者：{train['subject_id'].nunique()}人；
- 健康受试者：{train.loc[train['target'] == 0, 'subject_id'].nunique()}人；
- 帕金森受试者：{train.loc[train['target'] == 1, 'subject_id'].nunique()}人；
- 每名训练受试者录音记录数：{int(train.groupby('subject_id').size().median())}条（中位数）；
- 验证方式：{N_REPEATS}次重复、每次{N_SPLITS}折的分层受试者分组交叉验证；
- 官方补充测试记录：{len(test)}条，{test['subject_id'].nunique()}名受试者；
- 官方补充测试集仅含阳性患者，因此不计算准确率、特异度或ROC-AUC。

## 模型比较

以重复交叉验证的平均受试者级ROC-AUC选择最佳模型：**{best_name}**。

- 平均ROC-AUC：{best_summary['roc_auc_mean']:.3f} ± {best_summary['roc_auc_std']:.3f}
- 平均F1：{best_summary['f1_mean']:.3f} ± {best_summary['f1_std']:.3f}
- 平均灵敏度：{best_summary['recall_sensitivity_mean']:.3f} ± {best_summary['recall_sensitivity_std']:.3f}
- 平均特异度：{best_summary['specificity_mean']:.3f} ± {best_summary['specificity_std']:.3f}
- 平均平衡准确率：{best_summary['balanced_accuracy_mean']:.3f} ± {best_summary['balanced_accuracy_std']:.3f}

固定随机种子下最佳模型的折外预测结果：

- ROC-AUC：{oof_scores['roc_auc']:.3f}
- F1：{oof_scores['f1']:.3f}
- 灵敏度：{oof_scores['recall_sensitivity']:.3f}
- 特异度：{oof_scores['specificity']:.3f}
- 平衡准确率：{oof_scores['balanced_accuracy']:.3f}

官方阳性测试集中，阈值0.5下的阳性检出率为：**{positive_sensitivity:.1%}**。
由于该测试集没有健康对照，这一数值只是补充的灵敏度检查，不能称为测试准确率。

## 结果解释限制

1. 样本只有40名训练受试者，指标不确定性较大；
2. 数据研究的是帕金森病，不能外推到认知障碍；
3. 数据提供的是预提取语音特征，不包含本项目老人的原始普通话回答；
4. 模型输出只能用于教学和技术原型展示；
5. 后续接入认知语音数据时，应保留按受试者分组的验证设计，并重新训练和校准模型。
"""
    (OUTPUT_DIR / "result_report.md").write_text(report, encoding="utf-8")


def run_experiment() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    repeat_metrics, all_predictions = repeated_group_cv(train)
    summary = summarize_metrics(repeat_metrics)
    best_name = str(summary.iloc[0]["model"])

    best_predictions = (
        all_predictions.loc[
            (all_predictions["model"] == best_name) & (all_predictions["repeat"] == 0)
        ]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    final_model = MODEL_FACTORIES[best_name](RANDOM_SEED)
    final_model.fit(train[FEATURE_NAMES], train["target"])
    importance = feature_importance(final_model)

    test_probability = final_model.predict_proba(test[FEATURE_NAMES])[:, 1]
    positive_only = _subject_predictions(
        test["subject_id"],
        test["target"],
        test_probability,
    )
    positive_only["prediction"] = (positive_only["probability"] >= 0.5).astype(int)

    repeat_metrics.to_csv(OUTPUT_DIR / "cv_repeat_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    best_predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    positive_only.to_csv(OUTPUT_DIR / "positive_only_test_predictions.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    joblib.dump(final_model, OUTPUT_DIR / "best_model.joblib")

    best_summary = summary.loc[summary["model"] == best_name].iloc[0]
    metadata: dict[str, object] = {
        "experiment": "UCI Parkinson's speech feature classification prototype",
        "best_model": best_name,
        "selection_metric": "mean subject-level ROC-AUC",
        "n_training_records": int(len(train)),
        "n_training_subjects": int(train["subject_id"].nunique()),
        "n_positive_only_test_subjects": int(test["subject_id"].nunique()),
        "cv_splits": N_SPLITS,
        "cv_repeats": N_REPEATS,
        "best_model_mean_metrics": {
            key.removesuffix("_mean"): float(value)
            for key, value in best_summary.items()
            if key.endswith("_mean")
        },
        "positive_only_detection_rate": float(
            (positive_only["probability"] >= 0.5).mean()
        ),
        "warning": "This is not a cognitive impairment diagnostic model.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_visualizations(summary, best_predictions, importance, positive_only)
    write_report(train, test, summary, best_name, best_predictions, positive_only)

    print("\n模型比较（受试者级重复分组交叉验证）：")
    print(
        summary[
            ["model", "roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std"]
        ].to_string(index=False)
    )
    print(f"\n最佳模型：{best_name}")
    print(f"结果目录：{OUTPUT_DIR}")
    return metadata


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
