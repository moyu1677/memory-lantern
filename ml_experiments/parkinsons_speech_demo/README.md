# UCI 帕金森语音特征分类实验

这是一个与“忆路明灯”网站完全独立的机器学习原型实验，用于验证：

1. 公开临床语音特征数据的读取与清洗；
2. 按受试者分组的模型训练和交叉验证；
3. 分类概率、ROC、混淆矩阵和特征重要性可视化；
4. 可复现的指标与中文实验报告输出。

> **边界说明**：该数据集研究的是帕金森病语音特征，不是认知障碍或普通话语音。
> 本实验只能证明语音特征分类与机器学习流程能够运行，不能将结果解释为认知障碍识别准确率，也不能用于医学诊断。

## 数据来源

- UCI Machine Learning Repository：Parkinson's Speech with Multiple Types of Sound Recordings
- 数据集 DOI：<https://doi.org/10.24432/C5NC8M>
- 数据页面：<https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings>
- 许可：CC BY 4.0

训练文件包含40名受试者，每人26条语音记录；测试文件只包含28名帕金森患者，没有健康对照。
因此，本项目使用训练文件进行受试者级交叉验证，并将官方测试文件仅用于补充报告阳性检出率。

## 为什么必须“按受试者分组”

同一受试者的26条录音具有很强的相关性。如果把每一行随机拆分，同一个人的录音会同时进入训练集和测试集，产生数据泄漏和虚高结果。

本实验采用：

- `StratifiedGroupKFold`；
- 同一受试者的全部录音只出现在一个折中；
- 先预测每条录音，再对同一受试者的概率取平均；
- 最终在受试者级别计算指标。

## 快速运行

在当前目录执行：

```powershell
python -m pip install -r requirements.txt
python run_experiment.py
```

第一次运行会从UCI官网下载约92 KB的压缩包。Windows 10/11自带的`tar`可以解压其中的RAR文件。

如果数据已经下载，后续运行可跳过下载：

```powershell
python train_model.py
```

## 输出文件

训练完成后，`outputs/`中会生成：

- `metrics_summary.csv`：三个模型的平均指标；
- `cv_repeat_metrics.csv`：每次重复交叉验证的详细指标；
- `oof_predictions.csv`：最佳模型的受试者级折外预测；
- `positive_only_test_predictions.csv`：官方阳性测试集预测；
- `metrics.json`：便于程序读取的结果；
- `result_report.md`：中文结果说明；
- `model_comparison.png`：模型比较图；
- `confusion_matrix.png`：受试者级混淆矩阵；
- `roc_curve.png`：ROC曲线；
- `feature_importance.png`：特征重要性；
- `positive_only_probabilities.png`：官方阳性测试集概率分布；
- `best_model.joblib`：使用全部训练数据拟合的演示模型。

## 论文中可以怎样描述

建议表述为：

> 为验证语音特征分类模块与机器学习训练流程，采用UCI公开神经系统疾病语音特征数据进行了原型实验。实验按受试者划分训练与验证数据，以避免同一参与者多次录音造成的数据泄漏。该实验仅验证技术流程，不代表系统具备认知障碍诊断效能。
