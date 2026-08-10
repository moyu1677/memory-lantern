# 画钟自动评分独立实验

本目录用于验证“手绘钟表图像 → 机器辅助评分”的训练流程。网站现在会通过根目录
`clock_model_service.py` **惰性读取** `outputs/best_model.pt`；模型缺失、依赖未安装
或置信度不足时会自动转为人工复核，不影响普通聊天和原有14分测试。

## 研究边界

- 当前数据标签为 **Shulman 0～5级整体评分**。
- 当前模型不是 MoCA 的“轮廓、数字、指针”三个独立评分头。
- 当前结果只能说明图像评分流程可运行，不能据此诊断认知障碍。
- 数据源仓库页面没有显示明确的数据许可证。原始图片只保存在本机；重新分发、
  公开模型或商业使用前，应向原作者确认授权。

数据与论文：

- 仓库：<https://github.com/cccnlab/CDT-API-Network>
- 论文 DOI：<https://doi.org/10.1038/s41598-023-44723-1>
- 数据量：3,108张，类别数量为13、20、53、352、1,047、1,623。

## 目录

```text
clock_drawing_demo/
├─ data/
│  ├─ raw/                 # 本机原始数据，不提交Git
│  ├─ splits/              # 固定分层划分，不提交Git
│  └─ dataset_metadata.json
├─ outputs/                # 模型、指标、CSV与图片，不提交Git
├─ dataset.py              # 图片发现、裁剪、增强与分层划分
├─ download_data.py        # 下载、SHA256与类别数量校验
├─ model.py                # 不依赖torchvision的轻量CNN
├─ train_model.py          # 训练、早停、测试与可视化
├─ predict_image.py        # 单图预测
├─ run_experiment.py       # 一键运行
└─ smoke_test.py           # 快速自检
```

## 环境

在项目根目录执行：

```powershell
python -m pip install -r ml_experiments/clock_drawing_demo/requirements.txt
```

本机当前已经具备 CPU 版 PyTorch，可以直接运行。若以后安装与显卡匹配的 CUDA
版 PyTorch，可使用 `--device cuda`。

## 运行

第一次运行（会从作者仓库下载约86MB数据）：

```powershell
python ml_experiments/clock_drawing_demo/run_experiment.py --epochs 25 --device cpu
```

数据已经下载时：

```powershell
python ml_experiments/clock_drawing_demo/run_experiment.py --skip-download --epochs 25 --device cpu
```

快速自检：

```powershell
python ml_experiments/clock_drawing_demo/smoke_test.py
```

单独重建分层划分并训练：

```powershell
python ml_experiments/clock_drawing_demo/train_model.py `
  --epochs 25 `
  --batch-size 64 `
  --device cpu `
  --rebuild-split
```

## 输出

训练完成后查看 `outputs/`：

```text
best_model.pt             最佳验证Macro-F1对应的模型
metrics.json              完整机器可读指标
result_report.md          中文实验报告
training_history.csv      每轮指标
training_history.png      损失、Macro-F1和Kappa曲线
class_distribution.png    类别分布图
confusion_matrix.png      测试集混淆矩阵
test_predictions.csv      每张测试图片的预测与概率
error_examples.png        高置信度错误示例
```

单图预测：

```powershell
python ml_experiments/clock_drawing_demo/predict_image.py "图片路径.png"
```

输出包含预测分数、置信度、六类概率和是否需要人工复核。

## 评估原则

主要指标为 Macro-F1、各类别Recall、二次加权Kappa和平均绝对分数误差。由于0分
与1分只有13和20张图，它们在测试集中只有2和3张，单次划分结果波动会很大；
不能只报告Accuracy，也不能把该数据集上的结果直接外推到浏览器画布或中国老人。

下一阶段应对500～1,000张图片进行三个分项的双人标注，将网络替换为三个二分类
输出头，再用本网站画布采集的匿名绘图做域适配和独立验证。
