# 潮汕话 Teochew-Wild 30条测试

本测试使用腾讯云 `16k_zh_en` 实时语音识别引擎。该引擎已支持潮汕话，
不需要填写一个不存在的“潮汕话专用引擎名”。

## 1. 获取数据

1. 登录 Hugging Face。
2. 打开 <https://huggingface.co/datasets/panlr/teochew_wild>。
3. 阅读并接受数据集的研究使用条件，等待访问权限通过。
4. 下载 `teochew_wild.zip` 和 `annotation.txt`，解压到本机目录。

原始数据、标注和生成的测试音频都不得提交到 Git。项目只提交处理脚本和最终的
汇总统计结果。数据集说明对音频版权和非商业研究用途有额外限制，使用时应采用
其中更严格的条件。

## 2. 固定抽取30条

在项目根目录运行：

```powershell
python tests\datasets\prepare_chaoshan_teochew_wild.py `
  --dataset-root "D:\datasets\teochew_wild" `
  --count 30 `
  --seed 20260808
```

脚本优先覆盖全部20名说话人，然后轮转抽取剩余样本，并统一转换为16 kHz、
单声道、16-bit PCM WAV。输出清单为：

```text
tests/datasets/chaoshan_teochew_wild/generated/chaoshan_manifest.csv
```

## 3. 先测1条

保持 Flask 应用运行，在另一个 PowerShell 窗口执行：

```powershell
python tests\benchmarks\benchmark_06_chaoshan_asr.py `
  --base-url http://127.0.0.1:5000 `
  --username "测试账号" `
  --password "测试密码" `
  --manifest tests\datasets\chaoshan_teochew_wild\generated\chaoshan_manifest.csv `
  --limit 1
```

确认返回 HTTP 200 且 `prediction` 非空后，再移除 `--limit 1` 执行30条正式测试。

## 4. 生成汇总和可视化

```powershell
python tests\benchmarks\benchmark_15_summarize_chaoshan_teochew_wild.py
```

结果写入 `tests/results/`，包括 CSV、JSON、Markdown、错误样本表和PNG仪表板。
报告同时保留严格CER与规范化CER。规范化只统一简繁、标点和空白，不会自动将
同音字改成参考答案。
