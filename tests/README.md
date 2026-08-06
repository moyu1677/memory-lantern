# 忆路明灯测试与论文数据采集

本目录用于生成论文“功能与性能测试”及“多方言语音识别测试”的原始数据。
所有自动化功能测试使用临时 SQLite 数据库，不会修改项目的正式用户、留言和
测试历史。

测试数据库会临时创建在 `tests/.runtime/`，测试结束后自动删除。这样既可以在
命令行运行，也可以直接使用 PyCharm 的 pytest 运行按钮，不依赖系统 Temp
目录权限。

## 1. 安装测试依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r tests/requirements-test.txt
python -m playwright install chromium
```

## 2. 自动功能与权限测试

先执行一次快速回归：

```powershell
python -m pytest -c tests/pytest.ini
```

论文要求同一版本每例重复 3 次时执行：

```powershell
python -m pytest -c tests/pytest.ini --count=3 `
  --junitxml=tests/results/functional_results.xml
```

测试结束后会生成：

```text
tests/results/functional_results.xml
tests/results/permission_results.csv
```

功能测试会 mock DeepSeek 和腾讯云，不产生 API 费用。不要用这些 mock 数据填写
真实语音时延。

## 3. 页面首屏加载

先启动项目或准备好线上地址，再执行：

```powershell
python tests/benchmarks/benchmark_03_page_first_load.py `
  --base-url http://127.0.0.1:5000 `
  --username 论文测试账号 `
  --password Test12345 `
  --runs 50
```

脚本使用全新的浏览器上下文、禁用缓存，并记录 FCP 与页面导航时延。测试前应
固定设备、浏览器、网络、服务器和代码版本；本地数据与线上数据不能混合统计。

## 4. 报告生成时延

数据库中先准备一条该账号可访问的测试记录，并找到记录 ID：

```powershell
python tests/benchmarks/benchmark_07_report_generation.py `
  --base-url http://127.0.0.1:5000 `
  --username 论文测试账号 `
  --password Test12345 `
  --assessment-id 1 `
  --runs 100
```

## 5. 粤语与潮汕话真实语音测试

复制并填写样本清单：

```powershell
Copy-Item tests/data/asr_manifest.example.csv tests/data/asr_manifest.csv
```

然后执行：

```powershell
python tests/benchmarks/benchmark_05_cantonese_asr.py `
  --base-url http://127.0.0.1:5000 `
  --username 论文测试账号 `
  --password Test12345 `
  --manifest tests/data/asr_manifest.csv

python tests/benchmarks/benchmark_06_chaoshan_asr.py `
  --base-url http://127.0.0.1:5000 `
  --username 论文测试账号 `
  --password Test12345 `
  --manifest tests/data/asr_manifest.csv
```

该命令会真实调用腾讯云并产生用量。音频必须经过知情同意和去标识化，不得包含
姓名、电话、住址等身份信息。

## 6. 普通话浏览器识别

普通话使用浏览器 Web Speech API，与腾讯云接口不是同一技术路径。启动本地
静态服务器：

```powershell
python -m http.server 8765 --directory tests/benchmarks
```

用 Edge 或 Chrome 打开：

```text
http://127.0.0.1:8765/benchmark_04_mandarin_browser.html
```

逐条填写标准文本、样本编号和环境，点击开始识别，最后下载 CSV。

## 6A. 普通话公开数据集自动测试

这条路径用于做可重复的论文技术测试，不会替换产品页面的浏览器普通话输入。
初次实验建议使用体积较小的 Common Voice 26.0 Beijing Chinese 子集：

```text
https://mozilladatacollective.com/datasets/cmruwsbag00c8md07lk1n7i1e
```

在 Mozilla Data Collective 登录、阅读并接受许可后下载并解压。不要把下载的
音频提交到 Git，也不得尝试识别录音者身份。然后在项目根目录执行：

```powershell
python tests/datasets/prepare_mandarin_commonvoice.py `
  --dataset-root "D:\datasets\common-voice-beijing" `
  --count 100 `
  --seed 20260806
```

脚本会固定抽取 100 条短句、中句、长句及数字/日期句，并统一转换为 16 kHz
单声道 WAV。输出位于：

```text
tests/datasets/mandarin_commonvoice/generated/mandarin_manifest.csv
tests/datasets/mandarin_commonvoice/generated/dataset_summary.json
tests/datasets/mandarin_commonvoice/generated/audio/
```

确保 `.env` 已配置腾讯云语音识别、项目已启动，并准备一个老人端测试账号，然后
执行真实接口测试：

```powershell
python tests/benchmarks/benchmark_11_mandarin_dataset.py `
  --base-url http://127.0.0.1:5000 `
  --username 论文测试账号 `
  --password Test12345 `
  --manifest tests/datasets/mandarin_commonvoice/generated/mandarin_manifest.csv

python tests/benchmarks/benchmark_12_summarize_mandarin_dataset.py
```

生成的 CSV 包含原始 CER、基础规范化 CER、端到端时延以及音频时长、停顿数等
观测字段；汇总脚本会生成 JSON 和 Markdown。该命令真实调用腾讯云并产生用量。
基础规范化只处理宽度、大小写、空白、标点和“〇/零”，不会使用同义词规则替
系统纠错。Common Voice 北京子集以青年录音为主，因此只能验证普通话转写链路，
不能代替老年人真实语音验证，也不能据此声明认知障碍识别效果。

## 7. 汇总表 3

完成上述测试后执行：

```powershell
python tests/benchmarks/benchmark_10_summarize_results.py
```

输出：

```text
tests/results/table3_summary.json
tests/results/table3_summary.md
```

脚本只汇总实际存在的原始结果文件。任何缺少的项目都会标记为“未测试”，不会
自动编造数据。
