# 普通话 Common Voice 本地数据目录

本目录只保存准备工具，公开数据集的音频和元数据不会提交到 Git。请先在
Mozilla Data Collective 页面阅读并接受数据集许可，再下载 Common Voice
Mandarin Chinese（中国大陆）或体积更小的 Beijing 子集。

将解压目录作为 `--dataset-root` 传给准备脚本。目录中通常应包含：

```text
数据集解压目录/
├─ test.tsv
└─ clips/
   ├─ common_voice_zh-CN_....mp3
   └─ ...
```

运行后，本目录的 `generated/` 中会生成固定抽样后的 WAV 音频、测试清单和
数据摘要。它们会被 `.gitignore` 排除；向队友分享代码时不要附带原始音频，
除非数据集许可明确允许，并已完成必要的伦理与隐私审查。
