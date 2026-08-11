# AudioMixer 音频混合模块

## 功能概述

AudioMixer负责将TTS生成的多个音频片段按时间戳拼接，并与原视频的BGM混合，生成最终的配音音频。

## 主要功能

### 1. TTS片段拼接
- 根据ASR的时间戳，将TTS生成的音频片段放置在正确的时间位置
- 自动处理片段间的空白
- 支持不连续的音频片段

### 2. 音量控制
- 独立调整人声音量
- 独立调整BGM音量
- 自动归一化防止爆音

### 3. BGM混合
- 支持与原视频BGM混合
- 自动调整BGM长度（循环或裁剪）
- 可单独输出人声（不混合BGM）

## 使用方法

### 基础用法

```python
from src.audio.audio_mixer import AudioMixer

# 初始化
mixer = AudioMixer(sample_rate=24000)

# 准备segments（来自ASR+翻译+TTS）
segments = [
    {
        "start": 0.0,
        "end": 2.0,
        "text": "原文本",
        "text": "ASR 原文文本",
        "tts_audio": "path/to/tts_0000.wav"
    },
    {
        "start": 2.5,
        "end": 4.5,
        "text": "第二句",
        "text": "Second sentence",
        "tts_audio": "path/to/tts_0001.wav"
    }
]

# 混合音频
output = mixer.mix_tts_segments(
    segments=segments,
    tts_output_dir="path/to/tts_outputs",
    output_path="final_audio.wav",
    bgm_path="path/to/bgm.wav",  # 可选
    vocal_volume=0,    # 人声音量调整（dB）
    bgm_volume=-3      # BGM音量调整（dB）
)
```

### 在Pipeline中使用

```python
from src.pipline import CrossLanguageDubbingPipeline

pipeline = CrossLanguageDubbingPipeline()

# Pipeline会自动调用AudioMixer
# 在步骤5中混合音频
result = pipeline.run("video.mp4")
```

## 参数说明

### mix_tts_segments()

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|--------|
| segments | list | ASR+翻译+TTS的segments列表 | 必需 |
| tts_output_dir | str | TTS输出目录 | 必需 |
| output_path | str | 输出音频路径 | 必需 |
| bgm_path | str | BGM音频路径 | None（可选） |
| vocal_volume | float | 人声音量调整（dB） | 0 |
| bgm_volume | float | BGM音量调整（dB） | -3 |

### 音量调整建议

- **vocal_volume**:
  - 0dB: 保持原音量
  - +3dB: 增加一倍响度
  - -3dB: 减半响度

- **bgm_volume**:
  - 建议设置为 -3 到 -6 dB
  - 确保人声清晰，BGM不会盖过人声

## 工作原理

### 步骤1: 创建音频底板
```
根据segments中的最大end时间，创建空白音频
例如：最后一句在10.5秒结束，创建10.5秒的静音
```

### 步骤2: 放置TTS片段
```
将每个TTS片段放置在对应的时间位置：
  0.0s ────┬──── 2.0s
           │ TTS片段1
  2.5s ────┬──── 4.5s
           │ TTS片段2
```

### 步骤3: 调整音量
```
人声 + vocal_volume (dB)
BGM + bgm_volume (dB)
```

### 步骤4: 混合BGM
```
如果提供了BGM：
  1. 加载BGM
  2. 调整BGM长度匹配人声
  3. overlay混合
```

### 步骤5: 归一化
```
防止爆音，归一化到合适音量
```

## 测试

### 运行测试脚本

```bash
cd tests
python test_audio_mixer.py
```

测试内容：
1. 基础混合（仅人声）
2. 混合BGM
3. 音量调整
4. 边缘情况处理

### 测试输出

测试会在 `tests/test_outputs/audio_mixer/` 生成：
- `mixed_vocal_only.wav` - 只有人声
- `mixed_with_bgm.wav` - 人声+BGM
- `volume_*.wav` - 不同音量设置

## 依赖

需要安装：
```bash
pip install pydub soundfile
```

注意：pydub需要ffmpeg支持，请确保系统已安装ffmpeg。

## 高级用法

### 1. 调整音量平衡

```python
# 人声更突出
mixer.mix_tts_segments(
    ...,
    vocal_volume=2,   # 人声提高2dB
    bgm_volume=-6     # BGM降低6dB
)

# BGM更明显
mixer.mix_tts_segments(
    ...,
    vocal_volume=-2,  # 人声降低2dB
    bgm_volume=-1     # BGM只降低1dB
)
```

### 2. 不使用BGM

```python
# 只输出人声
mixer.mix_tts_segments(
    ...,
    bgm_path=None
)
```

### 3. 处理长视频

对于长视频，BGM会自动循环：
```python
# BGM 10秒，视频60秒 → BGM循环6次
mixer.mix_tts_segments(
    ...,
    bgm_path="short_bgm.wav"  # 自动循环
)
```

## 常见问题

### Q1: 音频不同步？
**A**: 检查segments中的时间戳是否正确，确保与原视频匹配。

### Q2: 人声太小/太大？
**A**: 调整 `vocal_volume` 参数，建议范围 -3 到 +3 dB。

### Q3: BGM太吵？
**A**: 降低 `bgm_volume`，建议 -6 到 -3 dB。

### Q4: 输出音频有爆音？
**A**: 模块会自动归一化，如果仍有问题，降低vocal_volume和bgm_volume。

### Q5: 某些TTS片段缺失？
**A**: 检查TTS文件是否存在，查看控制台警告信息。

## 性能

- **内存**: 取决于音频长度，1小时视频约需1-2GB内存
- **速度**: 主要取决于音频I/O，通常<10秒
- **格式**: 输出为WAV格式（无损），可以后续转换为MP3等

## 下一步

音频混合完成后，使用 `_combine_video_audio()` 将音频合并到视频中，完成整个Pipeline。
