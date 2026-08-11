# 无翻译同声克隆系统

> 这是忆路明灯的独立语音原型，保存在 `prototypes/` 供复现实验与 GPU
> 部署参考。Web 主应用使用根目录的 `remote_asr/` 和 `remote_tts/` 服务；
> 不会在 Flask 进程中自动启动此原型。

这是从原“跨语言配音”项目拆出的部署副本。原项目没有被修改；本副本明确移除了翻译路径与翻译依赖。

## 两条工作流

```text
视频配音：视频 → 人声分离 → Whisper ASR → IndexTTS2 同声克隆 → 混音 → 输出视频
语音对话：录音 → Whisper ASR → AI 回复 → IndexTTS2 同声克隆 → 返回 WAV
```

第一次使用某个家属音色时，调用 `POST /v1/voice-profiles` 上传参考音频并确认授权，取得 `voice_id`。之后每次对话只传 `voice_id`，不再传参考音频或远程不可访问的本机路径。

## 部署原则

- `remote_tts`（IndexTTS2）固定在 GPU 1，监听 `127.0.0.1:5000`；
- 本项目的 ASR/对话 API 固定在 GPU 0；
- 两个服务不共享同一物理 GPU；
- `INDEXTTS2_API_KEY` 和 `CHAT_API_KEY` 仅写入服务器私有 `.env`，不从原项目或本机数据目录复制。

## 运行

```bash
cp .env.example .env
# 填入 INDEXTTS2_API_KEY 和 CHAT_API_KEY
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES=0 uvicorn server:app --host 127.0.0.1 --port 8000
```

语音对话请求为 multipart：`audio`、`voice_id`、可选 `session_id`。响应包含 ASR 文本、AI 回复和 Base64 WAV。视频配音可在 Python 中调用：

```python
from src.pipeline import VoiceCloneDubbingPipeline

pipeline = VoiceCloneDubbingPipeline(output_dir="data/output")
pipeline.run("data/input_videos/demo.mp4", voice_id="已授权音色编号")
```
