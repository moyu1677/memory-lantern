# GPU 服务器：IndexTTS2 语音服务

把此目录复制到已成功部署 IndexTTS2 的 GPU 服务器。它使用 FastAPI/Uvicorn，
可复用 IndexTTS2 的既有 Python 环境，不需要另建 Flask 环境。它提供固定接口：

- POST /v1/voices：首次、经声音所有者明确授权后上传参考音频，返回 voice_id；
- POST /v1/speech：提交 voice_id 和文本，直接返回 WAV 音频；
- DELETE /v1/voices/{voice_id}：撤回授权并删除云端参考音频；
- GET /health：健康检查。

应用服务器不会保存参考音频，只保存 voice_id、授权时间和上传人。
上传的参考音频会由服务器上的 `ffmpeg` 统一转换为最多 30 秒、24kHz 单声道 WAV；
无法解码的文件会被拒绝，不会生成无效音色档案。

## 启动

1. 先检查 IndexTTS2 的 Python 环境是否已有此目录 requirements-voice-server.txt 的依赖；缺失时才安装。
2. 复制 .env.example 为 .env，设置 VOICE_API_KEY 和模型路径。
3. 从此目录运行：

       /path/to/indextts2/.venv/bin/python remote_indextts_server.py

4. 默认仅监听 `127.0.0.1:5000`。从本地应用访问时建立 SSH 隧道；不要将该端口直接暴露到公网：

       ssh -N -L 15000:127.0.0.1:5000 teacher-server

   应用服务器 `.env` 中填写：

       INDEXTTS2_URL=http://127.0.0.1:15000
       INDEXTTS2_API_KEY=<与远程相同的密钥>

当应用和语音服务同机部署时，填写 `INDEXTTS2_URL=http://127.0.0.1:5000`，并将 Flask 应用端口设为其他值（例如 `FLASK_PORT=8000`）。

此服务以单个推理锁串行处理合成请求，避免同一张 GPU 上的并发推理相互抢显存。生产部署建议用反向代理和 TLS。
