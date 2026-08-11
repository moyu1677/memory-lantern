# Remote Whisper ASR service

This FastAPI service is deployed on the GPU host and is intentionally bound to loopback. The Flask application should reach it only through an SSH tunnel or a controlled private network.

1. Prepare a local Whisper large-v3 checkpoint on the GPU host and install `ffmpeg`.
2. Copy `.env.example` to `.env`; set a long random `ASR_API_KEY`, the absolute `ASR_MODEL_PATH`, and the intended GPU mapping.
3. Install `python -m pip install -r requirements-asr-server.txt` in the server environment.
4. Start `uvicorn remote_whisper_server:app --host 127.0.0.1 --port 8000` from this directory.
5. Forward the port to the application host and set its `ASR_URL`, `ASR_API_KEY`, and timeout in the root `.env`.

The API accepts only WAV, MP3, M4A, WebM, or OGG files up to 20 MB. It requires the `X-API-Key` header, stores uploads only in temporary files while decoding, and returns text only.
