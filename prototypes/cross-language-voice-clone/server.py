"""部署在 GPU 服务器的语音输入/输出 API。"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.asr.asr_engine import ASREngine
from src.chat.llm_engine import ChatEngine, ChatEngineError
from src.conversation import VoiceConversationPipeline
from src.tts.indextts2_engine import IndexTTS2Engine, IndexTTS2Error


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
app = FastAPI(title="ASR + IndexTTS2 Voice Clone API", docs_url=None, redoc_url=None)
_conversation: VoiceConversationPipeline | None = None


async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持的音频格式")
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="音频必须大于 0 且不超过 25MB")
    handle = tempfile.NamedTemporaryFile(prefix="voice_input_", suffix=suffix, delete=False)
    try:
        handle.write(content)
        return Path(handle.name)
    finally:
        handle.close()


def conversation() -> VoiceConversationPipeline:
    global _conversation
    if _conversation is None:
        _conversation = VoiceConversationPipeline(
            asr=ASREngine(
                model_size=os.getenv("ASR_MODEL_SIZE", "small"),
                language=os.getenv("ASR_LANGUAGE", "zh"),
                device=os.getenv("ASR_DEVICE") or None,
            ),
            chat=ChatEngine(),
            tts=IndexTTS2Engine(),
        )
    return _conversation


@app.get("/health")
def health() -> dict[str, bool]:
    return {"status": True, "asr_loaded": _conversation is not None}


@app.post("/v1/voice-profiles", status_code=201)
async def enroll_voice(
    reference_audio: UploadFile = File(...),
    consent_confirmed: bool = Form(...),
):
    path = await save_upload(reference_audio)
    try:
        voice_id = IndexTTS2Engine().enroll_voice(str(path), consent_confirmed)
        return {"voice_id": voice_id}
    except IndexTTS2Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/conversation/turn")
async def conversation_turn(
    audio: UploadFile = File(...),
    voice_id: str = Form(...),
    session_id: str = Form("default"),
):
    source = await save_upload(audio)
    descriptor, output_name = tempfile.mkstemp(prefix="voice_reply_", suffix=".wav")
    os.close(descriptor)
    output = Path(output_name)
    try:
        result = conversation().handle_turn(
            str(source), voice_id, str(output), session_id=session_id
        )
        audio_base64 = base64.b64encode(output.read_bytes()).decode("ascii")
        return {
            "transcript": result["transcript"],
            "reply": result["reply"],
            "audio_base64": audio_base64,
            "content_type": "audio/wav",
        }
    except (IndexTTS2Error, ChatEngineError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        source.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
