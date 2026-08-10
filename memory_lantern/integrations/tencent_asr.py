"""腾讯云实时语音识别客户端，用于短音频方言识别。"""

import base64
import hashlib
import hmac
import json
import random
import time
import urllib.parse
import uuid

import websocket


ASR_HOST = "asr.cloud.tencent.com"
ASR_PATH = "/asr/v2"
PCM_BYTES_PER_SECOND = 16000 * 2  # 16kHz、16bit、单声道
CHUNK_SIZE = 6400  # 200ms PCM


class TencentASRError(RuntimeError):
    pass


def _signed_websocket_url(app_id, secret_id, secret_key, engine):
    timestamp = int(time.time())
    params = {
        "engine_model_type": engine,
        "expired": timestamp + 86400,
        "filter_empty_result": 1,
        "filter_modal": 0,
        "filter_punc": 0,
        "needvad": 1,
        "nonce": random.randint(100000, 999999999),
        "secretid": secret_id,
        "timestamp": timestamp,
        "vad_silence_time": 1000,
        "voice_format": 1,
        "voice_id": uuid.uuid4().hex,
    }
    query = urllib.parse.urlencode(sorted(params.items()))
    sign_text = f"{ASR_HOST}{ASR_PATH}/{app_id}?{query}"
    digest = hmac.new(
        secret_key.encode("utf-8"),
        sign_text.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = urllib.parse.quote(base64.b64encode(digest).decode("utf-8"), safe="")
    return f"wss://{sign_text}&signature={signature}"


def _parse_message(raw_message, stable_results):
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    payload = json.loads(raw_message)

    if payload.get("code", 0) != 0:
        raise TencentASRError(
            f"腾讯云语音识别失败（{payload.get('code')}）："
            f"{payload.get('message', '未知错误')}"
        )

    result = payload.get("result")
    if result and result.get("slice_type") == 2:
        stable_results[result.get("index", len(stable_results))] = (
            result.get("voice_text_str", "")
        )
    return payload.get("final") == 1


def recognize_pcm(pcm_path, app_id, secret_id, secret_key, engine="16k_zh_en"):
    """识别16kHz、16bit、单声道小端PCM文件，返回稳定文本。"""
    url = _signed_websocket_url(app_id, secret_id, secret_key, engine)
    stable_results = {}
    connection = None

    try:
        connection = websocket.create_connection(url, timeout=10)
        handshake = connection.recv()
        _parse_message(handshake, stable_results)

        with open(pcm_path, "rb") as audio_file:
            while True:
                chunk = audio_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                connection.send_binary(chunk)
                # 实时接口要求音频按约1:1的速度发送。
                time.sleep(len(chunk) / PCM_BYTES_PER_SECOND)

        connection.send(json.dumps({"type": "end"}))
        connection.settimeout(20)

        while True:
            message = connection.recv()
            if _parse_message(message, stable_results):
                break

    except TencentASRError:
        raise
    except Exception as exc:
        raise TencentASRError(f"无法连接腾讯云语音识别服务：{exc}") from exc
    finally:
        if connection is not None:
            connection.close()

    return "".join(stable_results[index] for index in sorted(stable_results)).strip()
