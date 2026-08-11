"""无翻译的视频同声克隆流水线。

流程：视频/音频 → 人声分离 → ASR → IndexTTS2 同声克隆 → 混音 → 视频输出。
"""

from __future__ import annotations

import json
from pathlib import Path

from .asr.asr_engine import ASREngine
from .audio.audio_mixer import AudioMixer
from .audio.audio_separator import AudioSeparator
from .tts.indextts2_engine import IndexTTS2Engine
from .video.video_composer import VideoComposer


class VoiceCloneDubbingPipeline:
    """将原声文字直接用已授权的目标音色重新合成，不做翻译。"""

    def __init__(
        self,
        asr_model: str = "small",
        tts_base_url: str | None = None,
        tts_api_key: str | None = None,
        output_dir: str = "data/output",
        asr_device: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.separator = AudioSeparator()
        self.asr = ASREngine(
            model_size=asr_model,
            language="zh",
            device=asr_device,
        )
        self.tts = IndexTTS2Engine(
            base_url=tts_base_url,
            api_key=tts_api_key,
        )
        self.mixer = AudioMixer()
        self.composer = VideoComposer()

    def run(self, video_path: str, voice_id: str) -> str:
        """生成使用 ``voice_id`` 对应已授权音色的配音视频。"""

        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"找不到输入视频：{source}")

        separated = self.separator.separate_video_audio(
            str(source), str(self.output_dir / "separated")
        )
        if not separated:
            raise RuntimeError("音频分离失败")

        _, segments = self.asr.transcribe_one(separated["vocals"])
        segments = [
            segment
            for segment in segments
            if str(segment.get("text", "")).strip()
        ]
        if not segments:
            raise RuntimeError("ASR 未识别到可合成的语音内容")

        (self.output_dir / "asr_segments.json").write_text(
            json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        tts_dir = self.output_dir / "tts_outputs"
        self.tts.synthesize_segments(segments, str(tts_dir), voice_id=voice_id)

        final_audio = self.mixer.mix_tts_segments(
            segments=segments,
            tts_output_dir=str(tts_dir),
            output_path=str(self.output_dir / "final_audio.wav"),
            bgm_path=separated["bgm"],
            vocal_volume=0,
            bgm_volume=-3,
        )
        if not final_audio:
            raise RuntimeError("音频混合失败")

        output_path = self.output_dir / f"{source.stem}_voice_clone.mp4"
        result = self.composer.combine_video_audio(
            video_path=str(source),
            audio_path=final_audio,
            output_path=str(output_path),
            video_codec="copy",
            audio_codec="aac",
            audio_bitrate="192k",
        )
        if not result:
            raise RuntimeError("视频合成失败")
        return result
