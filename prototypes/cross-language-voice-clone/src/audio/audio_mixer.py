"""
音频混合模块
将TTS生成的多个音频片段按时间戳拼接，并与BGM混合
"""

import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import soundfile as sf

class AudioMixer:
    def __init__(self, sample_rate=24000):
        """
        初始化音频混合器

        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        print(f"[*] AudioMixer initialized with sample_rate={sample_rate}")

    def mix_tts_segments(self, segments, tts_output_dir, output_path, bgm_path=None,
                        vocal_volume=0, bgm_volume=-3):
        """
        混合TTS片段和BGM

        Args:
            segments: ASR 分段列表，包含时间戳和 TTS 音频路径
            tts_output_dir: TTS输出目录
            output_path: 最终混合音频输出路径
            bgm_path: BGM音频路径（可选）
            vocal_volume: 人声音量调整（dB）
            bgm_volume: BGM音量调整（dB）

        Returns:
            混合后的音频路径
        """
        print("\n[*] Starting audio mixing...")
        print(f"    Segments: {len(segments)}")
        print(f"    TTS dir: {tts_output_dir}")
        print(f"    Output: {output_path}")

        # 步骤1: 按时间戳拼接TTS片段
        print("\n[1/3] Concatenating TTS segments by timestamp...")
        full_vocal = self._concatenate_tts_segments(segments, tts_output_dir)

        if full_vocal is None:
            print("[-] Failed to concatenate TTS segments")
            return None

        print(f"    [+] Total vocal duration: {len(full_vocal)/1000:.2f}s")

        # 步骤2: 调整人声音量
        print("\n[2/3] Adjusting vocal volume...")
        full_vocal = full_vocal + vocal_volume  # dB adjustment
        print(f"    [+] Vocal volume adjusted by {vocal_volume}dB")

        # 步骤3: 与BGM混合（如果有）
        if bgm_path and os.path.exists(bgm_path):
            print("\n[3/3] Mixing with BGM...")
            final_audio = self._mix_with_bgm(full_vocal, bgm_path, bgm_volume)
        else:
            print("\n[3/3] No BGM provided, using vocal only...")
            final_audio = full_vocal

        # 归一化并导出
        print("\n[*] Normalizing and exporting...")
        final_audio = normalize(final_audio)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_audio.export(output_path, format="wav")

        print(f"[+] Audio mixing completed: {output_path}")
        print(f"    Duration: {len(final_audio)/1000:.2f}s")

        return output_path

    def _concatenate_tts_segments(self, segments, tts_output_dir):
        """
        按时间戳拼接TTS片段

        策略：
        1. 找到所有段落的时间范围
        2. 创建空白音频作为底板
        3. 将每个TTS片段放置在对应的时间位置
        """
        # 获取总时长
        if not segments:
            print("[-] No segments provided")
            return None

        # 找到最后一个段落的结束时间
        max_end_time = max(seg.get("end", 0) for seg in segments)
        total_duration_ms = int(max_end_time * 1000)  # 转换为毫秒

        print(f"    Total duration: {max_end_time:.2f}s")

        # 创建空白音频底板
        silence = AudioSegment.silent(duration=total_duration_ms)

        # 将每个TTS片段放置到对应位置
        placed_count = 0
        for i, seg in enumerate(segments):
            # 获取TTS音频路径
            tts_path = seg.get("tts_audio")

            if not tts_path:
                # 如果没有tts_audio字段，尝试使用索引
                tts_path = os.path.join(tts_output_dir, f"tts_{i:04d}.wav")

            if not os.path.exists(tts_path):
                print(f"    [!] TTS file not found: {tts_path}, skipping...")
                continue

            # 加载TTS音频
            try:
                tts_audio = AudioSegment.from_wav(tts_path)
            except Exception as e:
                print(f"    [!] Failed to load {tts_path}: {e}")
                continue

            # 获取时间戳
            start_time = seg.get("start", 0)
            start_ms = int(start_time * 1000)

            # 放置音频
            silence = silence.overlay(tts_audio, position=start_ms)
            placed_count += 1

            if (i + 1) % 10 == 0:
                print(f"    Placed {i + 1}/{len(segments)} segments")

        print(f"    [+] Successfully placed {placed_count}/{len(segments)} segments")

        return silence

    def _mix_with_bgm(self, vocal, bgm_path, bgm_volume=-3):
        """
        将人声与BGM混合

        Args:
            vocal: 人声AudioSegment
            bgm_path: BGM文件路径
            bgm_volume: BGM音量调整（dB）

        Returns:
            混合后的AudioSegment
        """
        print(f"    Loading BGM: {bgm_path}")

        try:
            bgm = AudioSegment.from_file(bgm_path)
        except Exception as e:
            print(f"    [-] Failed to load BGM: {e}")
            return vocal

        # 调整BGM音量
        bgm = bgm + bgm_volume
        print(f"    [+] BGM volume adjusted by {bgm_volume}dB")

        # 调整BGM长度以匹配人声
        vocal_duration = len(vocal)
        bgm_duration = len(bgm)

        if bgm_duration < vocal_duration:
            # BGM太短，循环播放
            print(f"    [*] BGM too short, looping...")
            loops_needed = int(vocal_duration / bgm_duration) + 1
            bgm = bgm * loops_needed

        # 裁剪BGM到与人声相同长度
        bgm = bgm[:vocal_duration]
        print(f"    [+] BGM adjusted to {len(bgm)/1000:.2f}s")

        # 混合
        mixed = vocal.overlay(bgm)
        print(f"    [+] Mixed vocal and BGM")

        return mixed

    def create_silence_with_segments(self, segments, output_path):
        """
        创建带时间戳的静音音频（用于测试）

        Args:
            segments: 包含时间戳的segments
            output_path: 输出路径
        """
        if not segments:
            return None

        max_end_time = max(seg.get("end", 0) for seg in segments)
        total_duration_ms = int(max_end_time * 1000)

        silence = AudioSegment.silent(duration=total_duration_ms)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        silence.export(output_path, format="wav")

        print(f"[+] Created silence audio: {output_path} ({max_end_time:.2f}s)")
        return output_path


if __name__ == "__main__":
    # 测试示例
    mixer = AudioMixer()

    # 示例segments
    test_segments = [
        {"start": 0.0, "end": 2.0, "text": "你好", "tts_audio": "test1.wav"},
        {"start": 2.5, "end": 4.5, "text": "世界", "tts_audio": "test2.wav"},
    ]

    # 混合（需要实际的音频文件）
    # mixer.mix_tts_segments(test_segments, "tts_output", "mixed.wav", "bgm.wav")

    print("[*] AudioMixer module ready")
