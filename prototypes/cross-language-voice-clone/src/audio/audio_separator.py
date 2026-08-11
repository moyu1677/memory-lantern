"""
音频分离模块 - 使用Demucs分离人声和BGM
"""

import os
import subprocess
from pathlib import Path

class AudioSeparator:
    def __init__(self, model="htdemucs"):
        """
        初始化音频分离器

        Args:
            model: Demucs模型名称 (htdemucs, htdemucs_ft, etc.)
        """
        self.model = model
        print(f"🔧 音频分离器初始化: {model}")

    def separate_vocals(self, audio_path: str, output_dir: str) -> dict:
        """
        分离人声和BGM

        Args:
            audio_path: 输入音频路径
            output_dir: 输出目录

        Returns:
            dict: {"vocals": 人声路径, "bgm": BGM路径}
        """
        print(f"🎵 开始分离音频: {audio_path}")

        os.makedirs(output_dir, exist_ok=True)

        try:
            # 运行demucs命令
            cmd = [
                "demucs",
                "--two-stems", "vocals",  # 只分离人声和其他
                "-o", output_dir,
                "--float32",  # 使用float32精度
                "-n", self.model,
                audio_path
            ]

            print("   运行Demucs...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Demucs输出路径
            audio_name = Path(audio_path).stem
            model_output_dir = os.path.join(output_dir, self.model, audio_name)

            vocals_path = os.path.join(model_output_dir, "vocals.wav")
            no_vocals_path = os.path.join(model_output_dir, "no_vocals.wav")

            if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
                print("✅ 音频分离成功!")
                return {
                    "vocals": vocals_path,
                    "bgm": no_vocals_path
                }
            else:
                print("❌ 输出文件未找到")
                return None

        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs运行失败: {e}")
            print(f"   stderr: {e.stderr}")
            return None

        except FileNotFoundError:
            print("❌ 未找到demucs命令")
            print("   请安装: pip install demucs")
            return None

    def separate_video_audio(self, video_path: str, output_dir: str) -> dict:
        """
        从视频中提取音频并分离

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录

        Returns:
            dict: 分离结果
        """
        # 先提取音频
        audio_path = self._extract_audio_from_video(video_path, output_dir)

        if not audio_path:
            return None

        # 分离音频
        return self.separate_vocals(audio_path, output_dir)

    def _extract_audio_from_video(self, video_path: str, output_dir: str) -> str:
        """使用FFmpeg提取音频"""
        print("🎬 从视频提取音频...")

        output_path = os.path.join(output_dir, "extracted_audio.wav")

        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # 不要视频
                "-acodec", "pcm_s16le",  # PCM 16bit
                "-ar", "44100",  # 采样率44.1kHz
                "-ac", "2",  # 立体声
                "-y",  # 覆盖输出
                output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            print(f"✅ 音频提取完成: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 音频提取失败: {e}")
            return None


if __name__ == "__main__":
    # 测试音频分离
    separator = AudioSeparator()

    test_audio = "data/extracted_audio/test.wav"
    output_dir = "data/separated_audio"

    if os.path.exists(test_audio):
        result = separator.separate_vocals(test_audio, output_dir)

        if result:
            print(f"人声: {result['vocals']}")
            print(f"BGM: {result['bgm']}")
    else:
        print(f"测试音频不存在: {test_audio}")
