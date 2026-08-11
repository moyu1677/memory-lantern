"""
视频合成模块
使用FFmpeg将新音频与原视频合成
"""

import os
import subprocess
import ffmpeg

class VideoComposer:
    def __init__(self):
        """
        初始化视频合成器
        """
        print("[*] VideoComposer initialized")

    def combine_video_audio(self, video_path, audio_path, output_path,
                           video_codec="copy", audio_codec="aac", audio_bitrate="192k"):
        """
        将音频与视频合成

        Args:
            video_path: 原视频路径
            audio_path: 新音频路径
            output_path: 输出视频路径
            video_codec: 视频编码器 ("copy"表示不重新编码)
            audio_codec: 音频编码器
            audio_bitrate: 音频比特率

        Returns:
            输出视频路径或None（失败时）
        """
        print("\n[*] Starting video-audio composition...")
        print(f"    Video: {video_path}")
        print(f"    Audio: {audio_path}")
        print(f"    Output: {output_path}")

        # 检查输入文件
        if not os.path.exists(video_path):
            print(f"[-] Video file not found: {video_path}")
            return None

        if not os.path.exists(audio_path):
            print(f"[-] Audio file not found: {audio_path}")
            return None

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            # 方法1: 使用ffmpeg-python（推荐）
            print("[*] Using ffmpeg-python...")

            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(audio_path)

            # 合成：取视频流和新音频流
            output = ffmpeg.output(
                video.video,  # 原视频流（不重新编码）
                audio.audio,  # 新音频流
                output_path,
                vcodec=video_codec,  # 视频编码器
                acodec=audio_codec,  # 音频编码器
                audio_bitrate=audio_bitrate,  # 音频比特率
                shortest=None  # 以较短的流为准
            )

            # 覆盖已存在的文件
            output = ffmpeg.overwrite_output(output)

            # 执行
            ffmpeg.run(output, capture_stdout=True, capture_stderr=True)

            print(f"[+] Video composition completed: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            print(f"[-] FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")

            # 尝试方法2: 直接调用ffmpeg命令
            print("[*] Trying direct ffmpeg command...")
            return self._combine_with_command(video_path, audio_path, output_path,
                                             video_codec, audio_codec, audio_bitrate)

        except Exception as e:
            print(f"[-] Composition failed: {e}")
            return None

    def _combine_with_command(self, video_path, audio_path, output_path,
                             video_codec="copy", audio_codec="aac", audio_bitrate="192k"):
        """
        使用直接命令行调用FFmpeg（备用方法）
        """
        cmd = [
            'ffmpeg',
            '-i', video_path,      # 输入视频
            '-i', audio_path,      # 输入音频
            '-c:v', video_codec,   # 视频编码器
            '-c:a', audio_codec,   # 音频编码器
            '-b:a', audio_bitrate, # 音频比特率
            '-map', '0:v:0',       # 取第一个文件的视频流
            '-map', '1:a:0',       # 取第二个文件的音频流
            '-shortest',           # 以较短的流为准
            '-y',                  # 覆盖输出文件
            output_path
        ]

        try:
            print(f"[*] Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                print(f"[+] Video composition completed: {output_path}")
                return output_path
            else:
                print(f"[-] FFmpeg command failed:")
                print(result.stderr)
                return None

        except subprocess.TimeoutExpired:
            print("[-] FFmpeg command timeout")
            return None
        except Exception as e:
            print(f"[-] Command execution failed: {e}")
            return None

    def extract_audio(self, video_path, output_audio_path, audio_format="wav"):
        """
        从视频中提取音频（辅助功能）

        Args:
            video_path: 视频路径
            output_audio_path: 输出音频路径
            audio_format: 音频格式

        Returns:
            音频路径或None
        """
        print(f"\n[*] Extracting audio from video...")
        print(f"    Video: {video_path}")
        print(f"    Output: {output_audio_path}")

        if not os.path.exists(video_path):
            print(f"[-] Video not found: {video_path}")
            return None

        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_audio_path, acodec='pcm_s16le', ar='24000')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            print(f"[+] Audio extracted: {output_audio_path}")
            return output_audio_path

        except Exception as e:
            print(f"[-] Audio extraction failed: {e}")
            return None

    def get_video_info(self, video_path):
        """
        获取视频信息（辅助功能）

        Returns:
            dict with video info
        """
        try:
            probe = ffmpeg.probe(video_path)

            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

            info = {
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'video': {
                    'codec': video_info['codec_name'],
                    'width': video_info['width'],
                    'height': video_info['height'],
                    'fps': eval(video_info['r_frame_rate'])
                }
            }

            if audio_info:
                info['audio'] = {
                    'codec': audio_info['codec_name'],
                    'sample_rate': audio_info['sample_rate'],
                    'channels': audio_info['channels']
                }

            return info

        except Exception as e:
            print(f"[-] Failed to get video info: {e}")
            return None


if __name__ == "__main__":
    # 测试
    composer = VideoComposer()

    # 示例：合成视频
    # composer.combine_video_audio("input.mp4", "new_audio.wav", "output.mp4")

    print("[*] VideoComposer module ready")
