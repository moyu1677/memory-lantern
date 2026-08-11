import os
import json
from glob import glob

class ASREngine:
    def __init__(self, model_size="small", language="zh", device=None):
        """
        Whisper ASR 模块
        """
        # 延迟导入，令部署前的静态检查不依赖已下载的 Whisper。
        import whisper

        print(f"🔧 正在加载 Whisper 模型: {model_size}")
        self.model = whisper.load_model(model_size, device=device)
        self.language = language

    def transcribe_one(self, audio_path):
        """
        识别单个音频片段
        """
        print(f"🎙️ 正在识别: {audio_path}")
        result = self.model.transcribe(audio_path, language=self.language)
        return result["text"], result["segments"]

    def transcribe_folder(self, folder_path="data/segmented_audio", output_path="data/asr_results/asr_output.json"):
        """
        批量识别 segmented_audio 文件夹中的音频
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio_files = sorted(glob(os.path.join(folder_path, "segment_*.wav")))

        all_segments = []
        all_texts = []

        for i, audio_path in enumerate(audio_files):
            text, segments = self.transcribe_one(audio_path)

            # 记录段落整体文本
            all_texts.append(text)

            # 记录每句带时间的分段
            for seg in segments:
                seg["file_id"] = i
                seg["file_name"] = os.path.basename(audio_path)
                all_segments.append(seg)

        # 整合输出
        result = {
            "full_text": " ".join(all_texts),
            "segments": all_segments
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ 全部识别完成，共 {len(audio_files)} 段，结果已保存至: {output_path}")
        return result


if __name__ == "__main__":
    asr = ASREngine(model_size="small", language="zh")
    asr.transcribe_folder()
