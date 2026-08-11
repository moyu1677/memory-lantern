"""兼容旧文件名的导入入口。请优先使用 :mod:`src.pipeline`。"""

from .pipeline import VoiceCloneDubbingPipeline

# 保留旧类名，避免既有脚本的导入路径失效。
CrossLanguageDubbingPipeline = VoiceCloneDubbingPipeline

__all__ = ["VoiceCloneDubbingPipeline", "CrossLanguageDubbingPipeline"]
