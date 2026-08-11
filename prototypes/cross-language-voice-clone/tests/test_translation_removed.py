"""不依赖模型下载的结构性回归检查。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TranslationRemovalTests(unittest.TestCase):
    def test_runtime_pipeline_does_not_import_translator(self):
        source = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("translation.translator", source)
        self.assertNotIn("translate_segments", source)

    def test_tts_uses_voice_id_contract(self):
        source = (ROOT / "src" / "tts" / "indextts2_engine.py").read_text(encoding="utf-8")
        self.assertIn('/v1/voices', source)
        self.assertIn('/v1/speech', source)
        self.assertIn('"voice_id": voice_id', source)


if __name__ == "__main__":
    unittest.main()
