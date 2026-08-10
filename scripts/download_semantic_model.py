"""预下载日常判断题使用的本地多语言语义模型。"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_lantern.assessment.semantic_scoring import DEFAULT_MODEL_NAME  # noqa: E402


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "缺少 sentence-transformers。请先执行："
            "python -m pip install -r requirements-ml.txt"
        ) from error

    print(f"正在下载并校验模型：{DEFAULT_MODEL_NAME}")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    dimension = model.get_sentence_embedding_dimension()
    print(f"模型准备完成，向量维度：{dimension}。运行应用时可保持 SEMANTIC_MODEL_LOCAL_ONLY=true。")


if __name__ == "__main__":
    main()
