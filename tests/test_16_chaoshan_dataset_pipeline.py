"""Teochew-Wild 潮汕话固定抽样管线的离线测试。"""

from pathlib import Path

from tests.datasets.prepare_chaoshan_teochew_wild import (
    category_for,
    parse_annotation,
    select_across_speakers,
)


def test_category_for_uses_stable_character_thresholds():
    assert category_for("一二三") == "short"
    assert category_for("一" * 13) == "medium"
    assert category_for("一" * 25) == "long"


def test_fixed_selection_covers_speakers_before_repeating(tmp_path: Path):
    annotation_lines = []
    for speaker_number in range(1, 21):
        speaker = f"S{speaker_number:03d}"
        speaker_dir = tmp_path / speaker
        speaker_dir.mkdir()
        for clip_number in range(2):
            filename = f"{speaker}C{clip_number:03d}.wav"
            (speaker_dir / filename).write_bytes(b"test")
            annotation_lines.append(
                f"{speaker}/{filename}|{speaker}|潮汕话样本{speaker_number}|dio5 jiu1 ue7"
            )
    annotation = tmp_path / "annotation.txt"
    annotation.write_text("\n".join(annotation_lines), encoding="utf-8")

    candidates = parse_annotation(annotation, tmp_path)
    first = select_across_speakers(candidates, count=30, seed=20260808)
    second = select_across_speakers(candidates, count=30, seed=20260808)

    assert [row["relative_audio"] for row in first] == [
        row["relative_audio"] for row in second
    ]
    assert len(first) == 30
    assert len({row["speaker"] for row in first[:20]}) == 20
    assert len({row["speaker"] for row in first}) == 20
