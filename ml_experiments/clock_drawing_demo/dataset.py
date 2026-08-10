"""画钟图片发现、分层划分和安全预处理。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset


CLASS_NAMES = {
    0: "no_clock",
    1: "severe_visuospatial_error",
    2: "moderate_visuospatial_or_hand_error",
    3: "hand_or_time_error",
    4: "minor_visuospatial_error",
    5: "perfect_clock",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def discover_samples(data_root):
    data_root = Path(data_root).resolve()
    samples = []
    for class_directory in sorted(data_root.iterdir()):
        if not class_directory.is_dir():
            continue
        match = re.match(r"^([0-5])(?:_|$)", class_directory.name)
        if not match:
            continue
        label = int(match.group(1))
        for path in sorted(class_directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(Sample(path=path, label=label))
    if not samples:
        raise FileNotFoundError(f"没有在 {data_root} 找到按0～5分目录存放的图片。")
    return samples


def _counts(samples):
    return {label: sum(sample.label == label for sample in samples) for label in range(6)}


def create_split_manifest(data_root, manifest_path, seed=20260808):
    """在类别内部固定洗牌，按照70/15/15划分，避免增强后泄漏。"""
    data_root = Path(data_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    generator = random.Random(seed)
    grouped = {label: [] for label in range(6)}
    for sample in discover_samples(data_root):
        grouped[sample.label].append(sample)

    rows = []
    for label, samples in grouped.items():
        if len(samples) < 3:
            raise RuntimeError(f"类别 {label} 少于3张图片，无法进行三段划分。")
        generator.shuffle(samples)
        test_count = max(1, round(len(samples) * 0.15))
        validation_count = max(1, round(len(samples) * 0.15))
        if test_count + validation_count >= len(samples):
            validation_count = 1
            test_count = 1
        train_count = len(samples) - validation_count - test_count
        split_samples = (
            [("train", item) for item in samples[:train_count]]
            + [("validation", item) for item in samples[train_count:train_count + validation_count]]
            + [("test", item) for item in samples[train_count + validation_count:]]
        )
        for split, sample in split_samples:
            rows.append({
                "split": split,
                "label": sample.label,
                "class_name": CLASS_NAMES[sample.label],
                "image_path": sample.path.relative_to(data_root).as_posix(),
            })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest(manifest_path, split=None):
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if split:
        rows = [row for row in rows if row["split"] == split]
    return rows


def crop_and_pad(image, image_size=224):
    """去除大块空白并保持比例；黑色笔迹最终转换为高像素值。"""
    image = ImageOps.grayscale(image)
    inverted = ImageOps.invert(image)
    mask = inverted.point(lambda value: 255 if value >= 18 else 0)
    bbox = mask.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        margin = max(8, int(max(right - left, bottom - top) * 0.08))
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(image.width, right + margin)
        bottom = min(image.height, bottom + margin)
        image = image.crop((left, top, right, bottom))
    side = max(image.width, image.height, 1)
    canvas = Image.new("L", (side, side), color=255)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((image_size, image_size), Image.Resampling.LANCZOS)


def augment_image(image, generator):
    """只做不会改变钟表语义的轻度增强，禁止翻转和大角度旋转。"""
    angle = generator.uniform(-3.0, 3.0)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
    image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.9, 1.12))
    image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.94, 1.06))
    return image


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = 1.0 - array
    return torch.from_numpy(array.copy()).unsqueeze(0)


def load_image_tensor(path, image_size=224, augment=False, generator=None):
    with Image.open(path) as image:
        image = crop_and_pad(image, image_size=image_size)
    if augment:
        image = augment_image(image, generator or random.Random())
    return image_to_tensor(image)


class ClockDrawingDataset(Dataset):
    def __init__(self, data_root, manifest_path, split, image_size=224, augment=False, seed=0):
        self.data_root = Path(data_root).resolve()
        self.rows = read_manifest(manifest_path, split=split)
        self.image_size = image_size
        self.augment = augment
        self.generator = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["image_path"]
        tensor = load_image_tensor(
            path,
            image_size=self.image_size,
            augment=self.augment,
            generator=self.generator,
        )
        return tensor, int(row["label"]), str(path)


def manifest_statistics(rows):
    result = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        result[split] = {
            "total": len(selected),
            "by_class": {
                str(label): sum(int(row["label"]) == label for row in selected)
                for label in range(6)
            },
        }
    result["all"] = {
        "total": len(rows),
        "by_class": {
            str(label): sum(int(row["label"]) == label for row in rows)
            for label in range(6)
        },
    }
    return result
