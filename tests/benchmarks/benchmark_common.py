"""论文性能测试脚本的公共统计、登录和CER工具。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import statistics
import unicodedata

import requests
from opencc import OpenCC


TESTS_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = TESTS_ROOT / "results"
TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mean(values):
    return statistics.fmean(values) if values else None


def p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def write_csv(filename, rows, fieldnames):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / filename
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def extract_csrf(html):
    patterns = (
        r'name="_csrf_token"\s+value="([^"]+)"',
        r'name="csrf-token"\s+content="([^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    raise RuntimeError("页面中没有找到CSRF token。")


def login_session(base_url, username, password):
    session = requests.Session()
    login_page = session.get(f"{base_url.rstrip('/')}/login", timeout=30)
    login_page.raise_for_status()
    token = extract_csrf(login_page.text)
    response = session.post(
        f"{base_url.rstrip('/')}/login",
        data={
            "_csrf_token": token,
            "username": username,
            "password": password,
        },
        timeout=30,
    )
    response.raise_for_status()
    if "/login" in response.url:
        raise RuntimeError("登录失败，请检查测试账号和密码。")
    token = extract_csrf(response.text)
    return session, token


def normalize_raw_transcript(text):
    """只统一字符宽度并移除空白，用于保留标点差异的原始 CER。"""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(character for character in normalized if not character.isspace())


def normalize_chinese(text):
    """基础规范化：统一宽度、大小写并忽略空白与标点。

    这里刻意不自动改写同义词，也不把所有数字强行转换成中文数字，避免测试
    脚本在没有人工依据时替识别系统“改对”答案。
    """
    normalized = normalize_raw_transcript(text).replace("〇", "零")
    # FLEURS 粤语参考文本使用繁体，而腾讯云可能返回简体。先统一到简体，
    # 避免把“兩/两、週/周”等书写体系差异错误计入识别 CER。
    normalized = TRADITIONAL_TO_SIMPLIFIED.convert(normalized)
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "S"))
    )


def edit_distance(reference, prediction, normalizer=normalize_chinese):
    """返回字符级Levenshtein距离。"""
    reference = normalizer(reference)
    prediction = normalizer(prediction)
    previous = list(range(len(prediction) + 1))
    for row_index, reference_character in enumerate(reference, start=1):
        current = [row_index]
        for column_index, prediction_character in enumerate(
            prediction,
            start=1,
        ):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + int(reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference, prediction):
    normalized_reference = normalize_chinese(reference)
    if not normalized_reference:
        raise ValueError("标准文本不能为空。")
    return edit_distance(reference, prediction) / len(normalized_reference)


def raw_character_error_rate(reference, prediction):
    normalized_reference = normalize_raw_transcript(reference)
    if not normalized_reference:
        raise ValueError("标准文本不能为空。")
    return edit_distance(
        reference,
        prediction,
        normalizer=normalize_raw_transcript,
    ) / len(normalized_reference)
