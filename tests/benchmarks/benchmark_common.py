"""论文性能测试脚本的公共统计、登录和CER工具。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import statistics

import requests


TESTS_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = TESTS_ROOT / "results"


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


def normalize_chinese(text):
    return re.sub(
        r"[\s，。！？、,.!?；;：:（）()\"'“”‘’\-]",
        "",
        str(text or "").strip().lower(),
    )


def edit_distance(reference, prediction):
    """返回字符级Levenshtein距离。"""
    reference = normalize_chinese(reference)
    prediction = normalize_chinese(prediction)
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

