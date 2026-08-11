"""Evaluate the deployed Memory Lantern ASR endpoint on a FLEURS test subset.

The script deliberately calls the same private ASR endpoint used by the web app.
It needs only the standard library plus ``requests`` (already used by the app).
Audio archives and FLEURS TSV manifests live outside the repository, so they are
never confused with user-uploaded voice material.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import tarfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import requests


LANGUAGES = {
    "en_us": {
        "archive": "en_us_test_audio.tar.gz",
        "manifest": "en_us_test.tsv",
        "metric": "WER",
    },
    "cmn_hans_cn": {
        "archive": "cmn_hans_cn_test_audio.tar.gz",
        "manifest": "cmn_hans_cn_test.tsv",
        "metric": "CER",
    },
}


def read_manifest(path: Path, sample_size: int, seed: int) -> list[dict[str, str]]:
    """Sample distinct FLEURS sentence IDs deterministically."""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) < 4:
                raise ValueError(f"Invalid FLEURS row in {path}: {fields!r}")
            groups[fields[0]].append(
                {
                    "sentence_id": fields[0],
                    "audio_name": fields[1],
                    "raw_reference": fields[2],
                    "normalized_reference": fields[3],
                }
            )

    sentence_ids = sorted(groups)
    if sample_size > len(sentence_ids):
        raise ValueError(
            f"Requested {sample_size} distinct sentences, only {len(sentence_ids)} available"
        )
    sampled_ids = sorted(random.Random(seed).sample(sentence_ids, sample_size))
    # FLEURS has several speakers for some sentences. Pick the first filename so
    # that reruns select the identical recording without relying on TSV order.
    return [sorted(groups[sentence_id], key=lambda row: row["audio_name"])[0] for sentence_id in sampled_ids]


def character_tokens(text: str) -> list[str]:
    """Chinese CER convention: compare normalized non-space characters."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    return [char for char in normalized if char.isalnum()]


def word_tokens(text: str) -> list[str]:
    """English WER convention used here: lower-case lexical tokens only."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", normalized)


def alignment(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return substitutions, deletions, and insertions via Levenshtein backtrace."""

    rows, columns = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * columns for _ in range(rows)]
    trace = [[""] * columns for _ in range(rows)]
    for index in range(1, rows):
        cost[index][0] = index
        trace[index][0] = "D"
    for index in range(1, columns):
        cost[0][index] = index
        trace[0][index] = "I"

    for row in range(1, rows):
        for column in range(1, columns):
            candidates = [
                (cost[row - 1][column - 1] + (reference[row - 1] != hypothesis[column - 1]), "E" if reference[row - 1] == hypothesis[column - 1] else "S"),
                (cost[row - 1][column] + 1, "D"),
                (cost[row][column - 1] + 1, "I"),
            ]
            cost[row][column], trace[row][column] = min(candidates, key=lambda item: item[0])

    substitutions = deletions = insertions = 0
    row, column = len(reference), len(hypothesis)
    while row or column:
        operation = trace[row][column]
        if operation == "S":
            substitutions += 1
            row -= 1
            column -= 1
        elif operation == "E":
            row -= 1
            column -= 1
        elif operation == "D":
            deletions += 1
            row -= 1
        elif operation == "I":
            insertions += 1
            column -= 1
        else:
            raise RuntimeError("Invalid alignment backtrace")
    return substitutions, deletions, insertions


def call_asr(url: str, api_key: str, audio_name: str, audio: bytes, timeout: int) -> tuple[str, int]:
    started_at = time.perf_counter()
    response = requests.post(
        url.rstrip("/") + "/v1/transcriptions",
        headers={"X-API-Key": api_key},
        data={"language": "zh"},  # Production route sets zh, which selects auto LID remotely.
        files={"audio": (audio_name, io.BytesIO(audio), "audio/wav")},
        timeout=timeout,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    response.raise_for_status()
    transcript = str(response.json().get("text") or "").strip()
    if not transcript:
        raise RuntimeError(f"Empty ASR transcript for {audio_name}")
    return transcript, elapsed_ms


def evaluate_language(
    language: str,
    eval_root: Path,
    sample_size: int,
    seed: int,
    asr_url: str,
    api_key: str,
    timeout: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = LANGUAGES[language]
    records = read_manifest(eval_root / config["manifest"], sample_size, seed)
    tokenise = word_tokens if config["metric"] == "WER" else character_tokens
    archive_path = eval_root / config["archive"]
    records_by_audio = {record["audio_name"]: record for record in records}
    evaluated: list[dict[str, object]] = []

    with tarfile.open(archive_path, "r:gz") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        missing = sorted(set(records_by_audio) - set(members))
        if missing:
            raise RuntimeError(f"Missing audio members in {archive_path}: {missing[:3]}")

        for index, record in enumerate(records, start=1):
            member = members[record["audio_name"]]
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unable to read archive member: {member.name}")
            hypothesis, elapsed_ms = call_asr(
                asr_url, api_key, record["audio_name"], stream.read(), timeout
            )
            reference_tokens = tokenise(record["normalized_reference"])
            hypothesis_tokens = tokenise(hypothesis)
            substitutions, deletions, insertions = alignment(reference_tokens, hypothesis_tokens)
            evaluated.append(
                {
                    "language": language,
                    "metric": config["metric"],
                    **record,
                    "hypothesis": hypothesis,
                    "reference_units": len(reference_tokens),
                    "hypothesis_units": len(hypothesis_tokens),
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "asr_ms": elapsed_ms,
                }
            )
            print(f"{language} {index}/{len(records)} {record['audio_name']} {elapsed_ms}ms", flush=True)

    total_reference = sum(int(row["reference_units"]) for row in evaluated)
    substitutions = sum(int(row["substitutions"]) for row in evaluated)
    deletions = sum(int(row["deletions"]) for row in evaluated)
    insertions = sum(int(row["insertions"]) for row in evaluated)
    return (
        {
            "language": language,
            "metric": config["metric"],
            "utterances": len(evaluated),
            "reference_units": total_reference,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "error_rate": (substitutions + deletions + insertions) / total_reference,
            "mean_asr_ms": sum(int(row["asr_ms"]) for row in evaluated) / len(evaluated),
        },
        evaluated,
    )


def write_results(output_dir: Path, summary: Iterable[dict[str, object]], details: Iterable[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = list(summary)
    detail_rows = list(details)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "utterances.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--asr-url", default=os.getenv("ASR_URL", "http://127.0.0.1:15001"))
    parser.add_argument("--api-key", default=os.getenv("ASR_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=90)
    arguments = parser.parse_args()
    if not arguments.api_key:
        raise SystemExit("Missing ASR API key: use --api-key or ASR_API_KEY.")

    summary: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for language in ("cmn_hans_cn", "en_us"):
        language_summary, language_details = evaluate_language(
            language=language,
            eval_root=arguments.eval_root,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
            asr_url=arguments.asr_url,
            api_key=arguments.api_key,
            timeout=arguments.timeout,
        )
        summary.append(language_summary)
        details.extend(language_details)
    write_results(arguments.output_dir, summary, details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
