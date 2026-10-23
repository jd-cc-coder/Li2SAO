#!/usr/bin/env python3
"""Build markdown/csv/json tables from a ReTool AIME eval sweep (2024/2025/2026)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

_PASS_K_RE = re.compile(r"pass@(\d+)$")
_YEAR_RE = re.compile(r"aime(202[456])$")
_METRIC_YEAR_RE = re.compile(r"eval/aime(202[456])(?:-|$)")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pass_k_key(key: str) -> str | None:
    m = _PASS_K_RE.search(str(key))
    return f"pass@{m.group(1)}" if m else None


def _pass_k_sort_key(name: str) -> int:
    m = _PASS_K_RE.search(name)
    return int(m.group(1)) if m else 0


def _collect_pass_k(*sources: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src in sources:
        for key, val in src.items():
            name = _pass_k_key(key)
            if name is not None:
                out[name] = val
    return out


def _years_from_payload(payload: dict[str, Any]) -> list[str]:
    years: set[str] = set()
    for key in (payload.get("datasets") or {}):
        m = _YEAR_RE.search(str(key))
        if m:
            years.add(m.group(1))
    for key in (payload.get("metrics") or {}):
        m = _METRIC_YEAR_RE.match(str(key))
        if m:
            years.add(m.group(1))
    return sorted(years)


def _year_metrics(payload: dict[str, Any] | None, year: str) -> dict[str, Any]:
    if not payload:
        return {}
    metrics = payload.get("metrics") or {}
    datasets = payload.get("datasets") or {}
    aime = datasets.get(f"aime{year}") or {}
    pass_rate = aime.get("pass_rate") or {}
    mean_reward = aime.get("mean_reward")
    if mean_reward is None:
        mean_reward = metrics.get(f"eval/aime{year}")
    truncated = aime.get("truncated_ratio")
    if truncated is None:
        truncated = metrics.get(f"eval/aime{year}-truncated_ratio")
    return {
        **_collect_pass_k(pass_rate, {k: v for k, v in metrics.items() if f"aime{year}" in str(k)}),
        "mean_reward": mean_reward,
        "truncated_ratio": truncated,
        "sandbox_hit_ratio": metrics.get(f"eval/aime{year}-sandbox_hit_ratio"),
        "tool_call_ratio": metrics.get(f"eval/aime{year}-tool_call_ratio"),
        "n_samples": payload.get("n_samples_per_eval_prompt"),
    }


def _pass_k_headers(rows: list[dict[str, Any]]) -> list[str]:
    keys = {k for row in rows for k in row if _pass_k_key(k)}
    return sorted(keys, key=_pass_k_sort_key)


def _cell(row: dict[str, Any], key: str) -> str:
    if key == "checkpoint":
        return str(row.get("checkpoint", ""))
    if key in ("step", "n_samples"):
        return "" if row.get(key) is None else str(row[key])
    if key == "status":
        return str(row.get("status", ""))
    return _fmt(row.get(key))


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row, h) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, help="JSONL written by the sweep script")
    parser.add_argument("--out-dir", required=True, help="Eval JSON directory")
    parser.add_argument("--prefix", default="aime", help="Filename prefix for summary outputs")
    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    out_dir = Path(args.out_dir)
    if not jobs_path.is_file():
        print(f"ERROR: missing jobs file {jobs_path}", file=sys.stderr)
        return 1

    loaded: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    years: set[str] = set()
    for job in _read_jobs(jobs_path):
        tag = job.get("tag") or ""
        json_path = out_dir / f"{tag}_latest.json"
        payload = _load_json(json_path) if job.get("status") == "ok" else None
        if payload:
            years.update(_years_from_payload(payload))
        loaded.append((job, payload))
    year_list = sorted(years) or ["2024", "2025", "2026"]

    per_year_rows: dict[str, list[dict[str, Any]]] = {y: [] for y in year_list}
    combined_rows: list[dict[str, Any]] = []
    for job, payload in loaded:
        tag = job.get("tag") or ""
        json_path = out_dir / f"{tag}_latest.json"
        status = job.get("status")
        if json_path.is_file() is False and status == "ok":
            status = "missing_json"
        base = {
            "checkpoint": job.get("name"),
            "step": job.get("step"),
            "load": job.get("load"),
            "ckpt_step": job.get("ckpt_step"),
            "tag": tag,
            "status": status,
            "exit_code": job.get("exit_code"),
            "json_path": str(json_path) if json_path.is_file() else None,
            "n_samples": (payload or {}).get("n_samples_per_eval_prompt") if payload else None,
        }
        combined = dict(base)
        for year in year_list:
            metrics = _year_metrics(payload, year)
            per_year_rows[year].append({**base, **metrics})
            for key, val in metrics.items():
                if key == "n_samples":
                    continue
                combined[f"aime{year}_{key}"] = val
        combined_rows.append(combined)

    prefix = args.prefix
    md_parts = ["# ReTool AIME eval sweep (pass@k)\n"]
    for year in year_list:
        rows = per_year_rows[year]
        pass_headers = _pass_k_headers(rows)
        headers = [
            "checkpoint",
            "step",
            *pass_headers,
            "mean_reward",
            "truncated_ratio",
            "sandbox_hit_ratio",
            "n_samples",
            "status",
        ]
        md_parts.append(f"## AIME {year}\n")
        md_parts.append(_markdown_table(rows, headers))
        print(f"=== AIME {year} ===")
        print(_markdown_table(rows, headers), end="")

    md_path = out_dir / f"{prefix}_all_summary.md"
    csv_path = out_dir / f"{prefix}_all_summary.csv"
    json_path = out_dir / f"{prefix}_all_summary.json"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    fieldnames: list[str] = ["checkpoint", "step"]
    for year in year_list:
        pass_headers = _pass_k_headers(per_year_rows[year])
        fieldnames.extend(f"aime{year}_{h}" for h in pass_headers)
        fieldnames.extend(
            f"aime{year}_{k}"
            for k in ("mean_reward", "truncated_ratio", "sandbox_hit_ratio")
        )
    fieldnames.extend(["n_samples", "status", "tag", "json_path"])
    seen: set[str] = set()
    unique_fields = []
    for name in fieldnames:
        if name not in seen:
            seen.add(name)
            unique_fields.append(name)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unique_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)
    json_path.write_text(json.dumps(combined_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"summary markdown: {md_path}")
    print(f"summary csv:      {csv_path}")
    print(f"summary json:     {json_path}")
    return 0 if all(r.get("status") == "ok" for r in combined_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
