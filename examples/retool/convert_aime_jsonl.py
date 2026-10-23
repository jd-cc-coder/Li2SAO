"""Convert AIME jsonl into slime prompt/label format.

Accepts records with either:
  * already-converted ``prompt`` + ``label`` (copied through)
  * ``problem`` / ``question`` + ``answer`` / ``ground_truth`` / ``label``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _problem(rec: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if key == "prompt" and isinstance(val, list) and val:
            content = val[0].get("content") if isinstance(val[0], dict) else None
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError(f"no problem text in record keys={sorted(rec)}")


def _label(rec: dict[str, Any]) -> str:
    for key in ("label", "answer", "ground_truth"):
        if key in rec and rec[key] is not None and rec[key] != "":
            return str(rec[key]).strip()
    raise ValueError(f"no answer in record keys={sorted(rec)}")


def convert_record(rec: dict[str, Any]) -> dict[str, Any]:
    prompt = rec.get("prompt")
    if isinstance(prompt, list) and prompt and rec.get("label") is not None:
        return {"prompt": prompt, "label": str(rec["label"]).strip()}
    return {
        "prompt": [{"role": "user", "content": _problem(rec)}],
        "label": _label(rec),
    }


def convert_file(src: Path, dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{src}:{line_num} expected object")
            fout.write(json.dumps(convert_record(rec), ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise SystemExit(f"ERROR: no rows in {src}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    n = convert_file(args.src, args.dst)
    print(f"wrote {n} rows {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
