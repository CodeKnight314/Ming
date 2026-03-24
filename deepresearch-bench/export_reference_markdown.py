#!/usr/bin/env python3
"""Export deepresearch-bench/reference.jsonl to one markdown file per record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_markdown(record: dict) -> str:
    rid = record["id"]
    prompt = record.get("prompt", "")
    article = record.get("article", "")
    lines = [
        f"# Reference {rid}",
        "",
        "## Prompt",
        "",
        prompt.rstrip(),
        "",
        "## Article",
        "",
        article.rstrip(),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    here = Path(__file__).resolve().parent
    default_in = here / "reference.jsonl"
    default_out = here / "reference"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_in,
        help=f"Path to reference.jsonl (default: {default_in})",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Directory for id_N.md files (default: {default_out})",
    )
    args = p.parse_args()
    input_path: Path = args.input
    out_dir: Path = args.output_dir

    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"error: line {lineno}: invalid JSON: {e}", file=sys.stderr)
                return 1
            if "id" not in record:
                print(f"error: line {lineno}: missing 'id'", file=sys.stderr)
                return 1
            rid = record["id"]
            out_path = out_dir / f"id_{rid}.md"
            out_path.write_text(build_markdown(record), encoding="utf-8")
            count += 1

    print(f"wrote {count} file(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
