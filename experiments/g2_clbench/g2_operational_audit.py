#!/usr/bin/env python3
"""Build a public, content-free inventory of incomplete G2 run evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


AUDIT_VERSION = "g2-operational-audit-v1"
PATTERNS = {
    "tool_use_failed": re.compile(r"tool_use_failed", re.IGNORECASE),
    "json_validate_failed": re.compile(r"json_validate_failed", re.IGNORECASE),
    "empty_content": re.compile(
        r"empty content|LLM returned empty content", re.IGNORECASE
    ),
    "schema_or_parse_failure": re.compile(
        r"schema.{0,30}(?:invalid|fail)|(?:invalid|fail).{0,30}schema|"
        r"JSONDecodeError|ValidationError",
        re.IGNORECASE,
    ),
    "rate_limit_429": re.compile(
        r"\b429\b|Too Many Requests|rate[ -]?limit", re.IGNORECASE
    ),
    "http_200": re.compile(r"HTTP/\S+\s+200|200 OK", re.IGNORECASE),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def inventory_logs(private_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(private_dir.rglob("*.log")) if private_dir.exists() else []:
        text = path.read_text(encoding="utf-8", errors="replace")
        records.append(
            {
                "path": _safe_rel(path, private_dir),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "signal_counts": {
                    name: len(pattern.findall(text))
                    for name, pattern in PATTERNS.items()
                },
            }
        )
    return records


def _glob_count(root: Path, patterns: Iterable[str]) -> int:
    paths: set[Path] = set()
    if root.exists():
        for pattern in patterns:
            paths.update(path for path in root.rglob(pattern) if path.is_file())
    return len(paths)


def build_audit(results_dir: Path) -> dict[str, object]:
    private_dir = results_dir / "private"
    public_dir = results_dir / "public"
    logs = inventory_logs(private_dir)
    totals: Counter[str] = Counter()
    for record in logs:
        totals.update(record["signal_counts"])  # type: ignore[arg-type]

    completed_clean = _glob_count(private_dir, ("clean_run_*.json",))
    completed_counterfactual = _glob_count(
        private_dir, ("counterfactual_run_*.json",)
    )
    public_seed_summaries = _glob_count(public_dir, ("seed_*.json",))
    no_research_outcome = (
        completed_clean == 0
        and completed_counterfactual == 0
        and public_seed_summaries == 0
    )
    failure_signals = sum(
        totals[key]
        for key in (
            "tool_use_failed",
            "json_validate_failed",
            "empty_content",
            "schema_or_parse_failure",
            "rate_limit_429",
        )
    )
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": _utc_now(),
        "scope": "frozen_g2_groq_qwen_operational_evidence",
        "log_count": len(logs),
        "logs": logs,
        "aggregate_signal_counts": dict(sorted(totals.items())),
        "completed_artifacts": {
            "clean_trajectories": completed_clean,
            "counterfactual_trajectories": completed_counterfactual,
            "public_seed_summaries": public_seed_summaries,
        },
        "no_research_outcome": no_research_outcome,
        "operational_status": (
            "infeasible_no_g2_outcome"
            if no_research_outcome and failure_signals > 0
            else "manual_review_required"
        ),
        "privacy": "hashes_and_pattern_counts_only_no_log_content",
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results_dir / "public" / "g2_operational_infeasibility_audit.json"
    )
    audit = build_audit(results_dir)
    _atomic_json(output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["operational_status"] == "infeasible_no_g2_outcome" else 2


if __name__ == "__main__":
    raise SystemExit(main())
