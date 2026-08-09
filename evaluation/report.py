"""Small helpers to turn summaries into console tables, JSON and Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

__all__ = ["COMPARISON_COLUMNS", "format_table", "markdown_table", "save_json"]

COMPARISON_COLUMNS: list[tuple[str, str, str]] = [
    ("label", "method", "{}"),
    ("n_episodes", "episodes", "{}"),
    ("attack_success_rate", "success rate", "{:.3f}"),
    ("mean_baseline_confidence", "conf (clean)", "{:.3f}"),
    ("mean_best_confidence", "conf (attacked)", "{:.3f}"),
    ("mean_confidence_drop", "conf drop", "{:.3f}"),
    ("best_confidence_found", "best conf found", "{:.3f}"),
    ("mean_reward", "mean reward", "{:.3f}"),
    ("mean_movement_cost", "move cost", "{:.2f}"),
    ("mean_episode_length", "ep length", "{:.1f}"),
]


def _cell(row: dict[str, Any], key: str, fmt: str) -> str:
    value = row.get(key, row.get("method", "") if key == "label" else "")
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def format_table(
    rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str, str]] | None = None
) -> str:
    columns = list(columns or COMPARISON_COLUMNS)
    header = [title for _, title, _ in columns]
    body = [[_cell(row, key, fmt) for key, _, fmt in columns] for row in rows]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
        for i in range(len(columns))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    rule = "  ".join("-" * w for w in widths)
    out = [line, rule]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body]
    return "\n".join(out)


def markdown_table(
    rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str, str]] | None = None
) -> str:
    columns = list(columns or COMPARISON_COLUMNS)
    header = "| " + " | ".join(title for _, title, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_cell(row, key, fmt) for key, _, fmt in columns) + " |" for row in rows
    ]
    return "\n".join([header, rule, *body])


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
