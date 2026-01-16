# -*- coding: utf-8-sig -*-
"""
OpenAI cost tracking utilities (token-based, estimated).

This module is intentionally simple and transparent for a university assignment:
- We log every OpenAI call with timestamp + operation + tokens + estimated cost.
- Costs are ESTIMATES based on a configurable pricing table.

IMPORTANT:
- Prices may change. Update `MODEL_PRICING_USD_PER_1M` in `config.py` if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import EVALUATION_DIR


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_append_jsonl(path: Path):
    """
    Append JSONL safely while keeping UTF-8 BOM only once.
    """
    ensure_dir(path)
    if not path.exists():
        return open(path, "w", encoding="utf-8-sig")
    return open(path, "a", encoding="utf-8")


def log_event(event: Dict[str, Any], log_path: Optional[Path] = None) -> None:
    log_path = log_path or (EVALUATION_DIR / "openai_cost_log.jsonl")
    event = dict(event)
    event.setdefault("ts_utc", now_iso())
    with _open_append_jsonl(Path(log_path)) as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Pricing:
    input_per_1m: float
    output_per_1m: float


def get_pricing_usd_per_1m(model: str, pricing_table: Dict[str, Dict[str, float]]) -> Optional[Pricing]:
    """
    Return pricing for an exact model name, if present.
    """
    if not model:
        return None
    entry = pricing_table.get(model)
    if not entry:
        return None
    try:
        return Pricing(float(entry["input"]), float(entry["output"]))
    except Exception:
        return None


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_table: Dict[str, Dict[str, float]],
) -> Optional[float]:
    p = get_pricing_usd_per_1m(model, pricing_table)
    if p is None:
        return None
    return (prompt_tokens / 1_000_000.0) * p.input_per_1m + (completion_tokens / 1_000_000.0) * p.output_per_1m

