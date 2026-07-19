"""Helpers for parsing list-like dataframe fields."""
from __future__ import annotations

import ast
from typing import Any, Iterable, List


def parse_list_field(value: Any) -> List[str]:
    """Parse a cell that may be a list, stringified list, or comma-separated string."""
    if value is None:
        return []
    if isinstance(value, float):  # NaN
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def top_overlap(a: Iterable[str], b: Iterable[str], limit: int = 8) -> List[str]:
    overlap = sorted(set(a) & set(b), key=str.lower)
    return overlap[:limit]
