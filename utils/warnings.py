from __future__ import annotations

from typing import Iterable, List, Optional


def merge_warnings(*warning_groups: Optional[Iterable[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in warning_groups:
        if not group:
            continue
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged
