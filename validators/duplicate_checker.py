from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha1()
    file_size = path.stat().st_size
    digest.update(str(file_size).encode("ascii"))

    with path.open("rb") as handle:
        head = handle.read(1024 * 64)
        digest.update(head)
        if file_size > 1024 * 128:
            handle.seek(max(file_size - (1024 * 64), 0))
        tail = handle.read(1024 * 64)
        digest.update(tail)

    return digest.hexdigest()


def find_duplicate_groups(paths: Iterable[Path]) -> List[List[Path]]:
    groups: DefaultDict[str, List[Path]] = defaultdict(list)
    for path in paths:
        groups[_file_fingerprint(path)].append(path)
    return [sorted(group) for group in groups.values() if len(group) > 1]
