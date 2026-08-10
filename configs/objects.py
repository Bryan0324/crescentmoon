"""The object library: every stage's scene is built from entries in
``assets/objects.json``, each a real, background-removed cutout of one
photographed instance (see ``scripts/prepare_assets.py``).

Stage 1 attacks exactly one of these objects. Stage 2/3 compose several --
a target plus one or more obstacles -- into one scene before the attack
runs. Both read from this same library, by object id (``"person_0"``) or by
class name (``"person"``, resolving to that class's most confident instance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ObjectAsset", "ObjectLibrary", "load_library"]


@dataclass(frozen=True)
class ObjectAsset:
    id: str
    cls_name: str
    path: Path
    confidence: float
    source: str


class ObjectLibrary:
    def __init__(self, assets: list[ObjectAsset]) -> None:
        self._by_id = {a.id: a for a in assets}

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> list[str]:
        return list(self._by_id)

    def of_class(self, cls_name: str) -> list[ObjectAsset]:
        return [a for a in self._by_id.values() if a.cls_name == cls_name]

    def classes(self) -> list[str]:
        return sorted({a.cls_name for a in self._by_id.values()})

    def get(self, object_id: str) -> ObjectAsset:
        if object_id not in self._by_id:
            raise KeyError(f"unknown object id {object_id!r}; available: {sorted(self._by_id)}")
        return self._by_id[object_id]

    def resolve(self, ref: str) -> ObjectAsset:
        """``ref`` is either an exact object id (``"person_0"``) or a class
        name (``"person"``), in which case its most confident instance is used."""
        if ref in self._by_id:
            return self._by_id[ref]
        matches = self.of_class(ref)
        if not matches:
            raise KeyError(f"no object with id or class {ref!r}; available: {sorted(self._by_id)}")
        return max(matches, key=lambda a: a.confidence)


def load_library(index_path: str | Path, objects_dir: str | Path) -> ObjectLibrary:
    index_path = Path(index_path)
    objects_dir = Path(objects_dir)
    if not index_path.exists():
        return ObjectLibrary([])
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    assets = [
        ObjectAsset(
            id=e["id"],
            cls_name=e["class"],
            path=objects_dir / e["file"],
            confidence=float(e["confidence"]),
            source=e["source"],
        )
        for e in entries
    ]
    return ObjectLibrary(assets)
