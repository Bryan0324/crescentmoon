"""configs/objects.py::ObjectLibrary -- how every stage finds its objects."""

from __future__ import annotations

import json

import pytest

from configs.objects import ObjectAsset, ObjectLibrary, load_library


def _library() -> ObjectLibrary:
    return ObjectLibrary(
        [
            ObjectAsset(id="person_0", cls_name="person", path="p0.png", confidence=0.88, source="bus"),
            ObjectAsset(id="person_1", cls_name="person", path="p1.png", confidence=0.60, source="bus"),
            ObjectAsset(id="bus_0", cls_name="bus", path="b0.png", confidence=0.84, source="bus"),
        ]
    )


def test_resolve_by_exact_id():
    asset = _library().resolve("person_1")
    assert asset.id == "person_1"


def test_resolve_by_class_picks_most_confident_instance():
    asset = _library().resolve("person")
    assert asset.id == "person_0"  # 0.88 > 0.60


def test_resolve_unknown_ref_raises():
    with pytest.raises(KeyError):
        _library().resolve("car")


def test_of_class_and_classes():
    library = _library()
    assert {a.id for a in library.of_class("person")} == {"person_0", "person_1"}
    assert library.classes() == ["bus", "person"]


def test_len_and_ids():
    library = _library()
    assert len(library) == 3
    assert set(library.ids()) == {"person_0", "person_1", "bus_0"}


def test_load_library_missing_index_is_empty(tmp_path):
    library = load_library(tmp_path / "missing.json", tmp_path / "objects")
    assert len(library) == 0


def test_load_library_reads_index_and_resolves_paths(tmp_path):
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    index = tmp_path / "objects.json"
    index.write_text(
        json.dumps(
            [{"id": "person_0", "class": "person", "file": "person_0.png", "confidence": 0.9, "source": "bus"}]
        ),
        encoding="utf-8",
    )
    library = load_library(index, objects_dir)
    asset = library.get("person_0")
    assert asset.path == objects_dir / "person_0.png"
    assert asset.cls_name == "person"
