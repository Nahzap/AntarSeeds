"""Tests de rutas .seg con identidad absoluta (split + placa + stem)."""

import pytest
from pathlib import Path

from src.grain_detection.annotation_paths import (
    AnnotationIdentityError,
    annotation_stem_for,
    find_existing_seg_path,
    migrate_legacy_annotations,
    plate_id_for,
    seg_path_for,
)


def _link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(src)
    except OSError:
        dest.write_bytes(src.read_bytes())


def test_plate_id_from_resolved_sample_dir(tmp_path):
    sample = tmp_path / "SAMPLE_005"
    sample.mkdir()
    img = sample / "C_QUITENSIS_0001_f1.png"
    img.write_bytes(b"x")
    link = tmp_path / "processed" / "train" / "C_Quitensis" / img.name
    _link_or_copy(img, link)
    assert plate_id_for(link) == "SAMPLE_005"
    assert annotation_stem_for(link) == "train__SAMPLE_005__C_QUITENSIS_0001_f1"
    assert seg_path_for(link, tmp_path / "annotations").name == (
        "train__SAMPLE_005__C_QUITENSIS_0001_f1.seg"
    )


def test_missing_plate_or_split_raises(tmp_path):
    img = tmp_path / "C_Quitensis" / "no_identity.png"
    img.parent.mkdir()
    img.write_bytes(b"x")
    with pytest.raises(AnnotationIdentityError):
        annotation_stem_for(img)


def test_same_stem_different_plates_get_distinct_seg_paths(tmp_path):
    ann = tmp_path / "annotations"
    for plate in ("SAMPLE_005", "SAMPLE_006"):
        src = tmp_path / plate / "same_stem_f1.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"img")
        split = "train" if plate.endswith("5") else "val"
        link = tmp_path / "processed" / split / "C_Quitensis" / src.name
        _link_or_copy(src, link)

    train_img = tmp_path / "processed" / "train" / "C_Quitensis" / "same_stem_f1.png"
    val_img = tmp_path / "processed" / "val" / "C_Quitensis" / "same_stem_f1.png"
    assert seg_path_for(train_img, ann) != seg_path_for(val_img, ann)
    assert seg_path_for(train_img, ann).name.startswith("train__SAMPLE_005__")
    assert seg_path_for(val_img, ann).name.startswith("val__SAMPLE_006__")


def test_find_existing_does_not_cross_plates_after_canonical(tmp_path):
    ann = tmp_path / "annotations" / "C_Quitensis"
    ann.mkdir(parents=True)
    (ann / "train__SAMPLE_005__stem_f1.seg").write_text(
        "# Plate: SAMPLE_005\n# Split: train\n", encoding="utf-8"
    )

    for plate, split in (("SAMPLE_005", "train"), ("SAMPLE_006", "val")):
        src = tmp_path / plate / "stem_f1.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"img")
        link = tmp_path / "processed" / split / "C_Quitensis" / "stem_f1.png"
        _link_or_copy(src, link)

    train_img = tmp_path / "processed" / "train" / "C_Quitensis" / "stem_f1.png"
    val_img = tmp_path / "processed" / "val" / "C_Quitensis" / "stem_f1.png"
    root = tmp_path / "annotations"
    assert find_existing_seg_path(train_img, root) is not None
    assert find_existing_seg_path(val_img, root) is None


def test_migrate_does_not_guess_collisions(tmp_path):
    ann_root = tmp_path / "annotations"
    cls = ann_root / "C_Quitensis"
    cls.mkdir(parents=True)
    legacy = cls / "stem_f1.seg"
    legacy.write_text("# Imagen: stem_f1.png\n", encoding="utf-8")

    processed = []
    for plate, split in (("SAMPLE_005", "train"), ("SAMPLE_006", "val")):
        src = tmp_path / plate / "stem_f1.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"img")
        link = tmp_path / "processed" / split / "C_Quitensis" / "stem_f1.png"
        _link_or_copy(src, link)
        processed.append(tmp_path / "processed" / split)

    stats = migrate_legacy_annotations(ann_root, processed)
    assert stats["renamed"] == 0
    assert stats["ambiguous_kept"] == 1
    assert legacy.exists()


def test_migrate_unique_plate_file(tmp_path):
    ann_root = tmp_path / "annotations"
    cls = ann_root / "C_Quitensis"
    cls.mkdir(parents=True)
    old = cls / "SAMPLE_005__stem_f1.seg"
    old.write_text("# Plate: SAMPLE_005\n", encoding="utf-8")

    src = tmp_path / "SAMPLE_005" / "stem_f1.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"img")
    link = tmp_path / "processed" / "train" / "C_Quitensis" / "stem_f1.png"
    _link_or_copy(src, link)

    stats = migrate_legacy_annotations(ann_root, [tmp_path / "processed" / "train"])
    assert stats["renamed"] == 1
    assert not old.exists()
    assert (cls / "train__SAMPLE_005__stem_f1.seg").exists()
