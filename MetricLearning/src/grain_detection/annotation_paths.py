"""
Rutas de anotacion .seg con identidad absoluta.

El mismo stem (p.ej. C_QUITENSIS_0001_..._f1) aparece en train/val/test
pero apunta a placas distintas. El nombre del .seg NO se infiere ni se
comparte: split + placa + stem, o no hay ruta.

Clave canonica: <clase>/<split>__<SAMPLE_xxx>__<stem>.seg
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_ANNOTATION_ROOT = "data/annotations"
SAMPLE_RE = re.compile(r"^SAMPLE_\d+$", re.IGNORECASE)
SPLIT_NAMES = {"train", "val", "test"}
CANONICAL_SEG_RE = re.compile(
    r"^(train|val|test)__(SAMPLE_\d+)__(.+)$", re.IGNORECASE
)
PLATE_SEG_RE = re.compile(r"^(SAMPLE_\d+)__(.+)$", re.IGNORECASE)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class AnnotationIdentityError(ValueError):
    """Falta split o SAMPLE_xxx: no se puede nombrar el .seg sin ambiguedad."""


def _normalize_sample_name(name: str) -> str:
    parts = name.split("_", 1)
    if len(parts) == 2:
        return f"SAMPLE_{parts[1]}"
    return name.upper()


def split_id_for(img_path: Union[str, Path]) -> Optional[str]:
    """train|val|test del path logico (no resolve: el symlink sale de processed/)."""
    path = Path(img_path)
    for parent in path.parents:
        name = parent.name.lower()
        if name in SPLIT_NAMES:
            return name
    return None


def plate_id_for(img_path: Union[str, Path]) -> Optional[str]:
    """SAMPLE_xxx del directorio real de la imagen (sigue symlinks/junctions)."""
    path = Path(img_path)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    for parent in (resolved, *resolved.parents):
        if SAMPLE_RE.match(parent.name):
            return _normalize_sample_name(parent.name)
    return None


def resolved_image_path(img_path: Union[str, Path]) -> str:
    path = Path(img_path)
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def original_stem_for(img_path: Union[str, Path]) -> str:
    """Stem del archivo de imagen, sin prefijos de split/placa que ya lleve."""
    stem = Path(img_path).stem
    m = CANONICAL_SEG_RE.match(stem)
    if m:
        return m.group(3)
    m = PLATE_SEG_RE.match(stem)
    if m:
        return m.group(2)
    return stem


def identity_for(img_path: Union[str, Path]) -> Tuple[str, str, str]:
    """(split, plate, original_stem). Error si falta split o placa."""
    path = Path(img_path)
    split = split_id_for(path)
    plate = plate_id_for(path)
    stem = original_stem_for(path)
    if not split or not plate or not stem:
        raise AnnotationIdentityError(
            f"Identidad incompleta para {path}: split={split!r} plate={plate!r} stem={stem!r}"
        )
    return split, plate, stem


def annotation_stem_for(img_path: Union[str, Path]) -> str:
    split, plate, stem = identity_for(img_path)
    return f"{split}__{plate}__{stem}"


def class_name_for(img_path: Union[str, Path]) -> str:
    return Path(img_path).parent.name


def identity_params_for(img_path: Union[str, Path]) -> Dict[str, str]:
    split, plate, stem = identity_for(img_path)
    return {
        "split": split,
        "plate": plate,
        "stem": stem,
        "source": resolved_image_path(img_path),
    }


def seg_path_for(
    img_path: Union[str, Path],
    annotation_root: Union[str, Path] = DEFAULT_ANNOTATION_ROOT,
) -> Path:
    """Ruta canonica: annotation_root/<clase>/<split>__<SAMPLE>__<stem>.seg."""
    path = Path(img_path)
    root = Path(annotation_root)
    return root / class_name_for(path) / f"{annotation_stem_for(path)}.seg"


def find_existing_seg_path(
    img_path: Union[str, Path],
    annotation_root: Union[str, Path] = DEFAULT_ANNOTATION_ROOT,
) -> Optional[Path]:
    """Solo la ruta canonica split__SAMPLE__stem. Sin fallback ambiguo."""
    try:
        canonical = seg_path_for(img_path, annotation_root)
    except AnnotationIdentityError:
        logger.warning("[SegPath] sin identidad absoluta: %s", img_path)
        return None
    return canonical if canonical.exists() else None


def parse_dimensions(meta_dims: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parsea 'WIDTHxHEIGHT' del header .seg → (w, h)."""
    if not meta_dims:
        return None
    text = str(meta_dims).lower().replace(" ", "")
    if "x" not in text:
        return None
    left, right = text.split("x", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def seg_matches_image(
    seg_path: Union[str, Path],
    img_path: Union[str, Path],
    image_wh: Optional[Tuple[int, int]] = None,
) -> bool:
    """True si metadata del .seg declara la misma identidad que la imagen."""
    from .seg_format import SegFileReader

    try:
        split, plate, _stem = identity_for(img_path)
    except AnnotationIdentityError:
        return False

    result = SegFileReader.read(str(seg_path))
    meta = result.get("metadata", {})
    meta_plate = meta.get("plate")
    meta_split = meta.get("split")
    if meta_plate and str(meta_plate).upper() != plate.upper():
        return False
    if meta_split and str(meta_split).lower() != split.lower():
        return False

    source = meta.get("source") or meta.get("image_name")
    if source:
        resolved = resolved_image_path(img_path)
        src_name = Path(str(source)).name
        img_name = Path(img_path).name
        if Path(str(source)).is_absolute() or "SAMPLE_" in str(source).upper():
            if Path(str(source)).name != Path(resolved).name and src_name != img_name:
                return False

    dims = parse_dimensions(meta.get("dimensions"))
    if dims and image_wh:
        w, h = image_wh
        mw, mh = dims
        if (mw, mh) != (w, h):
            return False
    return True


def migrate_legacy_annotations(
    annotation_root: Union[str, Path] = DEFAULT_ANNOTATION_ROOT,
    processed_roots: Optional[List[Union[str, Path]]] = None,
    prefer_split_order: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Renombra .seg al nombre absoluto solo cuando hay UN dueno.

    - SAMPLE_xxx__stem.seg → split__SAMPLE_xxx__stem.seg si exactamente
      una imagen de esa clase+placa+stem.
    - stem.seg suelto: solo si exactamente una imagen de esa clase+stem.
    Nunca asigna colisiones a train por defecto.
    """
    del prefer_split_order  # no se usa: no hay desempate
    root = Path(annotation_root)
    stats = {
        "renamed": 0,
        "skipped_canonical": 0,
        "ambiguous_kept": 0,
        "errors": 0,
    }
    if not root.exists():
        return stats

    if processed_roots is None:
        processed_roots = [
            "data/processed/train",
            "data/processed/val",
            "data/processed/test",
        ]

    by_plate_stem: Dict[Tuple[str, str, str], List[Tuple[str, Path]]] = {}
    by_stem: Dict[Tuple[str, str], List[Tuple[str, Path, Optional[str]]]] = {}
    for processed in processed_roots:
        p = Path(processed)
        if not p.exists():
            continue
        split_name = p.name.lower()
        if split_name not in SPLIT_NAMES:
            continue
        for img in p.rglob("*"):
            if img.suffix.lower() not in IMAGE_EXTENSIONS or not img.is_file():
                continue
            cls = img.parent.name
            stem = original_stem_for(img)
            plate = plate_id_for(img)
            by_stem.setdefault((cls, stem), []).append((split_name, img, plate))
            if plate:
                by_plate_stem.setdefault((cls, plate, stem), []).append((split_name, img))

    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for seg in sorted(class_dir.glob("*.seg")):
            name = seg.stem
            if CANONICAL_SEG_RE.match(name):
                stats["skipped_canonical"] += 1
                continue

            dest: Optional[Path] = None
            plate_m = PLATE_SEG_RE.match(name)
            if plate_m:
                plate = _normalize_sample_name(plate_m.group(1))
                stem = plate_m.group(2)
                owners = by_plate_stem.get((class_dir.name, plate, stem), [])
                if len(owners) == 1:
                    dest = class_dir / f"{owners[0][0]}__{plate}__{stem}.seg"
                else:
                    stats["ambiguous_kept"] += 1
                    logger.warning(
                        "[SegMigrate] %s no se mueve: %d duenos para %s/%s/%s",
                        seg.name, len(owners), class_dir.name, plate, stem,
                    )
                    continue
            else:
                owners = by_stem.get((class_dir.name, name), [])
                unique = [(s, img, pl) for s, img, pl in owners if pl]
                if len(unique) == 1:
                    split, _img, plate = unique[0]
                    dest = class_dir / f"{split}__{plate}__{name}.seg"
                else:
                    stats["ambiguous_kept"] += 1
                    logger.warning(
                        "[SegMigrate] %s ambiguo (%d imagenes). No se asigna.",
                        seg.name, len(owners),
                    )
                    continue

            if dest is None:
                continue
            if dest.exists():
                stats["skipped_canonical"] += 1
                continue
            try:
                shutil.move(str(seg), str(dest))
                stats["renamed"] += 1
                logger.info("[SegMigrate] %s -> %s", seg.name, dest.name)
            except OSError as exc:
                stats["errors"] += 1
                logger.error("[SegMigrate] Error moviendo %s: %s", name, exc)

    return stats
