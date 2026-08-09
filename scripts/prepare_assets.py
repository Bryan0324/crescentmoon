"""Fetch/build the two assets every stage needs, then sanity-check the victim.

1. ``assets/scene_photo.jpg``  -- the Stage 1 photo.
2. ``assets/target_sprite.png`` -- an RGBA cutout of one detected object, used
   as the target that the 2D/3D worlds place in front of the camera.

The sprite is produced *by the victim itself*: we run YOLO on the photo, take
its most confident ``person``, and crop that box.  If the machine is offline we
fall back to a synthetic scene plus the ColorBlobVictim, so the pipeline still
runs end to end (with clearly weaker scientific value -- the script says so).

    uv run python scripts/prepare_assets.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config, resolve  # noqa: E402
from rendering.image_renderer import load_rgb  # noqa: E402
from rendering.renderer_2d import make_background  # noqa: E402

PHOTO_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
]


def download_photo(dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in PHOTO_URLS:
        try:
            print(f"[assets] downloading {url}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed, known URLs
            print(f"[assets] saved {dest}")
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[assets] failed ({exc.__class__.__name__}: {exc})")
    return False


def synthesize_photo(dest: Path, size: int = 512) -> None:
    """Offline fallback: a flat scene with one coloured slab as the 'object'."""
    print("[assets] building a synthetic scene instead (offline fallback)")
    frame = make_background(size, size, seed=1).convert("RGB")
    slab = Image.new("RGB", (130, 280), (0, 200, 0))
    frame.paste(slab, (size // 2 - 65, size // 2 - 100))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.save(dest)


def crop_sprite(photo: Path, sprite_path: Path, cfg: dict) -> bool:
    """Crop the victim's most confident target out of the photo."""
    try:
        from models.yolo_victim import YOLOVictim
    except Exception as exc:  # pragma: no cover
        print(f"[assets] ultralytics unavailable: {exc}")
        return False

    try:
        victim = YOLOVictim(
            weights=cfg["victim"]["weights"],
            conf_threshold=0.2,
            imgsz=640,
            device=cfg["victim"].get("device", "cpu"),
        )
    except Exception as exc:  # pragma: no cover - typically a download failure
        print(f"[assets] could not load YOLO weights: {exc}")
        return False

    image = load_rgb(photo)
    detections = victim.detect(image)
    target_class = cfg["victim"].get("target_class", "person")
    best = detections.best(target_class) or detections.best()
    if best is None:
        print("[assets] YOLO found nothing in the photo")
        return False

    x1, y1, x2, y2 = (int(round(v)) for v in best.bbox)
    pad_x = int(0.06 * (x2 - x1))
    pad_y = int(0.04 * (y2 - y1))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2 = min(image.shape[1], x2 + pad_x)
    y2 = min(image.shape[0], y2 + pad_y)

    crop = image[y1:y2, x1:x2]
    rgba = np.dstack([crop, np.full(crop.shape[:2] + (1,), 255, dtype=np.uint8)])
    sprite_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(sprite_path)
    print(
        f"[assets] sprite = {best.cls_name} @ conf {best.confidence:.3f}, "
        f"{crop.shape[1]}x{crop.shape[0]} -> {sprite_path}"
    )
    return True


def synthesize_sprite(sprite_path: Path) -> None:
    print("[assets] building a synthetic sprite instead (offline fallback)")
    sprite = Image.new("RGBA", (130, 280), (0, 200, 0, 255))
    sprite_path.parent.mkdir(parents=True, exist_ok=True)
    sprite.save(sprite_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="re-download / rebuild")
    args = parser.parse_args()

    cfg = load_config(args.config)
    photo = resolve(cfg["assets"]["photo"])
    sprite = resolve(cfg["assets"]["target_sprite"])

    if args.force or not photo.exists():
        if not download_photo(photo):
            synthesize_photo(photo)
    else:
        print(f"[assets] photo already present: {photo}")

    if args.force or not sprite.exists():
        if not crop_sprite(photo, sprite, cfg):
            synthesize_sprite(sprite)
    else:
        print(f"[assets] sprite already present: {sprite}")

    print("\n[assets] done.")
    print(f"  photo : {photo}  ({'ok' if photo.exists() else 'MISSING'})")
    print(f"  sprite: {sprite} ({'ok' if sprite.exists() else 'MISSING'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
