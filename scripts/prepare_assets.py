"""Fetch/build the two assets every stage needs, then sanity-check the victim.

1. ``assets/scene_photo.jpg``    -- a real photo, used only as a segmentation
   source (no environment renders this photo itself).
2. ``assets/target_sprite.png``  -- a real, **background-removed** cutout of
   one object (instance-segmentation mask as the alpha channel, not a
   bounding-box rectangle). Every stage's ``World`` uses this exact sprite as
   its ``target`` object, so all three stages attack the same real object,
   correctly shaped -- an attack can only ever occlude it, never blend into it.

If the machine is offline we fall back to a synthetic scene plus the
ColorBlobVictim, so the pipeline still runs end to end (with clearly weaker
scientific value -- the script says so).

    uv run python scripts/prepare_assets.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config, resolve  # noqa: E402
from rendering.image_renderer import load_rgb  # noqa: E402
from rendering.renderer_2d import make_background  # noqa: E402

PHOTO_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
]

SEG_WEIGHTS = "yolov8n-seg.pt"


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
    """Offline fallback source photo -- only ever fed to the segmenter."""
    print("[assets] building a synthetic scene instead (offline fallback)")
    frame = make_background(size, size, seed=1).convert("RGB")
    slab = Image.new("RGB", (130, 280), (0, 200, 0))
    frame.paste(slab, (size // 2 - 65, size // 2 - 100))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.save(dest)


def segment_sprite(photo: Path, sprite_path: Path, cfg: dict) -> bool:
    """Cut the victim's most confident target out of the photo by its own
    silhouette, using instance segmentation -- not a bounding-box rectangle.

    A rectangular crop would include background pixels inside the box, which
    is not what a real, physically-cut board/sticker of "the object" would
    ever look like. The segmentation mask becomes the alpha channel instead,
    so the sprite's own opaque pixels are exactly the object's own shape.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        print(f"[assets] ultralytics unavailable: {exc}")
        return False

    try:
        model = YOLO(SEG_WEIGHTS)
    except Exception as exc:  # pragma: no cover - typically a download failure
        print(f"[assets] could not load segmentation weights {SEG_WEIGHTS}: {exc}")
        return False

    image = load_rgb(photo)
    target_class = cfg["victim"].get("target_class", "person")
    results = model.predict(image[:, :, ::-1], conf=0.2, verbose=False)[0]
    if results.masks is None or len(results.boxes) == 0:
        print("[assets] segmentation found nothing in the photo")
        return False

    names = model.names
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    matches = [i for i, c in enumerate(cls_ids) if names.get(int(c), "") == target_class]
    best_idx = int(max(matches, key=lambda i: confs[i])) if matches else int(confs.argmax())
    best_conf = float(confs[best_idx])
    best_cls = names.get(int(cls_ids[best_idx]), str(cls_ids[best_idx]))

    mask = results.masks.data[best_idx].cpu().numpy()  # (Hm, Wm) in [0, 1], model resolution
    mask_full = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )
    )

    x1, y1, x2, y2 = (int(round(v)) for v in results.boxes.xyxy[best_idx].tolist())
    pad = 4
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)

    crop_rgb = image[y1:y2, x1:x2]
    crop_alpha = mask_full[y1:y2, x1:x2]
    rgba = np.dstack([crop_rgb, crop_alpha])

    sprite_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(sprite_path)
    opaque_frac = float((crop_alpha > 127).mean())
    print(
        f"[assets] sprite = {best_cls} @ conf {best_conf:.3f}, "
        f"{crop_rgb.shape[1]}x{crop_rgb.shape[0]}, "
        f"{opaque_frac:.0%} of the crop is the real silhouette -> {sprite_path}"
    )
    return True


def synthesize_sprite(sprite_path: Path) -> None:
    """Offline fallback: an oval alpha-matte cutout, not a rectangle -- so
    even the offline path exercises a non-rectangular sprite boundary."""
    print("[assets] building a synthetic cutout instead (offline fallback)")
    size = (130, 280)
    sprite = Image.new("RGBA", size, (0, 200, 0, 0))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([6, 6, size[0] - 6, size[1] - 6], fill=255)
    sprite.putalpha(mask)
    sprite_path.parent.mkdir(parents=True, exist_ok=True)
    sprite.save(sprite_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="re-download / rebuild")
    args = parser.parse_args()

    cfg = load_config(args.config)
    photo = resolve(cfg["assets"]["source_photo"])
    sprite = resolve(cfg["assets"]["target_sprite"])

    if args.force or not photo.exists():
        if not download_photo(photo):
            synthesize_photo(photo)
    else:
        print(f"[assets] source photo already present: {photo}")

    if args.force or not sprite.exists():
        if not segment_sprite(photo, sprite, cfg):
            synthesize_sprite(sprite)
    else:
        print(f"[assets] sprite already present: {sprite}")

    print("\n[assets] done.")
    print(f"  source photo  : {photo}  ({'ok' if photo.exists() else 'MISSING'})")
    print(f"  target sprite : {sprite} ({'ok' if sprite.exists() else 'MISSING'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
