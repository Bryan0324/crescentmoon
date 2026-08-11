"""Build the object library every stage draws from.

1. Download a couple of real source photos (or synthesize offline fallbacks).
2. Run YOLOv8-seg on each and cut out every confident instance by its own
   silhouette -- the segmentation mask becomes the alpha channel, not a
   bounding-box rectangle.
3. Reject instances that aren't *complete* objects:
   - cropped by the photo frame on any side (e.g. a waist-up portrait has no
     legs to cut out, or an outstretched arm runs off the edge) -- caught by
     a border check on all four sides, not just top/bottom;
   - heavily occluded overall, down to a sliver -- these tend to score low
     confidence and are caught by the confidence floor;
   - occluded by something *in front of* it mid-frame (e.g. pedestrians
     standing in front of a bus leave person-shaped notches in the bus's own
     mask). We tried a geometric detector for this (an enclosed-hole check
     cross-referenced against every other instance's mask) and it did not
     work: on bus.jpg it scored a visibly notched bus *lower* than people we
     confirmed by eye were complete -- gaps between crossed arms and the
     torso look the same to that math as a real occlusion notch. Automating
     "does this look like one whole object" reliably needs actual image
     understanding, which is out of scope for a small asset-prep script,
     so this one is instead caught by MANUAL_EXCLUDE below, chosen by
     looking at the crops -- honest for a library this small.
   A cutout that is only part of an object is not something a physical
   attacker could hold up or place -- it would not read as one object.
4. Save each surviving cutout under ``assets/objects/<class>_<n>.png`` and
   record it in ``assets/objects.json``.
5. Also save each source photo's own backdrop under
   ``assets/backgrounds/<source>.png``, with every detected instance (kept or
   rejected) painted out via inpainting -- a real photo background for Stage 2
   to compose cutouts onto, instead of a synthetic sky/ground gradient.

Stage 1 attacks exactly one of these objects (see ``stage1.target_object`` in
configs/default.yaml). Stage 2/3 compose several -- a target plus one or more
obstacles -- into one scene, all genuine cutouts, before the attack runs.

    uv run python scripts/prepare_assets.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)

from configs.loader import load_config, resolve  # noqa: E402
from rendering.image_renderer import load_rgb  # noqa: E402

SEG_WEIGHTS = "yolov8n-seg.pt"

# Two well-known Ultralytics quickstart demo photos -- used only to build a
# small, varied object library (several people, one bus). Neither photo is
# ever rendered as-is by any environment; only the cutouts extracted from
# them are.
SOURCE_PHOTOS: dict[str, list[str]] = {
    "bus": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
    ],
    "zidane": [
        "https://ultralytics.com/images/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg",
    ],
}

MIN_CONFIDENCE = 0.6
MAX_INSTANCES_PER_SOURCE = 6
#: Reject an instance whose mask reaches this close to *any* edge of the
#: photo -- it means the object is cut off by the frame itself (a waist-up
#: portrait has no legs to cut out; an outstretched arm can run off the side),
#: not a segmentation failure. There is nothing to recover: that part of the
#: object was never in the photo.
BORDER_MARGIN = 0.02
#: (source, class, confidence) of instances that pass every automatic check
#: but were found -- by looking at the actual crop -- to have a real defect
#: geometry can't reliably catch. Currently: bus.jpg's bus is visibly notched
#: by the three pedestrians standing in front of it (see the module docstring,
#: check 3). Keyed loosely (rounded confidence) since re-running segmentation
#: reproduces the same detections deterministically.
MANUAL_EXCLUDE: set[tuple[str, str, float]] = {("bus", "bus", 0.838)}


def download(dest: Path, urls: list[str]) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"[assets] downloading {url}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed, known URLs
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[assets] failed ({exc.__class__.__name__}: {exc})")
    return False


def build_background_plate(photo: Path, model, remove_classes: set[str]) -> Image.Image:
    """The photo's own backdrop, with every instance of ``remove_classes``
    painted out.

    Stage 2 composes real object cutouts onto this instead of a synthetic sky
    gradient -- a real photo, but without instances of the *target's own
    class* left behind in it. Two reasons to remove exactly those and nothing
    else: (1) the target/obstacle cutouts are themselves instances of this
    photo, so leaving the class in would show an object standing right next
    to its own original self once repositioned; (2) any leftover same-class
    instance would compete with our composited target when the environment
    picks "the" target by highest confidence at reset. A large,
    unrelated-class object (e.g. a bus behind our people) is left alone --
    inpainting a hole that big looks far worse than a bus in the background
    does, and it isn't the class anything gets matched against.
    """
    import cv2

    image = load_rgb(photo)
    results = model.predict(image[:, :, ::-1], conf=MIN_CONFIDENCE, verbose=False)[0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if results.masks is not None:
        names = model.names
        for i, inst_mask in enumerate(results.masks.data.cpu().numpy()):
            cls_name = names.get(int(results.boxes.cls[i]), "")
            if cls_name not in remove_classes:
                continue
            resized = np.asarray(
                Image.fromarray((inst_mask * 255).astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0]), Image.BILINEAR
                )
            )
            mask = np.maximum(mask, resized)
    if not np.any(mask):
        return Image.fromarray(image)
    # Dilate so the inpaint also covers each mask's antialiased edge halo.
    mask = cv2.dilate((mask > 32).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
    inpainted = cv2.inpaint(image, mask, 9, cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def synthesize_source(dest: Path, seed: int) -> None:
    """Offline fallback source photo -- only ever fed to the segmenter."""
    from rendering.renderer_2d import make_background

    print(f"[assets] building a synthetic source photo instead (offline fallback) -> {dest}")
    frame = make_background(512, 512, seed=seed).convert("RGB")
    colour = (0, 200, 0) if seed % 2 else (60, 90, 200)
    frame.paste(Image.new("RGB", (130, 280), colour), (512 // 2 - 65, 512 // 2 - 100))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.save(dest)


def segment_all_instances(photo: Path, source_name: str, model) -> list[dict]:
    """Every confident, complete instance in ``photo``, cut out by its own
    silhouette."""
    image = load_rgb(photo)
    results = model.predict(image[:, :, ::-1], conf=MIN_CONFIDENCE, verbose=False)[0]
    if results.masks is None or len(results.boxes) == 0:
        return []

    names = model.names
    confs = results.boxes.conf.cpu().numpy()
    height, width = image.shape[0], image.shape[1]
    order = np.argsort(-confs)

    out = []
    for idx in order:
        if len(out) >= MAX_INSTANCES_PER_SOURCE:
            break
        idx = int(idx)
        cls_name = names.get(int(results.boxes.cls[idx]), str(idx))
        conf = float(confs[idx])
        x1, y1, x2, y2 = results.boxes.xyxy[idx].tolist()

        h_margin = BORDER_MARGIN * width
        v_margin = BORDER_MARGIN * height
        if x1 <= h_margin or x2 >= width - h_margin or y1 <= v_margin or y2 >= height - v_margin:
            print(f"[assets]   skipping {cls_name} (conf={conf:.3f}): cropped by the photo frame, not a complete object")
            continue

        if (source_name, cls_name, round(conf, 3)) in MANUAL_EXCLUDE:
            print(f"[assets]   skipping {cls_name} (conf={conf:.3f}): manually excluded, see MANUAL_EXCLUDE")
            continue

        mask = results.masks.data[idx].cpu().numpy()
        mask_full = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
        )
        x1i, y1i, x2i, y2i = (int(round(v)) for v in (x1, y1, x2, y2))
        pad = 4
        x1i, y1i = max(0, x1i - pad), max(0, y1i - pad)
        x2i, y2i = min(width, x2i + pad), min(height, y2i + pad)
        crop_rgb = image[y1i:y2i, x1i:x2i]
        crop_alpha = mask_full[y1i:y2i, x1i:x2i]
        rgba = np.dstack([crop_rgb, crop_alpha])
        out.append({"class": cls_name, "confidence": conf, "image": Image.fromarray(rgba, mode="RGBA")})
    return out


def build_offline_library(objects_dir: Path, index_path: Path, backgrounds_dir: Path) -> list[dict]:
    """Two non-rectangular synthetic cutouts of different 'classes' -- enough
    for Stage 2/3 to compose a target + an obstacle with no network access."""
    print("[assets] building a synthetic object library instead (offline fallback)")
    objects_dir.mkdir(parents=True, exist_ok=True)
    backgrounds_dir.mkdir(parents=True, exist_ok=True)
    from rendering.renderer_2d import make_background

    make_background(512, 512, seed=0).convert("RGB").save(backgrounds_dir / "offline.png")
    shapes = [("target", (0, 200, 0), "ellipse"), ("obstacle", (90, 95, 105), "rectangle")]

    entries = []
    for cls, colour, shape in shapes:
        size = (130, 280)
        sprite = Image.new("RGBA", size, (*colour, 0))
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        if shape == "ellipse":
            draw.ellipse([6, 6, size[0] - 6, size[1] - 6], fill=255)
        else:
            draw.rounded_rectangle([6, 6, size[0] - 6, size[1] - 6], radius=18, fill=255)
        sprite.putalpha(mask)
        out_path = objects_dir / f"{cls}_0.png"
        sprite.save(out_path)
        entries.append(
            {"id": f"{cls}_0", "class": cls, "file": out_path.name, "confidence": 1.0, "source": "offline"}
        )
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def build_library(cfg: dict, force: bool) -> list[dict]:
    objects_dir = resolve(cfg["assets"]["objects_dir"])
    index_path = resolve(cfg["assets"]["objects_index"])
    backgrounds_dir = resolve(cfg["assets"].get("backgrounds_dir", "assets/backgrounds"))
    objects_dir.mkdir(parents=True, exist_ok=True)
    backgrounds_dir.mkdir(parents=True, exist_ok=True)

    if index_path.exists() and not force:
        entries = json.loads(index_path.read_text(encoding="utf-8"))
        have = {p.stem for p in backgrounds_dir.glob("*.png")}
        need = {e["source"] for e in entries}
        if need <= have:
            print(f"[assets] library already present: {index_path}")
            return entries
        print(f"[assets] object library present but background plate(s) missing ({need - have}) -- rebuilding")

    try:
        from ultralytics import YOLO

        model = YOLO(SEG_WEIGHTS)
    except Exception as exc:  # pragma: no cover - typically a download failure
        print(f"[assets] segmentation unavailable ({exc})")
        return build_offline_library(objects_dir, index_path, backgrounds_dir)

    entries: list[dict] = []
    counters: dict[str, int] = {}
    source_dir = resolve(cfg["assets"]["source_dir"])
    remove_classes = {cfg.get("victim", {}).get("target_class", "person")}

    for name, urls in SOURCE_PHOTOS.items():
        photo = source_dir / f"{name}.jpg"
        if not photo.exists():
            if not download(photo, urls):
                synthesize_source(photo, seed=hash(name) % 97)

        plate = build_background_plate(photo, model, remove_classes)
        plate.save(backgrounds_dir / f"{name}.png")
        print(f"[assets] background plate <- {name}.jpg (detected instances inpainted out)")

        for inst in segment_all_instances(photo, name, model):
            cls = inst["class"]
            counters[cls] = counters.get(cls, 0) + 1
            obj_id = f"{cls}_{counters[cls] - 1}"
            out_path = objects_dir / f"{obj_id}.png"
            inst["image"].save(out_path)
            entries.append(
                {
                    "id": obj_id,
                    "class": cls,
                    "file": out_path.name,
                    "confidence": round(inst["confidence"], 3),
                    "source": name,
                }
            )
            print(f"[assets] {obj_id:<12} conf={inst['confidence']:.3f}  <- {name}.jpg")

    if not entries:
        print("[assets] segmentation found nothing usable in any source photo")
        return build_offline_library(objects_dir, index_path, backgrounds_dir)

    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="re-download / rebuild")
    args = parser.parse_args()

    cfg = load_config(args.config)
    entries = build_library(cfg, args.force)

    print(f"\n[assets] library ({len(entries)} objects, {resolve(cfg['assets']['objects_index'])}):")
    for cls in sorted({e["class"] for e in entries}):
        members = [e for e in entries if e["class"] == cls]
        print(f"  {cls:<10} " + ", ".join(f"{e['id']} ({e['confidence']:.2f})" for e in members))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
