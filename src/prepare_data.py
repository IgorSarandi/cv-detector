"""Convert Oxford-IIIT Pet (PASCAL VOC head boxes) into a YOLO detection dataset.

Expected raw layout (what you downloaded and unpacked):

    <raw-dir>/images/*.jpg
    <raw-dir>/annotations/xmls/*.xml

Output layout (Ultralytics YOLO format):

    <out-dir>/images/{train,val,test}/*.jpg
    <out-dir>/labels/{train,val,test}/*.txt     # class x_center y_center w h  (normalized 0..1)
    <out-dir>/data.yaml
    <out-dir>/split_manifest.csv                # stem, breed, split  (for reproducibility)

Notes:
  * Only images that HAVE an xml annotation are used (roughly half of the dataset).
  * Species is decided by the file name: capital first letter = cat, lowercase = dog.
  * The boxes in this dataset cover the animal's HEAD, not the whole body.
  * The split is stratified by breed and fully determined by --seed.
  * The split is done here, BEFORE any training, so the model never sees val/test images.

Example:
    python src/prepare_data.py --raw-dir ~/datasets/oxford-pets --out-dir data/cats_yolo
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, required=True, help="folder containing images/ and annotations/")
    p.add_argument("--out-dir", type=Path, default=Path("data/cats_yolo"))
    p.add_argument("--species", choices=("cat", "dog", "both"), default="cat")
    p.add_argument("--class-name", default="cat_head", help="name of the single detection class")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None, help="optional cap on total images (for quick tests)")
    p.add_argument("--symlink", action="store_true", help="symlink images instead of copying them")
    p.add_argument("--force", action="store_true", help="overwrite an existing out-dir")
    return p.parse_args()


def is_cat(stem: str) -> bool:
    # Oxford-IIIT Pet convention: capitalised file name = cat, lowercase = dog.
    return stem[0].isupper()


def breed_of(stem: str) -> str:
    # "Egyptian_Mau_145" -> "Egyptian_Mau"
    return stem.rsplit("_", 1)[0]


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Return YOLO label lines (class 0) for every <object> in a VOC xml."""
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        # clip to the real image (a few VOC boxes slightly exceed the borders)
        xmin, xmax = max(0.0, xmin), min(float(img_w), xmax)
        ymin, ymax = max(0.0, ymin), min(float(img_h), ymax)
        w, h = xmax - xmin, ymax - ymin
        if w <= 1 or h <= 1:
            continue
        xc, yc = xmin + w / 2, ymin + h / 2
        lines.append(f"0 {xc / img_w:.6f} {yc / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return lines


def collect_samples(raw: Path, species: str, max_images: int | None, rng: random.Random):
    """Find (stem, image_path, label_lines) for every usable image."""
    img_dir, xml_dir = raw / "images", raw / "annotations" / "xmls"
    for d in (img_dir, xml_dir):
        if not d.is_dir():
            sys.exit(f"ERROR: folder not found: {d}")

    stats = defaultdict(int)
    samples = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        stem = xml_path.stem
        if species == "cat" and not is_cat(stem):
            continue
        if species == "dog" and is_cat(stem):
            continue
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            stats["no_image"] += 1
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:  # verify() invalidates the handle; reopen for size
                w, h = im.size
        except Exception:
            stats["corrupt_image"] += 1
            continue
        try:
            lines = voc_to_yolo(xml_path, w, h)
        except Exception:
            stats["bad_xml"] += 1
            continue
        if not lines:
            stats["no_valid_box"] += 1
            continue
        samples.append((stem, img_path, lines))

    if max_images is not None and len(samples) > max_images:
        samples = rng.sample(samples, max_images)
        samples.sort(key=lambda s: s[0])
    return samples, stats


def stratified_split(samples, ratios, rng: random.Random) -> dict[str, str]:
    """Split per breed so every breed appears in every split. Returns stem -> split."""
    by_breed = defaultdict(list)
    for stem, *_ in samples:
        by_breed[breed_of(stem)].append(stem)

    assignment = {}
    for breed in sorted(by_breed):
        stems = sorted(by_breed[breed])
        rng.shuffle(stems)
        n = len(stems)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, stem in enumerate(stems):
            assignment[stem] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def main() -> None:
    args = parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"ERROR: ratios must sum to 1, got {sum(args.ratios)}")

    out = args.out_dir.resolve()
    if out.exists():
        if not args.force:
            sys.exit(f"ERROR: {out} already exists. Use --force to overwrite.")
        shutil.rmtree(out)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (out / kind / split).mkdir(parents=True)

    rng = random.Random(args.seed)
    samples, skipped = collect_samples(args.raw_dir.expanduser(), args.species, args.max_images, rng)
    if not samples:
        sys.exit("ERROR: no usable images found. Check --raw-dir and --species.")
    assignment = stratified_split(samples, args.ratios, rng)

    counts = defaultdict(lambda: {"images": 0, "boxes": 0})
    areas = []
    rows = []
    for stem, img_path, lines in samples:
        split = assignment[stem]
        dst_img = out / "images" / split / img_path.name
        if args.symlink:
            dst_img.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dst_img)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(lines)
        areas += [float(l.split()[3]) * float(l.split()[4]) for l in lines]
        rows.append((stem, breed_of(stem), split))

    with open(out / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("stem", "breed", "split"))
        writer.writerows(rows)

    (out / "data.yaml").write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  0: {args.class_name}\n"
    )

    print(f"Output: {out}   (seed={args.seed}, species={args.species})")
    print(f"{'split':<8}{'images':>8}{'boxes':>8}")
    for split in SPLITS:
        print(f"{split:<8}{counts[split]['images']:>8}{counts[split]['boxes']:>8}")
    print(f"total   {len(samples):>8}")
    areas.sort()
    print(f"box area / image area: min={areas[0]:.3f}  median={areas[len(areas) // 2]:.3f}  max={areas[-1]:.3f}")
    if skipped:
        print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()
