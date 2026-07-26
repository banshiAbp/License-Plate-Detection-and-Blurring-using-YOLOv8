"""Check for exact and perceptual duplicate leakage across dataset splits.

This script strengthens evaluation reliability by looking for images that appear
in more than one split. It uses:
- MD5 hashes for exact file duplicates.
- Difference hashes (dHash) for visually similar images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "images"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "dataset_leakage_check"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(split: str) -> list[Path]:
    return sorted(
        path
        for path in (IMAGE_DIR / split).glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def md5_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> str:
    """Return a 64-bit difference hash as 16-character hex."""
    with Image.open(path) as image:
        image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())

    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            value = (value << 1) | int(left > right)
    return f"{value:016x}"


def hamming_distance(hex_a: str, hex_b: str) -> int:
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def group_cross_split_duplicates(hash_records: dict[str, list[dict]]) -> list[dict]:
    groups = []
    for hash_value, records in hash_records.items():
        splits = sorted({record["split"] for record in records})
        if len(splits) > 1:
            groups.append(
                {
                    "hash": hash_value,
                    "splits": ",".join(splits),
                    "image_count": len(records),
                    "images": "; ".join(f"{record['split']}:{record['path']}" for record in records),
                }
            )
    return groups


def find_near_duplicate_pairs(records: list[dict], max_hamming_distance: int) -> list[dict]:
    """Find cross-split perceptual hash pairs within a small Hamming distance."""
    pairs = []
    for left, right in combinations(records, 2):
        if left["split"] == right["split"]:
            continue
        distance = hamming_distance(left["dhash"], right["dhash"])
        if distance <= max_hamming_distance:
            pairs.append(
                {
                    "left_split": left["split"],
                    "left_image": left["path"],
                    "right_split": right["split"],
                    "right_image": right["path"],
                    "hamming_distance": distance,
                }
            )
    return pairs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset leakage across train/val/test splits.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=4,
        help="Maximum dHash Hamming distance for near-duplicate image pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    md5_records: dict[str, list[dict]] = defaultdict(list)
    dhash_records: dict[str, list[dict]] = defaultdict(list)
    all_records = []
    failed_files = []

    for split in ["train", "val", "test"]:
        for path in image_files(split):
            relative_path = str(path.relative_to(PROJECT_DIR))
            try:
                record = {
                    "split": split,
                    "path": relative_path,
                    "md5": md5_hash(path),
                    "dhash": dhash(path),
                }
            except Exception as exc:  # pragma: no cover - defensive data audit
                failed_files.append({"split": split, "path": relative_path, "error": str(exc)})
                continue

            all_records.append(record)
            md5_records[record["md5"]].append(record)
            dhash_records[record["dhash"]].append(record)

    exact_duplicate_groups = group_cross_split_duplicates(md5_records)
    perceptual_duplicate_groups = group_cross_split_duplicates(dhash_records)
    near_duplicate_pairs = find_near_duplicate_pairs(all_records, args.near_duplicate_threshold)

    summary = {
        "images_scanned": len(all_records),
        "failed_files": len(failed_files),
        "cross_split_exact_duplicate_groups": len(exact_duplicate_groups),
        "cross_split_perceptual_duplicate_groups": len(perceptual_duplicate_groups),
        "cross_split_near_duplicate_pairs": len(near_duplicate_pairs),
        "near_duplicate_hamming_threshold": args.near_duplicate_threshold,
    }

    write_csv(
        args.output / "image_hash_inventory.csv",
        all_records,
        ["split", "path", "md5", "dhash"],
    )
    write_csv(
        args.output / "cross_split_exact_duplicate_groups.csv",
        exact_duplicate_groups,
        ["hash", "splits", "image_count", "images"],
    )
    write_csv(
        args.output / "cross_split_perceptual_duplicate_groups.csv",
        perceptual_duplicate_groups,
        ["hash", "splits", "image_count", "images"],
    )
    write_csv(
        args.output / "cross_split_near_duplicate_pairs.csv",
        near_duplicate_pairs,
        ["left_split", "left_image", "right_split", "right_image", "hamming_distance"],
    )
    write_csv(args.output / "failed_files.csv", failed_files, ["split", "path", "error"])
    (args.output / "leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
