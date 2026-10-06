"""Build canonical manifests for the Sidhu and APDDv2 art experiments."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np


APDD_TARGETS = (
    "Total aesthetic score",
    "Theme and logic",
    "Creativity",
    "Layout and composition",
    "Space and perspective",
    "The sense of order",
    "Light and shadow",
    "Color",
    "Details and texture",
    "The overall",
    "Mood",
)

SIDHU_RATING_FILES = {
    ("abstract", "beauty"): ("Abstract_All_Raters.csv", "Beauty"),
    ("abstract", "liking"): ("Abstract_Liking_All_Raters.csv", "Liking"),
    ("representational", "beauty"): (
        "Representational_All_Raters.csv",
        "Beauty",
    ),
    ("representational", "liking"): (
        "Representational_Liking_All_Raters.csv",
        "Liking",
    ),
}


def resolve_sidhu_image(
    image_dir: Path, painting_index: int
) -> Path | None:
    """Resolve the repository's inconsistent numeric painting filenames."""

    stem = f"{painting_index + 1:02d}"
    candidates = (
        image_dir / f"{stem}.jpg",
        image_dir / f"{stem}.JPG",
        image_dir / f"{stem}.jpeg",
        image_dir / f"{stem}cropped.jpg",
        image_dir / f"{stem}cropped.JPG",
        image_dir / f"{stem}cropped.jpeg",
        image_dir / f"{stem}croppedtofit.jpg",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_sidhu(
    repository: Path, output: Path, resnet_output: Path | None = None
) -> dict[str, object]:
    feature_dir = repository / "code" / "deep_learning" / "feature"
    data_dir = repository / "Data"
    rows: dict[tuple[str, int], dict[str, str | float]] = {}

    for category in ("abstract", "representational"):
        image_dir = data_dir / f"{category.title()}_Images"
        for target in ("beauty", "liking"):
            ratings_file, rating_column = SIDHU_RATING_FILES[(category, target)]
            ratings_path = data_dir / ratings_file
            ratings_by_painting: dict[int, list[float]] = {}
            with ratings_path.open(newline="", encoding="utf-8-sig") as stream:
                for rating_row in csv.DictReader(stream):
                    painting_number = int(Path(rating_row["Painting"]).stem)
                    ratings_by_painting.setdefault(painting_number, []).append(
                        float(rating_row[rating_column])
                    )

            for painting_number in sorted(ratings_by_painting):
                painting_index = painting_number - 1
                image_path = resolve_sidhu_image(image_dir, painting_index)
                if image_path is None:
                    continue
                key = (category, painting_index)
                row = rows.setdefault(
                    key,
                    {
                        "dataset": "sidhu",
                        "item_id": f"{category}-{painting_index:03d}",
                        "image_path": str(image_path),
                        "category": category,
                    },
                )
                row[target] = float(
                    np.mean(ratings_by_painting[painting_number])
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "dataset",
        "item_id",
        "image_path",
        "category",
        "beauty",
        "liking",
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows.values())

    if resnet_output is not None:
        feature_blocks = []
        item_ids = []
        for category in ("abstract", "representational"):
            block = np.load(feature_dir / f"{category}_feature_origin.npy")
            image_dir = repository / "Data" / f"{category.title()}_Images"
            available_indices = [
                index
                for index in range(240)
                if resolve_sidhu_image(image_dir, index) is not None
            ]
            if len(block) != len(available_indices):
                raise ValueError(
                    f"Released {category} ResNet rows ({len(block)}) do not match "
                    f"available images ({len(available_indices)})"
                )
            feature_blocks.append(block.astype(np.float32))
            item_ids.extend(
                f"{category}-{index:03d}" for index in available_indices
            )
        resnet_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resnet_output,
            item_ids=np.asarray(item_ids),
            features=np.concatenate(feature_blocks),
        )

    return {
        "dataset": "sidhu",
        "rows": len(rows),
        "targets": ["beauty", "liking"],
        "categories": sorted({str(row["category"]) for row in rows.values()}),
        "excluded_missing_images": {
            "abstract": 240
            - sum(key[0] == "abstract" for key in rows),
            "representational": 240
            - sum(key[0] == "representational" for key in rows),
        },
        "excluded_missing_image_ids": {
            category: [
                painting_index + 1
                for painting_index in range(240)
                if resolve_sidhu_image(
                    data_dir / f"{category.title()}_Images", painting_index
                )
                is None
            ]
            for category in ("abstract", "representational")
        },
        "manifest": str(output.resolve()),
        "released_resnet_features": (
            None if resnet_output is None else str(resnet_output.resolve())
        ),
    }


def build_apdd(
    annotations: Path,
    images: Path,
    output: Path,
    max_missing_images: int = 0,
) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    missing_images: list[str] = []

    annotation_text = annotations.read_text(
        encoding="utf-8-sig", errors="replace"
    )
    decode_replacement_count = annotation_text.count("\ufffd")
    with io.StringIO(annotation_text, newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = set(("filename", "Artistic Categories", *APDD_TARGETS))
        missing_columns.difference_update(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "APDDv2 annotation file is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for source in reader:
            filename = source["filename"].strip()
            image_path = (images / filename).resolve()
            if not image_path.is_file():
                missing_images.append(filename)
                continue
            row = {
                "dataset": "apddv2",
                "item_id": Path(filename).stem,
                "image_path": str(image_path),
                "category": source["Artistic Categories"].strip(),
            }
            row.update({target: source[target].strip() for target in APDD_TARGETS})
            rows.append(row)

    if len(missing_images) > max_missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} APDDv2 images are missing beneath {images}; "
            f"allowed maximum is {max_missing_images}; "
            f"first missing files: {preview}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("dataset", "item_id", "image_path", "category", *APDD_TARGETS)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    target_counts = {
        target: sum(bool(row[target]) for row in rows) for target in APDD_TARGETS
    }
    return {
        "dataset": "apddv2",
        "rows": len(rows),
        "targets": list(APDD_TARGETS),
        "target_non_null_counts": target_counts,
        "annotation_decode_replacement_count": decode_replacement_count,
        "excluded_missing_images": missing_images,
        "excluded_missing_image_count": len(missing_images),
        "manifest": str(output.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sidhu", "apddv2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="comparative_painting repository (Sidhu only)",
    )
    parser.add_argument("--annotations", type=Path, help="APDDv2-10023.csv")
    parser.add_argument("--images", type=Path, help="APDDv2 image directory")
    parser.add_argument(
        "--max-missing-images",
        type=int,
        default=0,
        help="maximum documented APDDv2 annotation rows allowed without images",
    )
    parser.add_argument(
        "--resnet-output",
        type=Path,
        help="package released Sidhu ResNet features as an aligned NPZ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "sidhu":
        summary = build_sidhu(
            args.repository.resolve(), args.output, args.resnet_output
        )
    else:
        if args.annotations is None or args.images is None:
            raise SystemExit("APDDv2 requires --annotations and --images")
        summary = build_apdd(
            args.annotations,
            args.images,
            args.output,
            max_missing_images=args.max_missing_images,
        )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
