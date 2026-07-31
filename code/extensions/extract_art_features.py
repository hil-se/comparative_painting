"""Extract deterministic ResNet-50 or CLIP image embeddings from a manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, UnidentifiedImageError


CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "8092f5b35a22023f7a822152e20837ac59cb91a3"


def _is_webp(path: str | Path) -> bool:
    header = Path(path).read_bytes()[:12]
    return header.startswith(b"RIFF") and header[8:12] == b"WEBP"


def load_image_rgb(path: str | Path) -> Image.Image:
    """Load an RGB image, using ImageMagick only for unsupported WebP files."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as error:
        if not _is_webp(path):
            raise
        converter = shutil.which("convert")
        if converter is None:
            raise RuntimeError(
                f"Pillow cannot decode WebP image {path} and ImageMagick "
                "'convert' is unavailable"
            ) from error
        try:
            converted = subprocess.run(
                [converter, str(path), "png:-"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as conversion_error:
            stderr = conversion_error.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ImageMagick could not decode WebP image {path}: {stderr.strip()}"
            ) from conversion_error
        with Image.open(io.BytesIO(converted)) as image:
            return image.convert("RGB")


def read_manifest(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return (
        [row["item_id"] for row in rows],
        [row["image_path"] for row in rows],
    )


def extract_clip(
    image_paths: list[str],
    batch_size: int,
    device: str,
    cache_dir: Path | None,
) -> np.ndarray:
    import torch
    from transformers import CLIPImageProcessor, CLIPModel

    # Image-only extraction does not need CLIP's tokenizer. Avoid loading the
    # legacy fast-tokenizer artifact, which recent Transformers versions reject.
    image_processor = CLIPImageProcessor.from_pretrained(
        CLIP_MODEL, revision=CLIP_REVISION, cache_dir=cache_dir
    )
    model = CLIPModel.from_pretrained(
        CLIP_MODEL, revision=CLIP_REVISION, cache_dir=cache_dir
    ).to(device)
    model.eval()

    batches: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for path in image_paths[start : start + batch_size]:
            images.append(load_image_rgb(path))
        inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            values = model.get_image_features(pixel_values=pixel_values)
            values = torch.nn.functional.normalize(values, p=2, dim=-1)
        batches.append(values.cpu().numpy().astype(np.float32))
        print(f"CLIP: {min(start + batch_size, len(image_paths))}/{len(image_paths)}")
    return np.concatenate(batches)


def extract_resnet(image_paths: list[str], batch_size: int) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg"
    )
    batches: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for path in image_paths[start : start + batch_size]:
            try:
                image = tf.keras.utils.load_img(path, target_size=(224, 224))
            except UnidentifiedImageError:
                image = load_image_rgb(path).resize(
                    (224, 224), resample=Image.Resampling.NEAREST
                )
            images.append(tf.keras.utils.img_to_array(image))
        values = np.asarray(images, dtype=np.float32)
        values = tf.keras.applications.resnet50.preprocess_input(values)
        batches.append(
            model.predict(values, batch_size=batch_size, verbose=0).astype(
                np.float32
            )
        )
        print(
            f"ResNet-50: {min(start + batch_size, len(image_paths))}/"
            f"{len(image_paths)}"
        )
    return np.concatenate(batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("clip-vit-b32", "resnet50"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item_ids, image_paths = read_manifest(args.manifest)
    if args.representation == "clip-vit-b32":
        features = extract_clip(
            image_paths, args.batch_size, args.device, args.cache_dir
        )
        source = {"model": CLIP_MODEL, "revision": CLIP_REVISION}
    else:
        features = extract_resnet(image_paths, args.batch_size)
        source = {"model": "tf.keras.applications.ResNet50", "weights": "imagenet"}

    if len(item_ids) != len(features):
        raise AssertionError("Feature count does not match manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        item_ids=np.asarray(item_ids),
        features=features.astype(np.float32),
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "manifest": str(args.manifest.resolve()),
        "representation": args.representation,
        "rows": len(item_ids),
        "dimensions": int(features.shape[1]),
        "sha256": digest,
        **source,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
