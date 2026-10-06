"""Extract the fixed CLIP and standard-preprocessed ResNet features used in the paper."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "8092f5b35a22023f7a822152e20837ac59cb91a3"


def load_image_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def extract_clip(
    image_paths: list[str], batch_size: int, device: str, cache_dir: Path | None
) -> np.ndarray:
    import torch
    from transformers import CLIPImageProcessor, CLIPModel

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
            image = tf.keras.utils.load_img(path, target_size=(224, 224))
            images.append(tf.keras.utils.img_to_array(image))
        values = np.asarray(images, dtype=np.float32)
        values = tf.keras.applications.resnet50.preprocess_input(values)
        batches.append(
            model.predict(values, batch_size=batch_size, verbose=0).astype(np.float32)
        )
        print(
            f"ResNet-50: {min(start + batch_size, len(image_paths))}/{len(image_paths)}"
        )
    return np.concatenate(batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("clip-vit-b32", "resnet50"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    images = [row["image_path"] for row in rows]
    if args.representation == "clip-vit-b32":
        features = extract_clip(images, args.batch_size, args.device, args.cache_dir)
    else:
        features = extract_resnet(images, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        item_ids=np.asarray([row["item_id"] for row in rows]),
        features=features,
    )


if __name__ == "__main__":
    main()
