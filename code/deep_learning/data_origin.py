"""Extract Sidhu ResNet-50 vectors at original image sizes without preprocess_input."""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from build_art_manifests import resolve_sidhu_image


def load_data(path):
    """Extract 2048-dimensional vectors without resizing or preprocessing."""
    features = []
    for index in range(240):
        image_path = resolve_sidhu_image(Path(path), index)
        if image_path is None:
            continue
        image = tf.keras.utils.load_img(image_path, color_mode="rgb")
        pixels = tf.keras.utils.img_to_array(image)
        model = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=pixels.shape,
            pooling="avg",
        )
        features.append(model(tf.Variable([pixels])).numpy()[0])
    return np.asarray(features)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=root)
    args = parser.parse_args()
    for category in ("abstract", "representational"):
        images = args.repository / "Data" / f"{category.title()}_Images"
        output = (
            args.repository
            / "code/deep_learning/feature"
            / f"{category}_feature_origin.npy"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, load_data(str(images) + "/"))


if __name__ == "__main__":
    main()
