"""Feature extraction and rating data preparation (original image dimensions).

This module extracts visual features from painting images using a pre-trained
ResNet-50 model (ImageNet weights) with global average pooling, producing a
2048-dimensional feature vector per painting. Unlike data.py, images are NOT
resized to 224x224 -- they are processed at their original dimensions. This
means a new ResNet-50 model is instantiated per image to match each image's
unique input shape.

It also processes raw rater CSV files into per-painting rating matrices,
identical to data.py.

Used in the IEEE Access paper "Modeling Art Evaluations from Comparative
Judgments" as one of two feature extraction strategies. The original-dimension
approach preserves spatial information that may be lost during resizing.
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from pdb import set_trace


def load_data(path = "../../Data/Abstract_Images/"):
    """Extract ResNet-50 features from painting images at original dimensions.

    Loads each painting image at its native resolution (no resizing) and
    passes it through ResNet-50 (without the classification head) with
    global average pooling. Because each image may have a different size,
    a new ResNet-50 model is instantiated per image with the matching
    input shape. This produces a 2048-dimensional feature vector regardless
    of input dimensions, thanks to global average pooling.

    Images are looked up by numeric ID (1-240) with various filename
    conventions handled (e.g., "01.jpg", "12cropped.jpg",
    "05croppedtofit.jpg", ".jpeg" variants).

    Args:
        path (str): Directory path containing the painting image files.

    Returns:
        np.ndarray: Feature matrix of shape (n_paintings, 2048), where
            n_paintings <= 240 (some IDs may be missing).
    """
    features = []
    for id in range(240):
        # Painting IDs are 1-indexed; pad single-digit IDs with leading zero
        name = id+1
        if name < 10:
            name = "0"+str(name)
        else:
            name = str(name)
        # Try multiple filename conventions used in the dataset
        if os.path.exists(path+name+".jpg"):
            img = tf.keras.utils.load_img(path+name+".jpg", color_mode='rgb')
        elif os.path.exists(path+name+"cropped.jpg"):
            img = tf.keras.utils.load_img(path + name + "cropped.jpg", color_mode='rgb')
        elif os.path.exists(path+name+"croppedtofit.jpg"):
            img = tf.keras.utils.load_img(path + name + "croppedtofit.jpg", color_mode='rgb')
        elif os.path.exists(path+name+".jpeg"):
            img = tf.keras.utils.load_img(path + name + ".jpeg", color_mode='rgb')
        elif os.path.exists(path+name+"cropped.jpeg"):
            img = tf.keras.utils.load_img(path + name + "cropped.jpeg", color_mode='rgb')
        else:
            # Print the missing ID and skip this painting
            print(id)
            continue
        img_array = tf.keras.utils.img_to_array(img)
        # Instantiate ResNet-50 with this image's specific dimensions
        model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                                        input_shape=img_array.shape, pooling="avg")
        # Pass single image through ResNet-50; wrap in batch dimension
        feature = model(tf.Variable([img_array])).numpy()[0]
        features.append(list(feature))
    return np.array(features)


def rating():
    """Process raw rater CSVs into per-painting rating matrices.

    Reads the raw survey data files (one row per rater-painting pair) and
    reshapes them into matrices where each column is a rater and each row
    is a painting. Produces four CSV files:
      - feature/abstract_beauty.csv
      - feature/abstract_liking.csv
      - feature/representational_beauty.csv
      - feature/representational_liking.csv

    Certain painting IDs are excluded due to missing data:
      - Abstract: painting 173
      - Representational: paintings 90 and 157
    """
    # Define valid painting IDs, excluding known missing paintings
    id_abs = list(set(range(1,241)) - set([173]))
    id_rep = list(set(range(1,241)) - set([90,157]))

    # --- Abstract Beauty ---
    df = pd.read_csv("../../Data/Abstract_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_abs):
        tmp = df[df["Painting"]==str(id)+".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater]=[]
            data[rater].append(tmp["Beauty"].iloc[j])
    pd.DataFrame(data).to_csv("feature/abstract_beauty.csv",index=False)

    # --- Abstract Liking ---
    df = pd.read_csv("../../Data/Abstract_Liking_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_abs):
        tmp = df[df["Painting"] == str(id) + ".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater] = []
            data[rater].append(tmp["Liking"].iloc[j])
    pd.DataFrame(data).to_csv("feature/abstract_liking.csv", index=False)

    # --- Representational Beauty ---
    df = pd.read_csv("../../Data/Representational_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_rep):
        tmp = df[df["Painting"] == str(id) + ".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater] = []
            data[rater].append(tmp["Beauty"].iloc[j])
    pd.DataFrame(data).to_csv("feature/representational_beauty.csv", index=False)

    # --- Representational Liking ---
    df = pd.read_csv("../../Data/Representational_Liking_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_rep):
        tmp = df[df["Painting"] == str(id) + ".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater] = []
            data[rater].append(tmp["Liking"].iloc[j])
    pd.DataFrame(data).to_csv("feature/representational_liking.csv", index=False)


if __name__ == '__main__':
    # Extract features from abstract paintings at original dimensions and save
    path = "../../Data/Abstract_Images/"
    features = load_data(path)
    np.save('feature/abstract_feature_origin.npy', features)

    # Extract features from representational paintings at original dimensions and save
    path = "../../Data/Representational_Images/"
    features = load_data(path)
    np.save('feature/representational_feature_origin.npy', features)

    # Process raw rater data into rating CSV files
    rating()



