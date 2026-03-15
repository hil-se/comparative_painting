"""Feature extraction and rating data preparation (224x224 resized images).

This module extracts visual features from painting images using a pre-trained
ResNet-50 model (ImageNet weights) with global average pooling, producing a
2048-dimensional feature vector per painting. All images are resized to
224x224 pixels before feature extraction, which is the standard ResNet input
size.

It also processes raw rater CSV files into per-painting rating matrices for
both beauty and liking judgments, for abstract and representational paintings.

Used in the IEEE Access paper "Modeling Art Evaluations from Comparative
Judgments" as one of two feature extraction strategies (the other being
data_origin.py which preserves original image dimensions).
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from pdb import set_trace


def load_data(path = "../../Data/Abstract_Images/"):
    """Extract ResNet-50 features from painting images resized to 224x224.

    Loads each painting image, resizes it to 224x224x3, and passes it
    through ResNet-50 (without the classification head) with global average
    pooling to produce a 2048-dimensional feature vector. The ResNet-50
    model is instantiated once and reused for all images.

    Images are looked up by numeric ID (1-240) with various filename
    conventions handled (e.g., "01.jpg", "12cropped.jpg",
    "05croppedtofit.jpg", ".jpeg" variants).

    Args:
        path (str): Directory path containing the painting image files.

    Returns:
        np.ndarray: Feature matrix of shape (n_paintings, 2048), where
            n_paintings <= 240 (some IDs may be missing).
    """
    size = (224, 224, 3)
    features = []
    # Load ResNet-50 once with fixed input size for all images
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=size, pooling="avg")
    for id in range(240):
        # Painting IDs are 1-indexed; pad single-digit IDs with leading zero
        name = id+1
        if name < 10:
            name = "0"+str(name)
        else:
            name = str(name)
        # Try multiple filename conventions used in the dataset
        if os.path.exists(path+name+".jpg"):
            img = tf.keras.utils.load_img(path+name+".jpg", color_mode='rgb', target_size=size)
        elif os.path.exists(path+name+"cropped.jpg"):
            img = tf.keras.utils.load_img(path + name + "cropped.jpg", color_mode='rgb', target_size=size)
        elif os.path.exists(path+name+"croppedtofit.jpg"):
            img = tf.keras.utils.load_img(path + name + "croppedtofit.jpg", color_mode='rgb', target_size=size)
        elif os.path.exists(path+name+".jpeg"):
            img = tf.keras.utils.load_img(path + name + ".jpeg", color_mode='rgb', target_size=size)
        elif os.path.exists(path+name+"cropped.jpeg"):
            img = tf.keras.utils.load_img(path + name + "cropped.jpeg", color_mode='rgb', target_size=size)
        else:
            # Print the missing ID and skip this painting
            print(id)
            continue
        img_array = tf.keras.utils.img_to_array(img)
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
    # Extract features from abstract paintings and save
    path = "../../Data/Abstract_Images/"
    features = load_data(path)
    np.save('feature/abstract_feature.npy', features)

    # Extract features from representational paintings and save
    path = "../../Data/Representational_Images/"
    features = load_data(path)
    np.save('feature/representational_feature.npy', features)

    # Process raw rater data into rating CSV files
    rating()



