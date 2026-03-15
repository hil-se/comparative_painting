"""Direct regression baseline for predicting art evaluation scores.

This module implements the direct (absolute) regression approach described in
the IEEE Access paper "Modeling Art Evaluations from Comparative Judgments."
An MLP encoder maps ResNet-50 painting features to a single predicted score,
trained with Mean Absolute Error (MAE) loss. Two experimental paradigms are
provided:
  - within(): trains and evaluates on a single rater's ratings.
  - cross():  trains on the leave-one-out average of all other raters'
              ratings and evaluates against the held-out rater.

Architecture: 512 -> 256 -> 128 -> 1 with BatchNorm, Dropout, and L2
regularization. Trained for 200 epochs with batch size 10 using Adam.
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from metrics import Metrics
from pdb import set_trace


def build_model(input_dim):
    """Build the MLP encoder used for direct score regression.

    Constructs a four-layer sequential model (512 -> 256 -> 128 -> 1)
    with ReLU activations, BatchNormalization, Dropout (0.25), and L2
    regularization (1e-5). The final layer uses linear activation to
    output a continuous score. Compiled with Adam optimizer and MAE loss.

    Args:
        input_dim (int): Dimensionality of the input feature vector
            (2048 for ResNet-50 with average pooling).

    Returns:
        tf.keras.Sequential: Compiled Keras model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer='adam',
        loss="mae",
        metrics=['mae']
    )

    return model


def within(painting, type, rater, origin=True):
    """Train and evaluate a regression model on a single rater's scores.

    This implements the "within-rater" experiment: the model is trained
    directly on one rater's beauty or liking ratings and evaluated on
    held-out paintings from that same rater.

    Args:
        painting (str): Painting category, either "abstract" or
            "representational".
        type (str): Rating type, either "beauty" or "liking".
        rater (str or int): Identifier of the rater whose ratings are used
            for both training and evaluation.
        origin (bool): If True, use features extracted at original image
            dimensions; if False, use features from 224x224 resized images.

    Returns:
        dict: Evaluation results with keys "Rater", "mae", "r2", "rho"
            (Pearson correlation), and "rs" (Spearman correlation).
    """
    # Load pre-extracted ResNet-50 features and corresponding ratings
    if painting == "abstract":
        if origin:
            X = np.load('feature/abstract_feature_origin.npy')
        else:
            X = np.load('feature/abstract_feature.npy')
        if type == "beauty":
            rate = pd.read_csv("feature/abstract_beauty.csv")
        else:
            rate = pd.read_csv("feature/abstract_liking.csv")
        y = np.array(rate[str(rater)])
    else:
        if origin:
            X = np.load('feature/representational_feature_origin.npy')
        else:
            X = np.load('feature/representational_feature.npy')
        if type == "beauty":
            rate = pd.read_csv("feature/representational_beauty.csv")
        else:
            rate = pd.read_csv("feature/representational_liking.csv")
        y = np.array(rate[str(rater)])

    # Randomly split: 140 paintings for training, remainder for testing
    train = np.random.choice(len(y), 140, replace=False)
    test = np.array(list(set(list(range(len(y))))-set(list(train))))
    # all = list(range(len(y)))
    # train = all[:140]
    # test = all[140:]

    model = build_model((X[train].shape[1]))

    #checkpoint_path = "checkpoint/mlp.weights.h5"
    checkpoint_path = "checkpoint/mlp.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # ReduceLROnPlateau: reduce learning rate by factor 0.3 after 100 epochs of no improvement
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    # Save only the best model weights based on training loss
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=150, verbose=1,
    #                                                   restore_best_weights=True)

    model.fit(
        X[train], y[train],
        # validation_data=(X[test], y[test]),
        validation_split=0.0,
        batch_size=10,
        epochs=200,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )

    # Restore best weights and evaluate on the test set
    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)
    preds_test = model.predict(X[test]).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    print(result)
    return result


def cross(painting, type, rater, origin=True):
    """Train on leave-one-out average ratings and evaluate against the held-out rater.

    This implements the "cross-rater" experiment: the training target is the
    average rating of all raters *except* the target rater (leave-one-out),
    and the model is evaluated by comparing its predictions against the
    target rater's actual ratings on the test paintings.

    Args:
        painting (str): Painting category, either "abstract" or
            "representational".
        type (str): Rating type, either "beauty" or "liking".
        rater (str or int): Identifier of the rater to hold out. The model
            trains on the average of all other raters.
        origin (bool): If True, use features extracted at original image
            dimensions; if False, use features from 224x224 resized images.

    Returns:
        dict: Evaluation results with keys "Rater", "mae", "r2", "rho"
            (Pearson correlation), and "rs" (Spearman correlation).
    """
    # Load pre-extracted ResNet-50 features and corresponding ratings
    if painting == "abstract":
        if origin:
            X = np.load('feature/abstract_feature_origin.npy')
        else:
            X = np.load('feature/abstract_feature.npy')
        if type == "beauty":
            rate = pd.read_csv("feature/abstract_beauty.csv")
        else:
            rate = pd.read_csv("feature/abstract_liking.csv")
        y = np.array(rate[str(rater)])
    else:
        if origin:
            X = np.load('feature/representational_feature_origin.npy')
        else:
            X = np.load('feature/representational_feature.npy')
        if type == "beauty":
            rate = pd.read_csv("feature/representational_beauty.csv")
        else:
            rate = pd.read_csv("feature/representational_liking.csv")
        y = np.array(rate[str(rater)])

    # Randomly split: 140 paintings for training, remainder for testing
    train = np.random.choice(len(y), 140, replace=False)
    test = np.array(list(set(list(range(len(y))))-set(list(train))))

    # Compute leave-one-out average: subtract this rater's score from the
    # overall average to get the mean of all other raters' scores.
    # Formula: avg_others = (avg_all * N - rater_score) / (N - 1)
    y_avg = y = np.array(rate["Average"])
    num_rater = len(rate.columns)-1  # subtract 1 for the "Painting" column
    y_train = (y_avg * num_rater - y) / (num_rater-1)

    model = build_model((X[train].shape[1]))

    checkpoint_path = "checkpoint/mlp.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=150, verbose=1,
    #                                                   restore_best_weights=True)

    model.fit(
        X[train], y_train[train],
        # validation_data=(X[test], y[test]),
        validation_split=0.0,
        batch_size=10,
        epochs=200,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )

    # Restore best weights and evaluate on the test set
    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)
    preds_test = model.predict(X[test]).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    print(result)
    return result

if __name__ == '__main__':

    # types = ["beauty", "liking"]
    #
    # painting = "abstract"
    # for type in types:
    #     results = []
    #     for rater in range(49):
    #         result = within(painting, type, rater+1)
    #         results.append(result)
    #     pd.DataFrame(results).to_csv("result/"+painting+"_"+type+"_within.csv", index=False)
    #
    # painting = "reprensentational"
    # for type in types:
    #     results = []
    #     for rater in range(43):
    #         result = within(painting, type, rater + 1)
    #         results.append(result)
    #     pd.DataFrame(results).to_csv("result/" + painting + "_" + type + "_within.csv", index=False)

    type = "beauty"
    painting = "abstract"
    # result = within(painting, type, "3", origin=False)
    result = cross(painting, type, "1", origin=False)
    print(result)
