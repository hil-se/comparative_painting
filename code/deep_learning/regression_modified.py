import tensorflow as tf
import os
import numpy as np
import pandas as pd
from metrics import Metrics
from pdb import set_trace
import time
import random

def build_model(input_dim):
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

def within(painting, type, rater, origin=True, process_id=None):
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
    train = np.random.choice(len(y), 140, replace=False)
    test = np.array(list(set(list(range(len(y))))-set(list(train))))

    model = build_model((X[train].shape[1]))

    # Unique checkpoint path for each process
    if process_id is None:
        process_id = f"{os.getpid()}_{int(time.time()*1000000)}_{random.randint(0,999999)}"
    checkpoint_path = f"checkpoint/mlp_within_{process_id}.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=0)

    model.fit(
        X[train], y[train],
        validation_split=0.0,
        batch_size=10,
        epochs=200,
        callbacks=[reduce_lr, checkpoint],
        verbose=0
    )

    model.load_weights(checkpoint_path)
    preds_test = model.predict(X[test], verbose=0).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    
    # Clean up checkpoint file
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    except:
        pass
    
    return result

def cross(painting, type, rater, origin=True, process_id=None):
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
    train = np.random.choice(len(y), 140, replace=False)
    test = np.array(list(set(list(range(len(y))))-set(list(train))))

    y_avg = np.array(rate["Average"])
    num_rater = len(rate.columns)-1
    y_train = (y_avg * num_rater - y) / (num_rater-1)

    model = build_model((X[train].shape[1]))

    # Unique checkpoint path for each process
    if process_id is None:
        process_id = f"{os.getpid()}_{int(time.time()*1000000)}_{random.randint(0,999999)}"
    checkpoint_path = f"checkpoint/mlp_cross_{process_id}.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=0)

    model.fit(
        X[train], y_train[train],
        validation_split=0.0,
        batch_size=10,
        epochs=200,
        callbacks=[reduce_lr, checkpoint],
        verbose=0
    )

    model.load_weights(checkpoint_path)
    preds_test = model.predict(X[test], verbose=0).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    
    # Clean up checkpoint file
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    except:
        pass
    
    return result

if __name__ == '__main__':
    type = "beauty"
    painting = "abstract"
    # result = within(painting, type, "3", origin=False)
    result = cross(painting, type, "3", origin=False)
    print(result)