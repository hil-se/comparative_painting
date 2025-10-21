import tensorflow as tf
import os
import numpy as np
import pandas as pd
from metrics import Metrics
from pdb import set_trace

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

def within(painting, type, rater, origin=True):
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
    # all = list(range(len(y)))
    # train = all[:140]
    # test = all[140:]

    model = build_model((X[train].shape[1]))

    checkpoint_path = "checkpoint/mlp.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
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

    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)
    preds_test = model.predict(X[test]).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    print(result)
    return result

def cross(painting, type, rater, origin=True):
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

    checkpoint_path = "checkpoint/mlp.keras"
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

    type = "liking"
    painting = "abstract"
    # result = within(painting, type, "3", origin=False)
    result = cross(painting, type, "1", origin=False)
    print(result)
