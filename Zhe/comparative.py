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

    return model

class ComparativeModel(tf.keras.Model):
    def __init__(self, encoder, w_loss="hinge", **kwargs):
        super(ComparativeModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.w_loss = w_loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        return tf.subtract(encodings_A, encodings_B)

    def compute_loss(self, y, diff):
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff))))
        return loss

    def compute_loss_square(self, y, diff):
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.square(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff)))))
        return loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            diff = self(x)
            if self.w_loss == "hinge":
                loss = self.compute_loss(y, diff)
            else:
                loss = self.compute_loss_square(y, diff)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # def test_step(self, features):
    #     encodings_A, encodings_B = self(features)
    #     loss = self.compute_loss(encodings_A, encodings_B, features["Label"])
    #     self.loss_tracker.update_state(loss)
    #     return {"loss": self.loss_tracker.result()}

    def predict(self, X):
        """Predicts preference between two items."""
        return np.array(self.encoder(np.array(X.tolist())))

def generate_comparative_judgments(X, y, N=1):
    m = len(y)
    features = {"A": [], "B": [], "Label": []}
    seen = set()
    for i in range(m):
        n = 0
        while n < N:
            j = np.random.randint(0, m)
            if (i,j) in seen or (j,i) in seen:
                continue
            if y[i] > y[j]:
                features["A"].append(X[i])
                features["B"].append(X[j])
                features["Label"].append(1.0)
                # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": 1.0})
                n += 1
            elif y[i] < y[j]:
                features["A"].append(X[i])
                features["B"].append(X[j])
                features["Label"].append(-1.0)
                # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": -1.0})
                n += 1
            seen.add((i, j))
    features = {key: np.array(features[key]) for key in features}
    return features

def within(painting, type, rater, origin=True, N=1):
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

    features = generate_comparative_judgments(X[train], y[train], N=N)

    encoder = build_model((X[train].shape[1]))
    model = ComparativeModel(encoder)
    checkpoint_path = "checkpoint/comparative.keras"

    model.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )

    # Train model
    history = model.fit(features, features["Label"], epochs=100, batch_size=10, callbacks=[reduce_lr, checkpoint],
                     verbose=1)
    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)

    preds_test = model.predict(X[test]).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    print(result)
    return result

def cross(painting, type, rater, origin=True, N=1):
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

    features = generate_comparative_judgments(X[train], y_train[train], N=N)

    encoder = build_model((X[train].shape[1]))
    model = ComparativeModel(encoder)
    checkpoint_path = "checkpoint/comparative.keras"

    model.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )

    # Train model
    history = model.fit(features, features["Label"], epochs=100, batch_size=10, callbacks=[reduce_lr, checkpoint],
                        verbose=1)
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
    # result = within(painting, type, "3", origin=False, N=1)
    result = cross(painting, type, "3", origin=False, N=1)
    print(result)
