"""Comparative (pairwise) model for predicting art evaluation scores.

This module implements the comparative judgment approach described in the
IEEE Access paper "Modeling Art Evaluations from Comparative Judgments."
Instead of predicting absolute scores, the model learns from pairwise
comparisons: given two paintings, which one is rated higher?

The ComparativeModel uses a Siamese architecture: a shared MLP encoder
(512 -> 256 -> 128 -> 1) processes both paintings independently, and the
difference of their outputs is trained with a hinge loss. At inference
time, the encoder produces a latent score for each painting that preserves
the learned ranking.

Two experimental paradigms are provided:
  - within(): trains on comparative pairs from a single rater's ratings.
  - cross():  trains on comparative pairs derived from leave-one-out
              average ratings, evaluated against the held-out rater.

Trained for 100 epochs with batch size 10 using Adam.
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from metrics import Metrics
from pdb import set_trace


def build_model(input_dim):
    """Build the shared MLP encoder for the Siamese comparative model.

    Constructs the same four-layer architecture as the regression baseline
    (512 -> 256 -> 128 -> 1) with ReLU activations, BatchNormalization,
    Dropout (0.25), and L2 regularization (1e-5). The final layer outputs
    a single scalar latent score per painting. Unlike regression.py, this
    model is NOT compiled here because it will be wrapped inside
    ComparativeModel which defines its own training logic.

    Args:
        input_dim (int): Dimensionality of the input feature vector
            (2048 for ResNet-50 with average pooling).

    Returns:
        tf.keras.Sequential: Uncompiled Keras model to be used as the
            shared encoder in ComparativeModel.
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

    return model


class ComparativeModel(tf.keras.Model):
    """Siamese pairwise comparison model using a shared encoder.

    This model takes two painting feature vectors (A and B), passes each
    through the same encoder network, and outputs the difference
    score(A) - score(B). Training uses hinge loss (or squared hinge loss)
    to learn an embedding where the sign of the difference matches the
    pairwise label (+1 if A is rated higher, -1 if B is rated higher).

    At inference time, the encoder alone is used to produce a scalar
    score for each painting that preserves the learned ranking order.

    Attributes:
        encoder (tf.keras.Model): Shared MLP encoder that maps features
            to a scalar score.
        loss_tracker (tf.keras.metrics.Mean): Running mean of the training
            loss across batches.
        w_loss (str): Loss function type, either "hinge" (default) or
            "square" for squared hinge loss.
    """

    def __init__(self, encoder, w_loss="hinge", **kwargs):
        """Initialize the ComparativeModel.

        Args:
            encoder (tf.keras.Model): Pre-built encoder network (from
                build_model) to be shared across both branches.
            w_loss (str): Which loss to use. "hinge" for standard hinge
                loss, any other value for squared hinge loss.
            **kwargs: Additional keyword arguments passed to the parent
                tf.keras.Model constructor.
        """
        super(ComparativeModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.w_loss = w_loss

    @property
    def metrics(self):
        """Return the list of metrics tracked during training.

        Returns:
            list: List containing the loss tracker metric.
        """
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        """Forward pass: compute score difference between painting A and B.

        Both paintings are passed through the shared encoder, and the
        difference score(A) - score(B) is returned.

        Args:
            features (dict): Dictionary with keys "A" and "B", each
                containing a batch of feature vectors.
            trainable (bool): Whether the encoder should be in training
                mode (affects BatchNorm and Dropout behavior).

        Returns:
            tf.Tensor: Scalar difference score(A) - score(B) for each
                pair in the batch.
        """
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        return tf.subtract(encodings_A, encodings_B)

    def compute_loss(self, y, diff):
        """Compute standard hinge loss for pairwise comparisons.

        Hinge loss: mean(max(0, 1 - y * diff)), where y is +1 or -1
        and diff is score(A) - score(B). The loss is zero when the
        predicted difference agrees with the label by a margin >= 1.

        Args:
            y (tf.Tensor): Ground-truth pairwise labels (+1 or -1).
            diff (tf.Tensor): Predicted score differences.

        Returns:
            tf.Tensor: Scalar mean hinge loss.
        """
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff))))
        return loss

    def compute_loss_square(self, y, diff):
        """Compute squared hinge loss for pairwise comparisons.

        Squared hinge loss: mean(max(0, 1 - y * diff)^2). This variant
        penalizes margin violations more heavily than standard hinge loss.

        Args:
            y (tf.Tensor): Ground-truth pairwise labels (+1 or -1).
            diff (tf.Tensor): Predicted score differences.

        Returns:
            tf.Tensor: Scalar mean squared hinge loss.
        """
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.square(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff)))))
        return loss

    def train_step(self, data):
        """Execute one training step with gradient computation and update.

        Performs a forward pass, computes the selected loss (hinge or
        squared hinge), backpropagates gradients, and updates weights.

        Args:
            data (tuple): Tuple of (x, y) where x is the input features
                dict and y is the pairwise labels tensor.

        Returns:
            dict: Dictionary with current running loss value.
        """
        x, y = data
        with tf.GradientTape() as tape:
            diff = self(x)
            # Select loss function based on w_loss parameter
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
        """Predict absolute latent scores for individual paintings.

        Uses only the shared encoder (not the full Siamese forward pass)
        to produce a scalar score for each painting. These scores preserve
        the ranking learned from pairwise comparisons.

        Args:
            X (np.ndarray or similar): Feature matrix of shape
                (n_paintings, feature_dim).

        Returns:
            np.ndarray: Predicted latent scores of shape (n_paintings, 1).
        """
        return np.array(self.encoder(np.array(X.tolist())))


def generate_comparative_judgments(X, y, N=1):
    """Generate pairwise comparison training data from absolute ratings.

    For each painting, randomly samples N other paintings and creates a
    comparison pair with label +1 if the first painting is rated higher,
    or -1 if rated lower. Pairs where both paintings have equal ratings
    are skipped (no comparison can be made). Duplicate pairs are also
    prevented via a seen-set.

    Args:
        X (np.ndarray): Feature matrix of shape (n_paintings, feature_dim).
        y (np.ndarray): Rating array of shape (n_paintings,).
        N (int): Number of comparison partners to sample per painting.
            Controls the total number of generated pairs.

    Returns:
        dict: Dictionary with keys:
            - "A": np.ndarray of feature vectors for the first painting
                   in each pair.
            - "B": np.ndarray of feature vectors for the second painting
                   in each pair.
            - "Label": np.ndarray of pairwise labels (+1.0 or -1.0).
    """
    m = len(y)
    features = {"A": [], "B": [], "Label": []}
    seen = set()  # Track generated pairs to avoid duplicates
    for i in range(m):
        n = 0
        while n < N:
            j = np.random.randint(0, m)
            # Skip if this pair (in either order) was already generated
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
    # Convert lists to numpy arrays for Keras compatibility
    features = {key: np.array(features[key]) for key in features}
    return features


def within(painting, type, rater, origin=True, N=1):
    """Train and evaluate a comparative model on a single rater's scores.

    This implements the "within-rater" comparative experiment: pairwise
    comparisons are generated from one rater's beauty or liking ratings,
    the Siamese model is trained on these pairs, and evaluation compares
    the encoder's predicted latent scores against the rater's actual
    ratings on held-out paintings.

    Args:
        painting (str): Painting category, either "abstract" or
            "representational".
        type (str): Rating type, either "beauty" or "liking".
        rater (str or int): Identifier of the rater whose ratings are used.
        origin (bool): If True, use features extracted at original image
            dimensions; if False, use features from 224x224 resized images.
        N (int): Number of comparison partners per painting for pair
            generation (passed to generate_comparative_judgments).

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

    # Generate pairwise comparisons from the training set ratings
    features = generate_comparative_judgments(X[train], y[train], N=N)

    encoder = build_model((X[train].shape[1]))
    model = ComparativeModel(encoder)
    checkpoint_path = "checkpoint/comparative.weights.h5"

    model.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )

    # Train model on pairwise comparisons for 100 epochs
    history = model.fit(features, features["Label"], epochs=100, batch_size=10, callbacks=[reduce_lr, checkpoint],
                     verbose=1)
    # Restore best weights and evaluate on the test set
    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)

    # Use encoder alone to predict latent scores for test paintings
    preds_test = model.predict(X[test]).flatten()
    m = Metrics(y[test], preds_test)
    result = {"Rater": rater, "mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}
    print(result)
    return result


def cross(painting, type, rater, origin=True, N=1):
    """Train on leave-one-out comparative pairs, evaluate against the held-out rater.

    This implements the "cross-rater" comparative experiment: pairwise
    comparisons are generated from the leave-one-out average ratings
    (average of all raters except the target), and the model is evaluated
    by comparing its predicted latent scores against the target rater's
    actual ratings on the test paintings.

    Args:
        painting (str): Painting category, either "abstract" or
            "representational".
        type (str): Rating type, either "beauty" or "liking".
        rater (str or int): Identifier of the rater to hold out. The model
            trains on comparative pairs from the average of all other raters.
        origin (bool): If True, use features extracted at original image
            dimensions; if False, use features from 224x224 resized images.
        N (int): Number of comparison partners per painting for pair
            generation (passed to generate_comparative_judgments).

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

    # Generate pairwise comparisons from the leave-one-out average ratings
    features = generate_comparative_judgments(X[train], y_train[train], N=N)

    encoder = build_model((X[train].shape[1]))
    model = ComparativeModel(encoder)
    checkpoint_path = "checkpoint/comparative.weights.h5"

    model.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )

    # Train model on pairwise comparisons for 100 epochs
    history = model.fit(features, features["Label"], epochs=100, batch_size=10, callbacks=[reduce_lr, checkpoint],
                        verbose=1)
    # Restore best weights and evaluate on the test set
    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)

    # Use encoder alone to predict latent scores for test paintings
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
