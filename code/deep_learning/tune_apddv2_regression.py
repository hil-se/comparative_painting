"""Validation-safe APDDv2 CLIP regression tuning and locked test evaluation.

The ``screen`` and ``confirm`` phases never pass test features or labels to a
model or metric. The ``test`` phase accepts exactly one previously locked
configuration and is intended to run only after validation-based selection.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from run_art_extensions import load_dataset, seed_everything, split_indices


APDD_TARGETS = (
    "Total aesthetic score",
    "Theme and logic",
    "Creativity",
    "Layout and composition",
    "Space and perspective",
    "The sense of order",
    "Light and shadow",
    "Color",
    "Details and texture",
    "The overall",
    "Mood",
)


@dataclass(frozen=True)
class RegressionConfig:
    config_id: str
    architecture: str
    loss: str = "mse"
    learning_rate: float = 1e-3
    activation: str = "relu"
    normalization: str = "batch"
    dropout: float = 0.1
    l2: float = 1e-5
    batch_size: int = 128
    target_standardization: bool = True
    feature_standardization: bool = True
    reduce_lr: bool = True


CONFIGS = (
    RegressionConfig(
        "deep-mae-current-spearstop",
        "deep",
        loss="mae",
        dropout=0.25,
        batch_size=256,
        target_standardization=False,
        reduce_lr=False,
    ),
    RegressionConfig(
        "deep-mae-lr3e4-d25",
        "deep",
        loss="mae",
        learning_rate=3e-4,
        dropout=0.25,
        batch_size=256,
        target_standardization=False,
    ),
    RegressionConfig(
        "deep-huber-lr1e3-d25",
        "deep",
        loss="huber",
        dropout=0.25,
        batch_size=256,
        target_standardization=False,
    ),
    RegressionConfig(
        "deep-huber-lr3e4-d25",
        "deep",
        loss="huber",
        learning_rate=3e-4,
        dropout=0.25,
        batch_size=256,
        target_standardization=False,
    ),
    RegressionConfig(
        "deep-mse-z-lr1e3-d25",
        "deep",
        dropout=0.25,
        batch_size=256,
    ),
    RegressionConfig(
        "deep-mse-z-lr3e4-d25",
        "deep",
        learning_rate=3e-4,
        dropout=0.25,
        batch_size=256,
    ),
    RegressionConfig("deep-mae-z-d10-b128", "deep", loss="mae"),
    RegressionConfig("deep-huber-z-d10-b128", "deep", loss="huber"),
    RegressionConfig("deep-mse-z-d10-b128", "deep"),
    RegressionConfig(
        "shallow-mae-z-bn-d10", "shallow", loss="mae"
    ),
    RegressionConfig(
        "shallow-huber-z-bn-d10", "shallow", loss="huber"
    ),
    RegressionConfig("shallow-mse-z-bn-d10", "shallow"),
    RegressionConfig(
        "shallow-mse-z-ln-d10",
        "shallow",
        normalization="layer",
    ),
    RegressionConfig(
        "shallow-huber-z-ln-d10",
        "shallow",
        loss="huber",
        normalization="layer",
    ),
    RegressionConfig(
        "shallow-mse-z-gelu-ln",
        "shallow",
        activation="gelu",
        normalization="layer",
    ),
    RegressionConfig(
        "shallow-mse-z-gelu-ln-rawclip",
        "shallow",
        activation="gelu",
        normalization="layer",
        feature_standardization=False,
    ),
    RegressionConfig(
        "bottleneck-mse-z-ln",
        "bottleneck",
        normalization="layer",
    ),
    RegressionConfig(
        "deep-mse-z-gelu-ln-d10",
        "deep",
        activation="gelu",
        normalization="layer",
    ),
    RegressionConfig(
        "ridge-z-a1",
        "ridge",
        l2=1.0,
        dropout=0.0,
        normalization="none",
        reduce_lr=False,
    ),
    RegressionConfig(
        "ridge-z-a10",
        "ridge",
        l2=10.0,
        dropout=0.0,
        normalization="none",
        reduce_lr=False,
    ),
    RegressionConfig(
        "ridge-z-a100",
        "ridge",
        l2=100.0,
        dropout=0.0,
        normalization="none",
        reduce_lr=False,
    ),
    RegressionConfig(
        "ridge-z-a1-rawclip",
        "ridge",
        l2=1.0,
        dropout=0.0,
        normalization="none",
        feature_standardization=False,
        reduce_lr=False,
    ),
)
CONFIG_BY_ID = {config.config_id: config for config in CONFIGS}


def parse_range(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(piece) for piece in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def selected_config_ids(args: argparse.Namespace) -> list[str]:
    if args.config_ids_file:
        payload = json.loads(args.config_ids_file.read_text(encoding="utf-8"))
        if "selected_config_ids" in payload:
            ids = payload["selected_config_ids"]
        elif "winner_config_id" in payload:
            ids = [payload["winner_config_id"]]
        else:
            raise ValueError(
                "Selection JSON needs selected_config_ids or winner_config_id"
            )
    elif args.config_ids:
        ids = args.config_ids.split(",")
    else:
        ids = [config.config_id for config in CONFIGS]

    unknown = sorted(set(ids) - set(CONFIG_BY_ID))
    if unknown:
        raise ValueError(f"Unknown configuration IDs: {unknown}")
    if len(ids) != len(set(ids)):
        raise ValueError("Configuration IDs must be unique")
    if args.phase == "test" and len(ids) != 1:
        raise ValueError("The locked test phase requires exactly one config")
    return ids


def scale_features(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray | None,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not enabled:
        return train, validation, test
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    train_scaled = ((train - mean) / scale).astype(np.float32)
    validation_scaled = ((validation - mean) / scale).astype(np.float32)
    test_scaled = (
        None
        if test is None
        else ((test - mean) / scale).astype(np.float32)
    )
    return train_scaled, validation_scaled, test_scaled


def scale_targets(
    train: np.ndarray, validation: np.ndarray, enabled: bool
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not enabled:
        return train, validation, 0.0, 1.0
    mean = float(train.mean())
    scale = float(train.std())
    if scale == 0:
        scale = 1.0
    return (
        ((train - mean) / scale).astype(np.float32),
        ((validation - mean) / scale).astype(np.float32),
        mean,
        scale,
    )


def correlation(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> float:
    if kind == "spearman":
        value = spearmanr(y_true, y_pred).statistic
    elif kind == "pearson":
        value = pearsonr(y_true, y_pred).statistic
    else:
        raise ValueError(kind)
    return float(value)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}_spearman": correlation(y_true, y_pred, "spearman"),
        f"{prefix}_pearson": correlation(y_true, y_pred, "pearson"),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_mse": float(mean_squared_error(y_true, y_pred)),
    }


def build_model(input_dim: int, config: RegressionConfig):
    import tensorflow as tf

    hidden_sizes = {
        "deep": (512, 256, 128),
        "shallow": (256, 64),
        "bottleneck": (512, 128),
    }[config.architecture]
    layers: list[object] = [tf.keras.layers.Input(shape=(input_dim,))]
    for index, units in enumerate(hidden_sizes):
        regularizer = (
            tf.keras.regularizers.l2(config.l2)
            if config.l2 and index < len(hidden_sizes) - 1
            else None
        )
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation=config.activation,
                kernel_regularizer=regularizer,
            )
        )
        if config.normalization == "batch":
            layers.append(tf.keras.layers.BatchNormalization())
        elif config.normalization == "layer":
            layers.append(tf.keras.layers.LayerNormalization())
        elif config.normalization != "none":
            raise ValueError(config.normalization)
        if config.dropout and index < len(hidden_sizes) - 1:
            layers.append(tf.keras.layers.Dropout(config.dropout))
    layers.append(tf.keras.layers.Dense(1, activation="linear"))
    return tf.keras.Sequential(layers)


def keras_loss(config: RegressionConfig):
    import tensorflow as tf

    if config.loss == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    if config.loss == "mse":
        return tf.keras.losses.MeanSquaredError()
    if config.loss == "huber":
        return tf.keras.losses.Huber(delta=1.0)
    raise ValueError(config.loss)


def make_spearman_stopper(
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    patience: int,
    minimum_epochs: int,
):
    import tensorflow as tf

    class SpearmanEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best = -np.inf
            self.best_epoch = 0
            self.best_weights: list[np.ndarray] | None = None
            self.wait = 0

        def on_epoch_end(self, epoch, logs=None) -> None:
            logs = {} if logs is None else logs
            prediction = np.asarray(
                self.model(validation_features, training=False)
            ).ravel()
            score = correlation(
                validation_targets, prediction, kind="spearman"
            )
            logs["val_spearman"] = score
            if np.isfinite(score) and score > self.best + 1e-4:
                self.best = score
                self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
                self.wait = 0
            else:
                self.wait += 1
            if epoch + 1 >= minimum_epochs and self.wait >= patience:
                self.model.stop_training = True

        def on_train_end(self, logs=None) -> None:
            if self.best_weights is None:
                raise RuntimeError("No finite validation Spearman was observed")
            self.model.set_weights(self.best_weights)

    return SpearmanEarlyStopping()


def fit_predict(
    config: RegressionConfig,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    test_features: np.ndarray | None,
    epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    fit_train_targets, fit_validation_targets, target_mean, target_scale = (
        scale_targets(
            train_targets,
            validation_targets,
            config.target_standardization,
        )
    )

    if config.architecture == "ridge":
        model = Ridge(alpha=config.l2)
        model.fit(train_features, fit_train_targets)
        validation_prediction = model.predict(validation_features)
        test_prediction = (
            None if test_features is None else model.predict(test_features)
        )
        epochs_trained = best_epoch = 0
    else:
        import tensorflow as tf

        model = build_model(train_features.shape[1], config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate
            ),
            loss=keras_loss(config),
            jit_compile=False,
        )
        stopper = make_spearman_stopper(
            validation_features,
            fit_validation_targets,
            patience=patience,
            minimum_epochs=min(25, epochs),
        )
        callbacks: list[object] = [stopper]
        if config.reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_spearman",
                    mode="max",
                    factor=0.5,
                    patience=max(4, patience // 2),
                    min_delta=1e-4,
                    min_lr=1e-6,
                    verbose=0,
                )
            )
        history = model.fit(
            train_features,
            fit_train_targets,
            validation_data=(validation_features, fit_validation_targets),
            epochs=epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        validation_prediction = np.asarray(
            model(validation_features, training=False)
        ).ravel()
        test_prediction = (
            None
            if test_features is None
            else np.asarray(model(test_features, training=False)).ravel()
        )
        epochs_trained = len(history.epoch)
        best_epoch = stopper.best_epoch
        del history, stopper
        tf.keras.backend.clear_session()

    validation_prediction = (
        np.asarray(validation_prediction).ravel() * target_scale + target_mean
    )
    if test_prediction is not None:
        test_prediction = (
            np.asarray(test_prediction).ravel() * target_scale + target_mean
        )
    del model
    gc.collect()
    return validation_prediction, test_prediction, epochs_trained, best_epoch


def run(
    phase: str,
    manifest: Path,
    feature_file: Path,
    target: str,
    config_ids: Iterable[str],
    seeds: Iterable[int],
    epochs: int,
    patience: int,
) -> list[dict[str, object]]:
    data = load_dataset(manifest, feature_file, target)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train_indices, validation_indices, heldout_indices = split_indices(
            len(data.ratings), "apddv2", seed
        )
        raw_train = data.features[train_indices]
        raw_validation = data.features[validation_indices]
        raw_test = (
            data.features[heldout_indices] if phase == "test" else None
        )
        train_targets = data.ratings[train_indices]
        validation_targets = data.ratings[validation_indices]

        feature_views: dict[
            bool, tuple[np.ndarray, np.ndarray, np.ndarray | None]
        ] = {}
        for config_id in config_ids:
            config = CONFIG_BY_ID[config_id]
            if config.feature_standardization not in feature_views:
                feature_views[config.feature_standardization] = scale_features(
                    raw_train,
                    raw_validation,
                    raw_test,
                    config.feature_standardization,
                )
            train_features, validation_features, test_features = feature_views[
                config.feature_standardization
            ]
            seed_everything(seed)
            validation_prediction, test_prediction, trained, best = fit_predict(
                config,
                train_features,
                train_targets,
                validation_features,
                validation_targets,
                test_features,
                epochs,
                patience,
            )
            row: dict[str, object] = {
                "phase": phase,
                "dataset": "apddv2",
                "representation": "clip-vit-b32",
                "target": target,
                "config_id": config_id,
                "seed": seed,
                "train_examples": len(train_indices),
                "validation_examples": len(validation_indices),
                "epochs_trained": trained,
                "best_epoch": best,
                **regression_metrics(
                    validation_targets, validation_prediction, "validation"
                ),
            }
            if phase == "test":
                if test_prediction is None:
                    raise AssertionError("Locked test prediction is missing")
                row.update(
                    {
                        "test_examples": len(heldout_indices),
                        **regression_metrics(
                            data.ratings[heldout_indices],
                            test_prediction,
                            "test",
                        ),
                    }
                )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "confirm", "test"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--target", choices=APDD_TARGETS, required=True)
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--config-ids")
    parser.add_argument("--config-ids-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_ids and args.config_ids_file:
        raise ValueError("Use only one config selection argument")
    config_ids = selected_config_ids(args)
    seeds = parse_range(args.seeds)
    rows = run(
        phase=args.phase,
        manifest=args.manifest,
        feature_file=args.features,
        target=args.target,
        config_ids=config_ids,
        seeds=seeds,
        epochs=args.epochs,
        patience=args.patience,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "phase": args.phase,
        "test_access": args.phase == "test",
        "selection_metric": "unweighted macro validation Spearman",
        "target": args.target,
        "config_ids": config_ids,
        "configs": [asdict(CONFIG_BY_ID[value]) for value in config_ids],
        "seeds": seeds,
        "rows": len(rows),
        "epochs": args.epochs,
        "patience": args.patience,
        "split": "70% train / 15% validation / 15% held-out test",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
