from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
MODULE = None
if TENSORFLOW_AVAILABLE:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "extensions"
        / "run_art_extensions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_art_extensions", module_path
    )
    assert spec is not None and spec.loader is not None
    MODULE = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MODULE
    spec.loader.exec_module(MODULE)


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow is not installed")
class PairwiseLossShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        import tensorflow as tf

        assert MODULE is not None
        self.tf = tf
        self.labels = tf.constant([[1.0], [-1.0], [1.0], [-1.0]])
        self.differences = tf.constant([[2.0], [-2.0], [-0.5], [0.5]])

    def test_hinge_loss_is_elementwise_for_column_tensors(self) -> None:
        expected = np.mean([0.0, 0.0, 1.5, 1.5])
        actual = MODULE.hinge_pairwise_loss(
            self.labels, self.differences
        ).numpy()
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_bradley_terry_loss_is_elementwise_for_column_tensors(self) -> None:
        expected = np.mean(np.logaddexp(0.0, -np.array([2.0, 2.0, -0.5, -0.5])))
        actual = MODULE.bradley_terry_pairwise_loss(
            self.labels, self.differences
        ).numpy()
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_mismatched_pair_counts_raise(self) -> None:
        with self.assertRaises(self.tf.errors.InvalidArgumentError):
            MODULE.hinge_pairwise_loss(
                self.labels,
                self.tf.constant([[1.0], [2.0], [3.0]]),
            )


if __name__ == "__main__":
    unittest.main()
