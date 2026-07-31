from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np


DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("scipy", "sklearn")
)
MODULE = None
if DEPENDENCIES_AVAILABLE:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "extensions"
        / "run_art_extensions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_art_extensions_loading", module_path
    )
    assert spec is not None and spec.loader is not None
    MODULE = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MODULE
    spec.loader.exec_module(MODULE)


class CountingArchive:
    def __init__(self, values):
        self.values = values
        self.reads = Counter()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def __getitem__(self, key):
        self.reads[key] += 1
        return self.values[key]


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "SciPy/scikit-learn are not installed")
class DatasetLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("item_id", "score", "category")
            )
            writer.writeheader()
            for index in range(30):
                writer.writerow(
                    {
                        "item_id": f"item-{index}",
                        "score": index,
                        "category": "painting",
                    }
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_compressed_feature_members_are_read_once(self) -> None:
        assert MODULE is not None
        archive = CountingArchive(
            {
                "item_ids": np.asarray([f"item-{index}" for index in range(30)]),
                "features": np.arange(120, dtype=np.float32).reshape(30, 4),
            }
        )
        with mock.patch.object(MODULE.np, "load", return_value=archive):
            dataset = MODULE.load_dataset(
                self.manifest, self.root / "features.npz", "score"
            )

        self.assertEqual(archive.reads, {"item_ids": 1, "features": 1})
        self.assertEqual(dataset.features.shape, (30, 4))
        np.testing.assert_array_equal(dataset.features[29], [116, 117, 118, 119])


if __name__ == "__main__":
    unittest.main()
