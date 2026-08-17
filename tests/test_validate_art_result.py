from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "extensions"
    / "validate_art_result.py"
)
SPEC = importlib.util.spec_from_file_location("validate_art_result", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResultValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.result = self.root / "result.csv"
        self.features = self.root / "features.npz"
        self.features.write_bytes(b"features")
        rows = [
            {
                "dataset": "sidhu",
                "representation": "clip-vit-b32",
                "category": "abstract",
                "target": "beauty",
                "objective": "hinge",
                "N": str(n_value),
                "seed": "3",
            }
            for n_value in (1, 2)
        ]
        with self.result.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "dataset": "sidhu",
            "representation": "clip-vit-b32",
            "category": "abstract",
            "target": "beauty",
            "objectives": ["hinge"],
            "n_values": [1, 2],
            "seeds": [3],
            "rows": 2,
            "sha256": hashlib.sha256(self.result.read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(self.features.read_bytes()).hexdigest(),
        }
        self.result.with_suffix(".metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_validates_expected_configuration_and_provenance(self) -> None:
        MODULE.validate(
            self.result,
            2,
            expected_fields={"objective": "hinge", "seed": "3"},
            features=self.features,
        )

    def test_rejects_wrong_expected_objective(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected objective"):
            MODULE.validate(
                self.result,
                2,
                expected_fields={"objective": "bradley_terry"},
            )

    def test_rejects_provenance_mismatch(self) -> None:
        self.features.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "features_sha256 mismatch"):
            MODULE.validate(self.result, 2, features=self.features)


if __name__ == "__main__":
    unittest.main()
