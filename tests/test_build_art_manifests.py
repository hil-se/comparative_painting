from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "extensions"
    / "build_art_manifests.py"
)
SPEC = importlib.util.spec_from_file_location("build_art_manifests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ApddManifestMissingImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.annotations = self.root / "annotations.csv"
        fieldnames = (
            "filename",
            "Artistic Categories",
            *MODULE.APDD_TARGETS,
        )
        rows = []
        for filename in ("present.jpg", "missing.jpg"):
            row = {
                "filename": filename,
                "Artistic Categories": "painting",
            }
            row.update({target: "1" for target in MODULE.APDD_TARGETS})
            rows.append(row)
        with self.annotations.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (self.images / "present.jpg").write_bytes(b"test")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_one_missing_image_is_recorded_when_allowed(self) -> None:
        output = self.root / "manifest.csv"
        summary = MODULE.build_apdd(
            self.annotations,
            self.images,
            output,
            max_missing_images=1,
        )
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["excluded_missing_image_count"], 1)
        self.assertEqual(
            summary["excluded_missing_images"], ["missing.jpg"]
        )

    def test_missing_image_exceeding_limit_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            MODULE.build_apdd(
                self.annotations,
                self.images,
                self.root / "manifest.csv",
                max_missing_images=0,
            )


if __name__ == "__main__":
    unittest.main()
