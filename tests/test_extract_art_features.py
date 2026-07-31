from __future__ import annotations

import io
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "extensions"
    / "extract_art_features.py"
)
SPEC = importlib.util.spec_from_file_location("extract_art_features", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WebpFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary_directory.name) / "mislabeled.jpg"
        self.source.write_bytes(b"RIFF\x04\x00\x00\x00WEBP")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_uses_imagemagick_for_unsupported_webp(self) -> None:
        output = io.BytesIO()
        Image.new("RGBA", (3, 2), (10, 20, 30, 128)).save(
            output, format="PNG"
        )
        completed = subprocess.CompletedProcess(
            args=["/usr/bin/convert"],
            returncode=0,
            stdout=output.getvalue(),
            stderr=b"",
        )
        with mock.patch.object(MODULE.shutil, "which", return_value="/usr/bin/convert"):
            with mock.patch.object(MODULE.subprocess, "run", return_value=completed):
                image = MODULE.load_image_rgb(self.source)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (3, 2))

    def test_reports_missing_webp_decoder(self) -> None:
        with mock.patch.object(MODULE.shutil, "which", return_value=None):
            with self.assertRaisesRegex(
                RuntimeError, "ImageMagick 'convert' is unavailable"
            ):
                MODULE.load_image_rgb(self.source)


if __name__ == "__main__":
    unittest.main()
