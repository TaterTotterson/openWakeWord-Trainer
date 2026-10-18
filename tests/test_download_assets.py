from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_assets import download_file, file_is_valid


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


class DownloadAssetsTests(unittest.TestCase):
    def test_verified_download_replaces_invalid_cached_response(self) -> None:
        payload = b"verified checkpoint"
        expected_sha256 = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "en_us_cmudict_forward.pt"
            destination.write_text("AccessDenied", encoding="utf-8")

            with patch(
                "scripts.download_assets.urllib.request.urlopen",
                return_value=FakeResponse(payload),
            ):
                download_file(
                    "https://example.test/checkpoint.pt",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=expected_sha256,
                )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".pt.part").exists())
            self.assertTrue(
                file_is_valid(
                    destination,
                    expected_size=len(payload),
                    expected_sha256=expected_sha256,
                )
            )

    def test_bad_download_does_not_replace_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "checkpoint.pt"
            destination.write_bytes(b"existing invalid cache")

            with patch(
                "scripts.download_assets.urllib.request.urlopen",
                return_value=FakeResponse(b"wrong download"),
            ):
                with self.assertRaises(RuntimeError):
                    download_file(
                        "https://example.test/checkpoint.pt",
                        destination,
                        expected_size=123,
                        expected_sha256="0" * 64,
                    )

            self.assertEqual(destination.read_bytes(), b"existing invalid cache")
            self.assertFalse(destination.with_suffix(".pt.part").exists())


if __name__ == "__main__":
    unittest.main()
