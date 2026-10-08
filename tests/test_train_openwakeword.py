#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Artifact synchronization itself does not parse YAML. Keep this unit test
# runnable in a bare checkout without installing the full training stack.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))

from train_openwakeword import sync_artifacts  # noqa: E402


class ArtifactExportTests(unittest.TestCase):
    def test_sync_publishes_onnx_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            exported = root / "exported"
            output.mkdir()
            exported.mkdir()
            (output / "hey_tater.onnx").write_bytes(b"onnx")

            model = sync_artifacts(output, exported, "hey_tater", {})

            self.assertEqual(model, exported / "hey_tater.onnx")
            metadata = json.loads((exported / "hey_tater.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["artifacts"], ["hey_tater.onnx"])


if __name__ == "__main__":
    unittest.main()
