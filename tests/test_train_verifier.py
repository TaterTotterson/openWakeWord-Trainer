from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_verifier.py"


def load_script():
    spec = importlib.util.spec_from_file_location("train_verifier_script", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainVerifierTests(unittest.TestCase):
    def test_uses_absolute_base_model_with_onnx_runtime(self) -> None:
        module = load_script()
        calls: dict[str, object] = {}
        fake_openwakeword = types.ModuleType("openwakeword")

        def fake_trainer(**kwargs):
            calls.update(kwargs)

        fake_openwakeword.train_custom_verifier = fake_trainer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models" / "jarvis.onnx"
            positive_dir = root / "positive"
            negative_dir = root / "negative"
            output = root / "output" / "jarvis_verifier.pkl"
            model.parent.mkdir()
            positive_dir.mkdir()
            negative_dir.mkdir()
            model.write_bytes(b"onnx")
            (positive_dir / "positive.wav").write_bytes(b"wav")
            (negative_dir / "negative.wav").write_bytes(b"wav")

            argv = [
                str(SCRIPT),
                "--base-model",
                str(model),
                "--positive-dir",
                str(positive_dir),
                "--negative-dir",
                str(negative_dir),
                "--output",
                str(output),
            ]
            with mock.patch.dict(sys.modules, {"openwakeword": fake_openwakeword}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(module.main(), 0)

        self.assertEqual(calls["model_name"], str(model.resolve()))
        self.assertEqual(calls["inference_framework"], "onnx")


if __name__ == "__main__":
    unittest.main()
