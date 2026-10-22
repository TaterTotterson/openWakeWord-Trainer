import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingDependencyTests(unittest.TestCase):
    def test_linux_training_does_not_resolve_tflite_runtime(self) -> None:
        requirements = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
        launcher = (ROOT / "train_openwakeword.sh").read_text(encoding="utf-8")

        self.assertNotIn("openwakeword==", requirements.lower())
        self.assertIn("scikit-learn==", requirements.lower())
        self.assertIn(
            'pip install --no-build-isolation --no-deps -e "$OPENWAKEWORD_DIR"',
            launcher,
        )
        self.assertIn('OWW_ONNX_RUNTIME_VERSION:-1.22.0', launcher)
        self.assertIn('onnxruntime==1.22.0', requirements)

    def test_upstream_patch_streams_false_positive_validation_batches(self) -> None:
        patcher = (ROOT / "scripts" / "patch_openwakeword_device.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("FalsePositiveValidationDataset", patcher)
        self.assertIn('mmap_mode="r"', patcher)
        self.assertIn('config.get("validation_batch_size", 2048)', patcher)

    def test_upstream_patch_defaults_feature_extraction_to_cpu_onnx(self) -> None:
        patcher = (ROOT / "scripts" / "patch_openwakeword_device.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("OWW_FEATURE_DEVICE", patcher)
        self.assertIn('os.environ.get("OWW_FEATURE_DEVICE", "cpu")', patcher)

    def test_upstream_patch_makes_training_reproducible(self) -> None:
        patcher = (ROOT / "scripts" / "patch_openwakeword_device.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("OWW_TRAIN_SEED", patcher)
        self.assertIn("torch.manual_seed(seed)", patcher)
        self.assertIn("torch.cuda.manual_seed_all(seed)", patcher)
        self.assertIn("torch.use_deterministic_algorithms(True, warn_only=True)", patcher)

    def test_upstream_patch_updates_false_positive_tracking(self) -> None:
        patcher = (ROOT / "scripts" / "patch_openwakeword_device.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("self.best_val_fp = min(", patcher)
        self.assertIn("float(np.asarray(val_fp_per_hr).item())", patcher)


if __name__ == "__main__":
    unittest.main()
