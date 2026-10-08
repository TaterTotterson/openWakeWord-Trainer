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


if __name__ == "__main__":
    unittest.main()
