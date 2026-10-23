#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Artifact synchronization itself does not parse YAML. Keep this unit test
# runnable in a bare checkout without installing the full training stack.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))

from train_openwakeword import (  # noqa: E402
    make_config,
    merge_corrective_negative_paths,
    partition_uploaded_audio,
    quarantine_existing_training_exports,
    run_training_with_cpu_fallback,
    stage_corrective_negative_clips,
    stage_reviewed_negative_clips,
    stage_shared_positive_clips,
    sync_artifacts,
)


class ArtifactExportTests(unittest.TestCase):
    def test_prior_onnx_is_quarantined_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "hey_tater"
            output.mkdir()
            model = output / "hey_tater.onnx"
            sidecar = output / "weights.data"
            model.write_bytes(b"old-model")
            sidecar.write_bytes(b"old-weights")

            quarantined = quarantine_existing_training_exports(model)

            self.assertFalse(model.exists())
            self.assertFalse(sidecar.exists())
            self.assertEqual(
                sorted(path.read_bytes() for path in quarantined),
                [b"old-model", b"old-weights"],
            )

    def test_mps_failure_retries_on_cpu_and_requires_a_fresh_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hey_tater.onnx"
            env = {"OWW_ENABLE_MPS": "1", "KEEP_ME": "yes"}
            calls: list[dict[str, str]] = []

            def fake_run(_cmd, env=None):
                calls.append(dict(env or {}))
                if len(calls) == 1:
                    model.write_bytes(b"incomplete-mps-model")
                    return types.SimpleNamespace(returncode=1)
                self.assertFalse(model.exists())
                model.write_bytes(b"fresh-cpu-model")
                return types.SimpleNamespace(returncode=0)

            with (
                patch("train_openwakeword.sys.platform", "darwin"),
                patch("train_openwakeword.run_process", side_effect=fake_run),
            ):
                run_training_with_cpu_fallback(
                    ["python", "train.py"],
                    env=env,
                    expected_onnx=model,
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["OWW_ENABLE_MPS"], "1")
            self.assertEqual(calls[1]["OWW_ENABLE_MPS"], "0")
            self.assertEqual(calls[1]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["OWW_ENABLE_MPS"], "0")
            self.assertEqual(env["KEEP_ME"], "yes")
            self.assertEqual(model.read_bytes(), b"fresh-cpu-model")

    def test_failed_cpu_retry_cannot_publish_a_partial_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hey_tater.onnx"

            def failed_run(_cmd, env=None):
                model.write_bytes(b"partial")
                return types.SimpleNamespace(returncode=1)

            with (
                patch("train_openwakeword.sys.platform", "darwin"),
                patch("train_openwakeword.run_process", side_effect=failed_run),
                self.assertRaises(SystemExit),
            ):
                run_training_with_cpu_fallback(
                    ["python", "train.py"],
                    env={"OWW_ENABLE_MPS": "1"},
                    expected_onnx=model,
                )

            self.assertFalse(model.exists())

    def test_uploaded_audio_gets_a_stable_holdout_at_five_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "uploaded"
            source.mkdir()
            for index in range(5):
                (source / f"sample-{index}.wav").write_bytes(f"wav-{index}".encode())

            train, holdout = partition_uploaded_audio(source, root / "work", "personal")

            self.assertIsNotNone(holdout)
            self.assertEqual(len(list(train.glob("*.wav"))), 4)
            self.assertEqual(len(list(holdout.glob("*.wav"))), 1)

    def test_corrective_training_does_not_consume_real_device_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "generated"
            holdout = root / "holdout"
            generated.mkdir()
            holdout.mkdir()
            generated_clip = generated / "false-trigger.wav"
            holdout_clip = holdout / "real-device.wav"
            generated_clip.write_bytes(b"generated")
            holdout_clip.write_bytes(b"holdout")

            staged = stage_corrective_negative_clips(
                [str(generated_clip), str(holdout_clip)],
                root / "retry",
                excluded_roots=[holdout],
            )

            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0].read_bytes(), b"generated")

    def test_corrective_retries_accumulate_false_triggers(self) -> None:
        first = merge_corrective_negative_paths(
            [],
            {"misclassified": {"triggered_negative_paths": ["first.wav"]}},
        )
        second = merge_corrective_negative_paths(
            first,
            {
                "misclassified": {
                    "triggered_negative_paths": ["first.wav", "second.wav"]
                }
            },
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        self.assertTrue(second[0].endswith("first.wav"))
        self.assertTrue(second[1].endswith("second.wav"))

    def test_config_reads_broad_negatives_from_shared_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "shared"
            bank.mkdir()
            feature = bank / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
            feature.write_bytes(b"features")
            args = types.SimpleNamespace(
                data_dir=str(root / "assets"), piper_dir=str(root / "piper"),
                feature_bank_dir=str(bank), background_dir=[], rir_dir=[],
                positive_batch=50, adversarial_batch=70, custom_negative_batch=30,
                custom_negative_fraction=0.30, negative_batch=1000,
                phrase="hey tater", custom_negative_phrase=["hey later"], samples=10,
                validation_samples=2, tts_batch_size=1, negative_tts_batch_divisor=1,
                augmentation_batch_size=1, validation_batch_size=1,
                augmentation_rounds=1, model_type="dnn", layer_size=32, steps=1,
                target_accuracy=0.7, target_recall=0.5, max_negative_weight=10,
                target_fp_per_hour=0.2,
            )

            config = make_config(args, "hey_tater", root / "output")

            self.assertEqual(config["feature_data_files"]["ACAV100M_sample"], str(feature.resolve()))
            self.assertEqual(config["batch_n_per_class"]["adversarial_negative"], 70)
            self.assertEqual(config["batch_n_per_class"]["custom_negative"], 30)
            self.assertEqual(config["custom_negative_phrases"], [])
            self.assertEqual(config["weighted_custom_negative_phrases"], ["hey later"])

    def test_shared_positive_corpus_is_staged_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shared"
            output = root / "output"
            source.mkdir()
            for index in range(5):
                (source / f"sample-{index}.wav").write_bytes(f"wav-{index}".encode())

            stage_shared_positive_clips(output, "hey_tater", source, 3, 2)

            clip_root = output / "hey_tater"
            self.assertEqual(len(list((clip_root / "positive_train").glob("*.wav"))), 3)
            self.assertEqual(len(list((clip_root / "positive_test").glob("*.wav"))), 2)
            feature = clip_root / "positive_features_train.npy"
            feature.write_bytes(b"cached")

            stage_shared_positive_clips(output, "hey_tater", source, 3, 2)
            self.assertTrue(feature.exists())

            (source / "sample-0.wav").write_bytes(b"changed")
            stage_shared_positive_clips(output, "hey_tater", source, 3, 2)
            self.assertFalse(feature.exists())

    def test_reviewed_negatives_join_generated_training_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = root / "reviewed"
            reviewed.mkdir()
            (reviewed / "false-wake.wav").write_bytes(b"negative")
            feature = root / "output" / "hey_tater" / "reviewed_negative_features_train.npy"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b"stale")

            stage_reviewed_negative_clips(root / "output", "hey_tater", reviewed)

            links = list((root / "output" / "hey_tater" / "reviewed_negative_train").glob("*.wav"))
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0].read_bytes(), b"negative")
            self.assertFalse(feature.exists())

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
