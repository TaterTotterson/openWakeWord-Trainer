#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(message, flush=True)


def safe_name(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "wakeword"


def run(cmd: list[str], env: dict[str, str] | None = None, allow_onnx_failure: Path | None = None) -> None:
    log("")
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)
    if result.returncode == 0:
        return
    if allow_onnx_failure and allow_onnx_failure.exists():
        log(f"WARNING: command exited {result.returncode}, but ONNX exists at {allow_onnx_failure}")
        return
    raise SystemExit(result.returncode)


def existing_wav_dir(path: Path) -> bool:
    return path.exists() and any(path.glob("*.wav"))


def make_config(args: argparse.Namespace, model_name: str, output_dir: Path) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    piper_dir = Path(args.piper_dir).resolve()
    features = data_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation = data_dir / "validation_set_features.npy"

    background_paths = []
    for candidate in (data_dir / "audioset_16k", data_dir / "fma", data_dir / "background_clips"):
        if existing_wav_dir(candidate):
            background_paths.append(str(candidate))

    if not background_paths:
        fallback = data_dir / "background_clips"
        fallback.mkdir(parents=True, exist_ok=True)
        background_paths.append(str(fallback))
        log(f"WARNING: no background WAVs found; using empty {fallback}")

    feature_data_files: dict[str, str] = {}
    batch_n_per_class: dict[str, int] = {
        "positive": args.positive_batch,
        "adversarial_negative": args.adversarial_batch,
    }
    if features.exists():
        feature_data_files["ACAV100M_sample"] = str(features)
        batch_n_per_class["ACAV100M_sample"] = args.negative_batch
    else:
        log("WARNING: full negative feature file is missing; false-positive performance will be weaker")

    if not validation.exists():
        log("WARNING: validation_set_features.npy is missing; run scripts/download_assets.py")

    return {
        "model_name": model_name,
        "target_phrase": [args.phrase],
        "custom_negative_phrases": args.custom_negative_phrase,
        "n_samples": args.samples,
        "n_samples_val": args.validation_samples,
        "tts_batch_size": args.tts_batch_size,
        "negative_tts_batch_divisor": args.negative_tts_batch_divisor,
        "augmentation_batch_size": args.augmentation_batch_size,
        "piper_sample_generator_path": str(piper_dir),
        "output_dir": str(output_dir),
        "rir_paths": [str(data_dir / "mit_rirs")],
        "background_paths": background_paths,
        "background_paths_duplication_rate": [1 for _ in background_paths],
        "false_positive_validation_data_path": str(validation),
        "validation_batch_size": args.validation_batch_size,
        "augmentation_rounds": args.augmentation_rounds,
        "feature_data_files": feature_data_files,
        "batch_n_per_class": batch_n_per_class,
        "model_type": args.model_type,
        "layer_size": args.layer_size,
        "steps": args.steps,
        "target_accuracy": args.target_accuracy,
        "target_recall": args.target_recall,
        "max_negative_weight": args.max_negative_weight,
        "target_false_positives_per_hour": args.target_fp_per_hour,
    }


def sync_artifacts(output_dir: Path, export_dir: Path, model_name: str, metadata: dict[str, Any]) -> Path | None:
    export_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for path in sorted(output_dir.glob("*")):
        if path.suffix.lower() not in {".onnx", ".data"}:
            continue
        dest = export_dir / path.name
        shutil.copy2(path, dest)
        copied.append(dest.name)

    onnx_path = export_dir / f"{model_name}.onnx"
    metadata["artifacts"] = copied
    metadata["synced_at"] = datetime.now(timezone.utc).isoformat()
    (export_dir / f"{model_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if copied:
        log("Synced artifacts:")
        for name in copied:
            log(f"  {export_dir / name}")
    else:
        log(f"WARNING: no model artifacts found in {output_dir}")

    return onnx_path if onnx_path.exists() else None


def generated_feature_files(output_dir: Path, model_name: str) -> list[Path]:
    feature_dir = output_dir / model_name
    return [
        feature_dir / "positive_features_train.npy",
        feature_dir / "negative_features_train.npy",
        feature_dir / "positive_features_test.npy",
        feature_dir / "negative_features_test.npy",
    ]


def generated_clip_dirs(output_dir: Path, model_name: str) -> list[Path]:
    clip_dir = output_dir / model_name
    return [
        clip_dir / "positive_train",
        clip_dir / "positive_test",
        clip_dir / "negative_train",
        clip_dir / "negative_test",
    ]


def normalize_generated_clip_sample_rates(output_dir: Path, model_name: str, sample_rate: int = 16000) -> None:
    import numpy as np
    import scipy.io.wavfile
    import scipy.signal

    changed = 0
    checked = 0
    for directory in generated_clip_dirs(output_dir, model_name):
        if not directory.exists():
            continue
        for wav_path in directory.glob("*.wav"):
            checked += 1
            try:
                with wave.open(str(wav_path), "rb") as wav:
                    current_rate = wav.getframerate()
            except Exception:
                continue
            if current_rate == sample_rate:
                continue

            current_rate, audio = scipy.io.wavfile.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            gcd = math.gcd(int(current_rate), sample_rate)
            resampled = scipy.signal.resample_poly(
                audio.astype(np.float32),
                sample_rate // gcd,
                int(current_rate) // gcd,
            )
            resampled = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)
            scipy.io.wavfile.write(wav_path, sample_rate, resampled)
            changed += 1

    if changed:
        log(f"Resampled {changed} existing generated clip(s) to {sample_rate} Hz")
    elif checked:
        log(f"Generated clips already use {sample_rate} Hz")


def train_verifier(args: argparse.Namespace, base_model: Path, model_name: str) -> None:
    positive_dir = Path(args.positive_dir).resolve()
    negative_dir = Path(args.negative_dir).resolve()
    positives = list(positive_dir.glob("*.wav"))
    negatives = list(negative_dir.glob("*.wav"))

    if not positives or not negatives:
        log("Skipping verifier: personal and negative WAV clips are both required")
        return

    output = Path(args.export_dir).resolve() / f"{model_name}_verifier.pkl"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "train_verifier.py"),
        "--base-model",
        str(base_model),
        "--positive-dir",
        str(positive_dir),
        "--negative-dir",
        str(negative_dir),
        "--output",
        str(output),
    ]
    run(cmd)


def run_calibration(args: argparse.Namespace, onnx_path: Path, model_name: str, metadata_path: Path) -> None:
    positive_dirs = [
        Path(args.positive_dir).resolve(),
        Path(args.output_root).resolve() / model_name / model_name / "positive_test",
    ]
    negative_dirs = [
        Path(args.negative_dir).resolve(),
        Path(args.output_root).resolve() / model_name / model_name / "negative_test",
    ]
    if not any(path.exists() and any(path.glob("*.wav")) for path in negative_dirs):
        log("Skipping calibration: no negative WAV clips found")
        return

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "calibrate_model.py"),
        "--model",
        str(onnx_path),
        "--framework",
        "onnx",
        "--patience",
        str(args.calibration_patience),
        "--positive-limit",
        str(args.calibration_positive_limit),
        "--negative-limit",
        str(args.calibration_negative_limit),
        "--metadata-json",
        str(metadata_path),
    ]
    for directory in positive_dirs:
        if directory.exists():
            cmd.extend(["--positive-dir", str(directory)])
    for directory in negative_dirs:
        if directory.exists():
            cmd.extend(["--negative-dir", str(directory)])

    log("")
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=os.environ.copy())
    if result.returncode != 0:
        log(f"WARNING: calibration exited {result.returncode}; model artifact still exists at {onnx_path}")


def log_torch_devices() -> None:
    try:
        import torch
    except Exception as exc:
        log(f"WARNING: could not import torch for device check: {exc}")
        return

    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    selected = "cuda:0" if cuda_available else ("mps" if os.environ.get("OWW_ENABLE_MPS", "1") == "1" and mps_available else "cpu")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device availability: cuda={cuda_available}, mps={mps_available}, selected={selected}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the openWakeWord automatic training pipeline.")
    data_root = Path(os.environ.get("OWW_DATA_DIR", str(ROOT_DIR))).resolve()
    asset_dir = Path(os.environ.get("OWW_ASSET_DIR", os.environ.get("OWW_DATA_DIR", str(ROOT_DIR / "data")))).resolve()
    parser.add_argument("phrase", help="Wake phrase to train, for example 'hey tater'")
    parser.add_argument("--model-name")
    parser.add_argument("--samples", type=int, default=int(os.environ.get("OWW_DEFAULT_SAMPLES", "20000")))
    parser.add_argument("--validation-samples", type=int, default=int(os.environ.get("OWW_DEFAULT_VALIDATION_SAMPLES", "2000")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("OWW_DEFAULT_STEPS", "50000")))
    parser.add_argument("--tts-batch-size", type=int, default=int(os.environ.get("OWW_DEFAULT_TTS_BATCH", "50")))
    parser.add_argument(
        "--negative-tts-batch-divisor",
        type=int,
        default=int(os.environ.get("OWW_NEGATIVE_TTS_DIVISOR", "7")),
        help="Divide TTS batch size by this for adversarial negative clip generation; lower is faster but uses more memory.",
    )
    parser.add_argument("--augmentation-batch-size", type=int, default=int(os.environ.get("OWW_DEFAULT_AUG_BATCH", "16")))
    parser.add_argument("--augmentation-rounds", type=int, default=1)
    parser.add_argument("--positive-batch", type=int, default=50)
    parser.add_argument("--adversarial-batch", type=int, default=50)
    parser.add_argument(
        "--negative-batch",
        type=int,
        default=int(os.environ.get("OWW_DEFAULT_NEGATIVE_BATCH", "1024")),
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=int(os.environ.get("OWW_DEFAULT_VALIDATION_BATCH", "2048")),
        help="Number of false-positive validation windows evaluated at once.",
    )
    parser.add_argument("--model-type", choices=("dnn", "rnn"), default="dnn")
    parser.add_argument("--layer-size", type=int, default=32)
    parser.add_argument("--target-accuracy", type=float, default=0.7)
    parser.add_argument("--target-recall", type=float, default=0.5)
    parser.add_argument("--target-fp-per-hour", type=float, default=float(os.environ.get("OWW_DEFAULT_TARGET_FP", "0.2")))
    parser.add_argument("--max-negative-weight", type=int, default=int(os.environ.get("OWW_DEFAULT_MAX_NEGATIVE_WEIGHT", "1500")))
    parser.add_argument("--custom-negative-phrase", action="append", default=[])
    parser.add_argument("--data-dir", default=str(asset_dir))
    parser.add_argument("--output-root", default=os.environ.get("OWW_OUTPUT_ROOT", str(data_root / "output")))
    parser.add_argument(
        "--export-dir",
        default=os.environ.get("OWW_EXPORT_DIR", os.environ.get("OWW_TRAINED_DIR", str(data_root / "trained_wake_words"))),
    )
    parser.add_argument(
        "--openwakeword-dir",
        default=os.environ.get("OWW_OPENWAKEWORD_DIR", str(data_root / "vendor" / "openwakeword")),
    )
    parser.add_argument(
        "--piper-dir",
        default=os.environ.get("OWW_PIPER_DIR", str(data_root / "vendor" / "piper-sample-generator")),
    )
    parser.add_argument("--positive-dir", default=os.environ.get("OWW_PERSONAL_DIR", str(data_root / "personal_samples")))
    parser.add_argument("--negative-dir", default=os.environ.get("OWW_NEGATIVE_DIR", str(data_root / "negative_samples")))
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-positive-limit", type=int, default=int(os.environ.get("OWW_CALIBRATION_POSITIVE_LIMIT", "200")))
    parser.add_argument("--calibration-negative-limit", type=int, default=int(os.environ.get("OWW_CALIBRATION_NEGATIVE_LIMIT", "500")))
    parser.add_argument("--calibration-patience", type=int, default=int(os.environ.get("OWW_CALIBRATION_PATIENCE", "3")))
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=os.environ.get("OWW_FORCE_CPU", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument("--train-verifier", action="store_true")
    args = parser.parse_args()
    if args.negative_tts_batch_divisor < 1:
        raise SystemExit("--negative-tts-batch-divisor must be 1 or greater")
    if args.negative_batch < 1:
        raise SystemExit("--negative-batch must be 1 or greater")
    if args.validation_batch_size < 1:
        raise SystemExit("--validation-batch-size must be 1 or greater")

    model_name = safe_name(args.model_name or args.phrase)
    output_dir = Path(args.output_root).resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{model_name}.yaml"

    openwakeword_dir = Path(args.openwakeword_dir).resolve()
    train_py = openwakeword_dir / "openwakeword" / "train.py"
    if not train_py.exists():
        raise SystemExit(f"openWakeWord train.py not found at {train_py}. Run ./train_openwakeword.sh first.")

    config = make_config(args, model_name, output_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log(f"Wrote config: {config_path}")

    metadata = {
        "model_name": model_name,
        "phrase": args.phrase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config": config,
    }

    if args.config_only:
        sync_artifacts(output_dir, Path(args.export_dir).resolve(), model_name, metadata)
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # deep-phonemizer's checkpoint predates PyTorch 2.6's weights_only default.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
        env["OWW_ENABLE_MPS"] = "0"
        env["OWW_PIPER_DEVICE"] = "cpu"

    os.environ.update({key: value for key, value in env.items() if key in {"OWW_ENABLE_MPS", "CUDA_VISIBLE_DEVICES"}})
    log_torch_devices()
    log(f"Piper generator device policy: {env.get('OWW_PIPER_DEVICE', 'auto')}")
    negative_tts_batch = max(1, args.tts_batch_size // args.negative_tts_batch_divisor)
    log(
        "Piper negative TTS batch: "
        f"{negative_tts_batch} (tts_batch_size={args.tts_batch_size}, "
        f"divisor={args.negative_tts_batch_divisor})"
    )

    if not args.skip_generate:
        run([sys.executable, str(train_py), "--training_config", str(config_path), "--generate_clips"], env=env)
    if not args.skip_augment:
        normalize_generated_clip_sample_rates(output_dir, model_name)
        run([sys.executable, str(train_py), "--training_config", str(config_path), "--augment_clips"], env=env)
    if not args.skip_train:
        missing_features = [path for path in generated_feature_files(output_dir, model_name) if not path.exists()]
        if missing_features and args.skip_augment:
            log("Generated feature files are missing; running augmentation before training:")
            for path in missing_features:
                log(f"  {path}")
            normalize_generated_clip_sample_rates(output_dir, model_name)
            run([sys.executable, str(train_py), "--training_config", str(config_path), "--augment_clips"], env=env)

        expected_onnx = output_dir / f"{model_name}.onnx"
        run(
            [sys.executable, str(train_py), "--training_config", str(config_path), "--train_model"],
            env=env,
            allow_onnx_failure=expected_onnx,
        )

    expected_onnx = output_dir / f"{model_name}.onnx"
    if not expected_onnx.is_file():
        raise SystemExit(f"openWakeWord did not produce {expected_onnx}")

    export_dir = Path(args.export_dir).resolve()
    onnx_path = sync_artifacts(output_dir, export_dir, model_name, metadata)
    if onnx_path and not args.skip_calibration:
        run_calibration(args, onnx_path, model_name, export_dir / f"{model_name}.json")
    if args.train_verifier and onnx_path:
        train_verifier(args, onnx_path, model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
