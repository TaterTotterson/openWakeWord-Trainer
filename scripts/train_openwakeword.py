#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
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


def run_process(
    cmd: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    log("")
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    result = run_process(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def training_export_paths(expected_onnx: Path) -> list[Path]:
    """Return model files that could make a failed attempt look successful."""
    paths = [expected_onnx]
    paths.extend(sorted(expected_onnx.parent.glob("*.data")))
    return [path for path in paths if path.is_file()]


def quarantine_existing_training_exports(expected_onnx: Path) -> list[Path]:
    """Move prior exports aside so only this run can satisfy the model check."""
    existing = training_export_paths(expected_onnx)
    if not existing:
        return []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history_dir = (
        expected_onnx.parent
        / expected_onnx.stem
        / "quality_attempts"
        / "preexisting"
    )
    history_dir.mkdir(parents=True, exist_ok=True)
    quarantined: list[Path] = []
    for path in existing:
        destination = history_dir / f"{timestamp}-{path.name}"
        shutil.move(str(path), str(destination))
        quarantined.append(destination)
        log(f"Preserved pre-existing training artifact: {destination}")
    return quarantined


def discard_training_exports(expected_onnx: Path) -> None:
    """Remove incomplete exports before a clean retry or terminal failure."""
    for path in training_export_paths(expected_onnx):
        path.unlink(missing_ok=True)


def cpu_training_environment(env: dict[str, str]) -> dict[str, str]:
    cpu_env = dict(env)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    cpu_env["HIP_VISIBLE_DEVICES"] = ""
    cpu_env["ROCR_VISIBLE_DEVICES"] = ""
    cpu_env["OWW_ENABLE_MPS"] = "0"
    cpu_env["OWW_PIPER_DEVICE"] = "cpu"
    return cpu_env


def run_training_with_cpu_fallback(
    cmd: list[str],
    *,
    env: dict[str, str],
    expected_onnx: Path,
) -> None:
    """Run training and retry once on CPU after an Apple MPS failure."""
    discard_training_exports(expected_onnx)
    result = run_process(cmd, env=env)
    if result.returncode == 0 and expected_onnx.is_file():
        return

    mps_enabled = (
        sys.platform == "darwin"
        and env.get("OWW_ENABLE_MPS", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    if mps_enabled:
        if result.returncode == 0:
            reason = "training exited successfully without producing a new ONNX model"
        else:
            reason = f"training exited {result.returncode}"
        log(f"WARNING: Apple MPS {reason}; retrying this training pass once on CPU")
        discard_training_exports(expected_onnx)
        cpu_env = cpu_training_environment(env)
        cpu_result = run_process(cmd, env=cpu_env)
        if cpu_result.returncode == 0 and expected_onnx.is_file():
            # Keep later corrective passes on the known-good CPU path instead
            # of risking another Metal hang in the same training run.
            env.update(cpu_env)
            log("CPU retry completed and produced a fresh ONNX model")
            return
        discard_training_exports(expected_onnx)
        if cpu_result.returncode != 0:
            raise SystemExit(cpu_result.returncode)
        raise SystemExit("CPU retry completed without producing an ONNX model")

    discard_training_exports(expected_onnx)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    raise SystemExit("Training completed without producing an ONNX model")


def existing_wav_dir(path: Path) -> bool:
    return path.exists() and any(path.glob("*.wav"))


def _ordered_wavs(path: Path) -> list[Path]:
    """Return a stable, well-shuffled ordering without depending on RNG state."""
    return sorted(
        (item.resolve() for item in path.glob("*.wav") if item.is_file()),
        key=lambda item: hashlib.sha256(item.name.encode("utf-8")).hexdigest(),
    )


def partition_uploaded_audio(
    source_dir: Path,
    output_dir: Path,
    label: str,
    *,
    minimum_for_holdout: int = 5,
    holdout_fraction: float = 0.20,
) -> tuple[Path, Path | None]:
    """Create deterministic train/real-device-holdout views using symlinks."""
    paths = _ordered_wavs(source_dir)
    if len(paths) < minimum_for_holdout:
        return source_dir, None
    ordered = sorted(
        paths,
        key=lambda path: hashlib.sha256(
            f"{path.name}\0{path.stat().st_size}".encode("utf-8")
        ).digest(),
    )
    holdout_count = max(1, int(round(len(ordered) * holdout_fraction)))
    holdout = set(ordered[:holdout_count])
    asset_root = output_dir / "_trainer_assets"
    train_dir = asset_root / f"{label}_train"
    holdout_dir = asset_root / f"{label}_holdout"
    for directory, selected in (
        (train_dir, [path for path in paths if path not in holdout]),
        (holdout_dir, [path for path in paths if path in holdout]),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        expected = {path.name for path in selected}
        for stale in directory.glob("*.wav"):
            if stale.name not in expected:
                stale.unlink()
        for path in selected:
            destination = directory / path.name
            if destination.is_symlink() and destination.resolve() == path.resolve():
                continue
            destination.unlink(missing_ok=True)
            destination.symlink_to(path.resolve())
    log(
        f"Reserved {holdout_count}/{len(paths)} uploaded {label.replace('_', ' ')} "
        "clip(s) for real-device validation"
    )
    return train_dir, holdout_dir


def stage_shared_positive_clips(
    output_dir: Path,
    model_name: str,
    source_dir: Path,
    train_count: int,
    validation_count: int,
    personal_dir: Path | None = None,
) -> None:
    """Reuse the trainer's canonical TTS corpus instead of regenerating it with Piper."""
    source_wavs = _ordered_wavs(source_dir)
    required = train_count + validation_count
    if len(source_wavs) < required:
        raise SystemExit(
            f"Shared positive TTS corpus needs {required} WAV files but only "
            f"{len(source_wavs)} were found in {source_dir}"
        )

    clip_root = output_dir / model_name
    train_dir = clip_root / "positive_train"
    validation_dir = clip_root / "positive_test"
    personal_wavs = _ordered_wavs(personal_dir) if personal_dir and personal_dir.exists() else []
    selected = {
        train_dir: source_wavs[:train_count],
        validation_dir: source_wavs[train_count:required],
    }
    source_digest = hashlib.sha256()
    for path in source_wavs[:required]:
        stat = path.stat()
        source_digest.update(
            f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    for path in personal_wavs:
        stat = path.stat()
        source_digest.update(
            f"personal\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    fingerprint = {
        "source": str(source_dir.resolve()),
        "train_count": train_count,
        "validation_count": validation_count,
        "personal_count": len(personal_wavs),
        "source_digest": source_digest.hexdigest(),
    }
    marker = clip_root / ".tater-shared-positive.json"
    try:
        cached = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if cached == fingerprint and all(
        directory.exists()
        and len(list(directory.glob("*.wav")))
        == len(paths) + (len(personal_wavs) if directory == train_dir else 0)
        for directory, paths in selected.items()
    ):
        log(f"Reusing {required} shared positive TTS clips")
        return

    for directory, paths in selected.items():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(paths):
            (directory / f"tater_shared_{index:06d}.wav").symlink_to(source)
    for index, source in enumerate(personal_wavs):
        (train_dir / f"tater_personal_{index:06d}.wav").symlink_to(source)

    for feature_path in generated_feature_files(output_dir, model_name):
        if feature_path.name.startswith("positive_") and feature_path.exists():
            feature_path.unlink()
    marker.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    log(
        f"Staged {train_count} shared and {len(personal_wavs)} personal training positives, "
        f"plus {validation_count} shared validation positives"
    )


def stage_reviewed_negative_clips(
    output_dir: Path,
    model_name: str,
    source_dir: Path,
) -> tuple[Path, int]:
    """Stage reviewed false wakes as an independently weighted source."""
    clip_root = output_dir / model_name
    negative_dir = clip_root / "reviewed_negative_train"
    negative_dir.mkdir(parents=True, exist_ok=True)
    reviewed = _ordered_wavs(source_dir)
    digest = hashlib.sha256()
    for path in reviewed:
        stat = path.stat()
        digest.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    marker = clip_root / ".tater-reviewed-negatives.json"
    fingerprint = {
        "source": str(source_dir.resolve()),
        "count": len(reviewed),
        "source_digest": digest.hexdigest(),
    }
    try:
        cached = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    # Remove links created by older trainers, where reviewed clips were diluted
    # into the large automatic adversarial feature array.
    legacy_dir = clip_root / "negative_train"
    legacy_managed = list(legacy_dir.glob("tater_reviewed_*.wav"))
    if legacy_managed:
        for path in legacy_managed:
            path.unlink()
        (clip_root / "negative_features_train.npy").unlink(missing_ok=True)

    managed = list(negative_dir.glob("tater_reviewed_*.wav"))
    if cached == fingerprint and len(managed) == len(reviewed):
        if reviewed:
            log(f"Reusing {len(reviewed)} reviewed negative clips")
        return negative_dir, len(reviewed)

    for path in managed:
        path.unlink()
    for index, source in enumerate(reviewed):
        (negative_dir / f"tater_reviewed_{index:06d}.wav").symlink_to(source)
    feature = clip_root / "reviewed_negative_features_train.npy"
    if feature.exists():
        feature.unlink()
    marker.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    if reviewed:
        log(f"Staged {len(reviewed)} reviewed negative clips as a weighted OWW source")
    return negative_dir, len(reviewed)


def _generated_count(total: int, fraction: float) -> int:
    return max(0, int(round(max(0, total) * max(0.0, min(1.0, fraction)))))


def generate_custom_negative_clips(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Generate custom confusing phrases separately so their weight is explicit."""
    clip_root = output_dir / model_name
    train_dir = clip_root / "custom_negative_train"
    test_dir = clip_root / "custom_negative_test"
    phrases = list(config.get("weighted_custom_negative_phrases") or [])
    if not phrases:
        return train_dir, test_dir

    piper_dir = str(Path(args.piper_dir).resolve())
    sys.path.insert(0, piper_dir)
    try:
        from generate_samples import generate_samples
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(piper_dir)

    desired = (
        (train_dir, _generated_count(args.samples, args.custom_negative_fraction), [0.98], [0.98]),
        (test_dir, _generated_count(args.validation_samples, args.custom_negative_fraction), [1.0], [1.0]),
    )
    batch_size = max(1, args.tts_batch_size // args.negative_tts_batch_divisor)
    for directory, target_count, noise_scales, noise_scale_ws in desired:
        directory.mkdir(parents=True, exist_ok=True)
        current = len(list(directory.glob("*.wav")))
        needed = max(0, target_count - current)
        if needed:
            log(
                f"Generating {needed} weighted custom-negative clip(s) in {directory.name} "
                f"from {len(phrases)} phrase(s)"
            )
            generate_samples(
                text=phrases,
                max_samples=needed,
                batch_size=batch_size,
                noise_scales=noise_scales,
                noise_scale_ws=noise_scale_ws,
                length_scales=[0.75, 1.0, 1.25],
                output_dir=str(directory),
                auto_reduce_batch_size=True,
            )
        else:
            log(f"Reusing {current} weighted custom-negative clip(s) from {directory.name}")
    return train_dir, test_dir


def determine_total_length(positive_dir: Path) -> int:
    durations: list[int] = []
    for path in _ordered_wavs(positive_dir):
        try:
            with wave.open(str(path), "rb") as handle:
                rate = int(handle.getframerate() or 0)
                frames = int(handle.getnframes() or 0)
            if rate > 0 and frames > 0:
                durations.append(int(round((frames * 16000 / rate) / 1000) * 1000))
        except Exception:
            continue
    if not durations:
        return 32000
    durations.sort()
    middle = len(durations) // 2
    median_samples = (
        durations[middle]
        if len(durations) % 2
        else int(round((durations[middle - 1] + durations[middle]) / 2))
    )
    total_length = median_samples + 12000
    if total_length < 32000 or abs(total_length - 32000) <= 4000:
        return 32000
    return total_length


def _audio_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for raw in paths:
        directory = Path(raw)
        if directory.exists():
            files.extend(str(path) for path in sorted(directory.rglob("*.wav")))
    return files


def build_weighted_feature_source(
    config: dict[str, Any],
    source_dir: Path,
    feature_path: Path,
    *,
    augmentation_rounds: int = 1,
) -> int:
    """Create one independently sampled OWW feature source with cache validation."""
    clips = _ordered_wavs(source_dir)
    if not clips:
        feature_path.unlink(missing_ok=True)
        feature_path.with_suffix(feature_path.suffix + ".tater.json").unlink(missing_ok=True)
        return 0

    total_length = int(
        config.get("total_length")
        or determine_total_length(
            Path(config["output_dir"]) / str(config["model_name"]) / "positive_test"
        )
    )
    config["total_length"] = total_length
    rounds = max(1, int(augmentation_rounds))
    digest = hashlib.sha256()
    for path in clips:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    fingerprint = {
        "source": str(source_dir.resolve()),
        "source_digest": digest.hexdigest(),
        "clip_count": len(clips),
        "augmentation_rounds": rounds,
        "total_length": total_length,
        "feature_device": os.environ.get("OWW_FEATURE_DEVICE", "cpu"),
        "background_paths": list(config.get("background_paths") or []),
        "rir_paths": list(config.get("rir_paths") or []),
    }
    marker = feature_path.with_suffix(feature_path.suffix + ".tater.json")
    try:
        cached = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if cached == fingerprint and feature_path.exists():
        log(f"Reusing weighted feature source: {feature_path.name}")
        return len(clips) * rounds

    feature_path.unlink(missing_ok=True)
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator
    import torch

    augmented = augment_clips(
        [str(path) for path in clips] * rounds,
        total_length=total_length,
        batch_size=max(1, int(config.get("augmentation_batch_size", 16))),
        background_clip_paths=_audio_files(list(config.get("background_paths") or [])),
        RIR_paths=_audio_files(list(config.get("rir_paths") or [])),
    )
    cpu_count = os.cpu_count() or 1
    compute_features_from_generator(
        augmented,
        n_total=len(clips) * rounds,
        clip_duration=total_length,
        output_file=str(feature_path),
        device=os.environ.get("OWW_FEATURE_DEVICE", "cpu"),
        ncpu=1 if torch.cuda.is_available() else max(1, cpu_count // 2),
    )
    marker.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    log(f"Built weighted feature source: {feature_path.name} ({len(clips) * rounds} samples)")
    return len(clips) * rounds


def stage_corrective_negative_clips(
    paths: list[str],
    destination: Path,
    *,
    excluded_roots: list[Path],
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    for raw in paths:
        source = Path(raw).expanduser().resolve()
        if not source.is_file():
            continue
        if any(source.is_relative_to(root.resolve()) for root in excluded_roots if root):
            continue
        selected.append(source)
    expected: set[str] = set()
    for index, source in enumerate(sorted(set(selected), key=lambda path: str(path))):
        name = f"{index:04d}_{safe_name(source.stem)}.wav"
        expected.add(name)
        target = destination / name
        if target.is_symlink() and target.resolve() == source:
            continue
        target.unlink(missing_ok=True)
        target.symlink_to(source)
    for stale in destination.glob("*.wav"):
        if stale.name not in expected:
            stale.unlink()
    return _ordered_wavs(destination)


def merge_corrective_negative_paths(
    previous_paths: list[str],
    calibration: dict[str, Any],
) -> list[str]:
    """Accumulate hard negatives across retries while preserving stable order."""
    current_paths = list(
        (calibration.get("misclassified") or {}).get("triggered_negative_paths") or []
    )
    merged: list[str] = []
    seen: set[Path] = set()
    for raw in previous_paths + current_paths:
        path = Path(raw).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            merged.append(str(path))
    return merged


def make_config(args: argparse.Namespace, model_name: str, output_dir: Path) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    piper_dir = Path(args.piper_dir).resolve()
    feature_bank_dir = Path(args.feature_bank_dir).resolve()
    features = feature_bank_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation = data_dir / "validation_set_features.npy"

    background_paths = []
    background_candidates = [Path(value).resolve() for value in args.background_dir]
    if not background_candidates:
        background_candidates = [data_dir / "audioset_16k", data_dir / "fma", data_dir / "background_clips"]
    for candidate in background_candidates:
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

    target_key = " ".join(args.phrase.casefold().split())
    custom_negative_phrases = []
    seen_negative_phrases: set[str] = set()
    for value in args.custom_negative_phrase:
        phrase = " ".join(str(value).split())
        key = phrase.casefold()
        if phrase and key != target_key and key not in seen_negative_phrases:
            seen_negative_phrases.add(key)
            custom_negative_phrases.append(phrase)

    feature_save_dir = output_dir / model_name
    if custom_negative_phrases:
        feature_data_files["custom_negative"] = str(
            feature_save_dir / "custom_negative_features_train.npy"
        )
        batch_n_per_class["custom_negative"] = args.custom_negative_batch

    return {
        "model_name": model_name,
        "target_phrase": [args.phrase],
        # Keep custom phrases out of upstream's combined adversarial pool. They
        # are generated and sampled separately below with a guaranteed batch share.
        "custom_negative_phrases": [],
        "weighted_custom_negative_phrases": custom_negative_phrases,
        "custom_negative_fraction": args.custom_negative_fraction,
        "n_samples": args.samples,
        "n_samples_val": args.validation_samples,
        "tts_batch_size": args.tts_batch_size,
        "negative_tts_batch_divisor": args.negative_tts_batch_divisor,
        "augmentation_batch_size": args.augmentation_batch_size,
        "piper_sample_generator_path": str(piper_dir),
        "output_dir": str(output_dir),
        "rir_paths": [
            str(path)
            for path in ([Path(value).resolve() for value in args.rir_dir] or [data_dir / "mit_rirs"])
            if existing_wav_dir(path)
        ],
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
        clip_dir / "custom_negative_train",
        clip_dir / "custom_negative_test",
        clip_dir / "reviewed_negative_train",
        clip_dir / "quality_retry_negative",
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


def run_calibration(
    args: argparse.Namespace,
    onnx_path: Path,
    model_name: str,
    metadata_path: Path,
    *,
    sample_partition: int = 0,
) -> dict[str, Any] | None:
    clip_root = Path(args.output_root).resolve() / model_name / model_name
    positive_dirs = [clip_root / "positive_test"]
    negative_dirs = [clip_root / "negative_test", clip_root / "custom_negative_test"]
    priority_positive_dirs = [
        Path(value).resolve()
        for value in [getattr(args, "positive_holdout_dir", None) or args.positive_dir]
        if value
    ]
    priority_negative_dirs = [
        Path(value).resolve()
        for value in [getattr(args, "negative_holdout_dir", None) or args.negative_dir]
        if value
    ]
    all_negative_dirs = negative_dirs + priority_negative_dirs
    if not any(path.exists() and any(path.glob("*.wav")) for path in all_negative_dirs):
        log("Skipping calibration: no negative WAV clips found")
        return None

    report_path = clip_root / f"{model_name}.calibration.json"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "calibrate_model.py"),
        "--model",
        str(onnx_path),
        "--framework",
        "onnx",
        "--patience",
        str(args.calibration_patience),
        "--max-patience",
        str(args.calibration_max_patience),
        "--sample-partition",
        str(max(0, sample_partition)),
        "--positive-limit",
        str(args.calibration_positive_limit),
        "--negative-limit",
        str(args.calibration_negative_limit),
        "--metadata-json",
        str(metadata_path),
        "--output",
        str(report_path),
        "--min-positive-recall",
        str(args.calibration_min_positive_recall),
        "--max-generated-false-positive-rate",
        str(args.calibration_max_generated_false_positive_rate),
    ]
    for directory in positive_dirs:
        if directory.exists():
            cmd.extend(["--positive-dir", str(directory)])
    for directory in negative_dirs:
        if directory.exists():
            cmd.extend(["--negative-dir", str(directory)])
    for directory in priority_positive_dirs:
        if directory.exists():
            cmd.extend(["--priority-positive-dir", str(directory)])
    for directory in priority_negative_dirs:
        if directory.exists():
            cmd.extend(["--priority-negative-dir", str(directory)])

    log("")
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=os.environ.copy())
    if result.returncode != 0:
        log(f"WARNING: calibration exited {result.returncode}; model artifact still exists at {onnx_path}")
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"WARNING: could not read calibration report {report_path}: {exc}")
        return None


def build_corrective_retry(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
    model_name: str,
    calibration: dict[str, Any],
    *,
    previous_triggered_paths: list[str] | None = None,
    retry_number: int = 1,
) -> dict[str, Any] | None:
    paths = merge_corrective_negative_paths(
        previous_triggered_paths or [],
        calibration,
    )
    holdout_roots = [
        Path(value).resolve()
        for value in [getattr(args, "negative_holdout_dir", None)]
        if value
    ]
    clip_root = output_dir / model_name
    retry_dir = clip_root / "quality_retry_negative"
    staged = stage_corrective_negative_clips(
        paths,
        retry_dir,
        excluded_roots=holdout_roots,
    )
    if not staged:
        return None

    retry = copy.deepcopy(config)
    feature_path = clip_root / "quality_retry_negative_features.npy"
    samples = build_weighted_feature_source(
        retry,
        retry_dir,
        feature_path,
        augmentation_rounds=max(1, args.reviewed_negative_augmentation_rounds),
    )
    if samples <= 0:
        return None

    retry["feature_data_files"]["quality_retry_negative"] = str(feature_path)
    retry["batch_n_per_class"]["quality_retry_negative"] = min(
        max(
            1,
            int(
                round(
                    args.corrective_negative_batch
                    * (
                        1.0
                        + max(0, retry_number - 1)
                        * max(0.0, args.corrective_batch_multiplier - 1.0)
                    )
                )
            ),
        ),
        samples,
    )
    batch_multiplier = 1.0 + max(1, retry_number) * max(
        0.0, args.corrective_batch_multiplier - 1.0
    )
    for key in ("adversarial_negative", "custom_negative"):
        if key in retry["batch_n_per_class"]:
            retry["batch_n_per_class"][key] = max(
                1,
                int(round(retry["batch_n_per_class"][key] * batch_multiplier)),
            )
    weight_multiplier = 1.0 + max(1, retry_number) * max(
        0.0, args.corrective_weight_multiplier - 1.0
    )
    retry["max_negative_weight"] = max(
        3000,
        int(round(float(retry.get("max_negative_weight") or 1500) * weight_multiplier)),
    )
    retry["quality_retry"] = {
        "triggered_negative_paths": [str(path) for path in staged],
        "source_triggered_negative_paths": paths,
        "source_sample_partition": int(calibration.get("sample_partition") or 0),
        "retry_number": max(1, retry_number),
    }
    return retry


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


def log_feature_provider() -> None:
    import onnxruntime as ort

    requested = os.environ.get("OWW_FEATURE_DEVICE", "cpu").strip().lower()
    providers = ort.get_available_providers()
    log(f"ONNX feature provider policy: {requested}; available={providers}")
    if requested == "gpu" and "CUDAExecutionProvider" not in providers:
        raise SystemExit(
            "OWW_FEATURE_DEVICE requested GPU, but ONNX Runtime has no CUDAExecutionProvider"
        )


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
    parser.add_argument(
        "--adversarial-batch",
        type=int,
        default=int(os.environ.get("OWW_DEFAULT_ADVERSARIAL_BATCH", "70")),
        help="Automatic phonetic hard negatives included in every training batch.",
    )
    parser.add_argument(
        "--custom-negative-batch",
        type=int,
        default=int(os.environ.get("OWW_DEFAULT_CUSTOM_NEGATIVE_BATCH", "30")),
        help="Custom confusing-phrase examples included in every training batch.",
    )
    parser.add_argument(
        "--custom-negative-fraction",
        type=float,
        default=float(os.environ.get("OWW_CUSTOM_NEGATIVE_FRACTION", "0.30")),
        help="Custom-negative clip count relative to the automatic corpus (default: 0.30).",
    )
    parser.add_argument(
        "--reviewed-negative-batch",
        type=int,
        default=int(os.environ.get("OWW_DEFAULT_REVIEWED_NEGATIVE_BATCH", "25")),
    )
    parser.add_argument(
        "--reviewed-negative-augmentation-rounds",
        type=int,
        default=int(os.environ.get("OWW_REVIEWED_NEGATIVE_AUGMENTATION_ROUNDS", "10")),
    )
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
    parser.add_argument(
        "--layer-size",
        type=int,
        default=int(os.environ.get("OWW_DEFAULT_LAYER_SIZE", "64")),
    )
    parser.add_argument("--target-accuracy", type=float, default=0.7)
    parser.add_argument("--target-recall", type=float, default=0.5)
    parser.add_argument("--target-fp-per-hour", type=float, default=float(os.environ.get("OWW_DEFAULT_TARGET_FP", "0.2")))
    parser.add_argument("--max-negative-weight", type=int, default=int(os.environ.get("OWW_DEFAULT_MAX_NEGATIVE_WEIGHT", "1500")))
    parser.add_argument("--custom-negative-phrase", action="append", default=[])
    parser.add_argument(
        "--positive-tts-dir",
        default=os.environ.get("OWW_POSITIVE_TTS_DIR", ""),
        help="Reuse a shared generated positive-TTS WAV corpus instead of generating OWW positives.",
    )
    parser.add_argument(
        "--background-dir",
        action="append",
        default=[],
        help="Raw background-audio directory; repeat to combine multiple shared corpora.",
    )
    parser.add_argument(
        "--rir-dir",
        action="append",
        default=[],
        help="Room-impulse-response directory; repeat to combine multiple shared corpora.",
    )
    parser.add_argument("--data-dir", default=str(asset_dir))
    parser.add_argument(
        "--feature-bank-dir",
        default=os.environ.get("OWW_FEATURE_BANK_DIR", str(asset_dir)),
        help="Persistent directory containing the broad negative feature bank.",
    )
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
    parser.add_argument("--calibration-patience", type=int, default=int(os.environ.get("OWW_CALIBRATION_PATIENCE", "2")))
    parser.add_argument("--calibration-max-patience", type=int, default=int(os.environ.get("OWW_CALIBRATION_MAX_PATIENCE", "5")))
    parser.add_argument("--calibration-min-positive-recall", type=float, default=float(os.environ.get("OWW_CALIBRATION_MIN_RECALL", "0.90")))
    parser.add_argument(
        "--calibration-max-generated-false-positive-rate",
        type=float,
        default=float(os.environ.get("OWW_CALIBRATION_MAX_GENERATED_FP_RATE", "0.01")),
    )
    parser.add_argument("--calibration-retries", type=int, default=int(os.environ.get("OWW_CALIBRATION_RETRIES", "2")))
    parser.add_argument("--corrective-negative-batch", type=int, default=int(os.environ.get("OWW_CORRECTIVE_NEGATIVE_BATCH", "50")))
    parser.add_argument("--corrective-batch-multiplier", type=float, default=float(os.environ.get("OWW_CORRECTIVE_BATCH_MULTIPLIER", "1.5")))
    parser.add_argument("--corrective-weight-multiplier", type=float, default=float(os.environ.get("OWW_CORRECTIVE_WEIGHT_MULTIPLIER", "2.0")))
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
    if args.adversarial_batch < 1 or args.custom_negative_batch < 1:
        raise SystemExit("hard-negative batch sizes must be 1 or greater")
    if not 0.0 <= args.custom_negative_fraction <= 1.0:
        raise SystemExit("--custom-negative-fraction must be between 0 and 1")
    if args.calibration_max_patience < args.calibration_patience:
        raise SystemExit("--calibration-max-patience cannot be lower than --calibration-patience")
    if not 0.0 <= args.calibration_max_generated_false_positive_rate <= 1.0:
        raise SystemExit("--calibration-max-generated-false-positive-rate must be between 0 and 1")

    model_name = safe_name(args.model_name or args.phrase)
    output_dir = Path(args.output_root).resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{model_name}.yaml"
    clip_root = output_dir / model_name

    positive_training_dir, positive_holdout_dir = partition_uploaded_audio(
        Path(args.positive_dir).resolve(),
        clip_root,
        "personal_positive",
    )
    negative_training_dir, negative_holdout_dir = partition_uploaded_audio(
        Path(args.negative_dir).resolve(),
        clip_root,
        "reviewed_negative",
    )
    args.positive_holdout_dir = str(positive_holdout_dir) if positive_holdout_dir else ""
    args.negative_holdout_dir = str(negative_holdout_dir) if negative_holdout_dir else ""

    openwakeword_dir = Path(args.openwakeword_dir).resolve()
    train_py = openwakeword_dir / "openwakeword" / "train.py"
    if not train_py.exists():
        raise SystemExit(f"openWakeWord train.py not found at {train_py}. Run ./train_openwakeword.sh first.")

    config = make_config(args, model_name, output_dir)
    if args.positive_tts_dir:
        stage_shared_positive_clips(
            output_dir,
            model_name,
            Path(args.positive_tts_dir).resolve(),
            args.samples,
            args.validation_samples,
            positive_training_dir,
        )
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
    log_feature_provider()
    log(f"Piper generator device policy: {env.get('OWW_PIPER_DEVICE', 'auto')}")
    negative_tts_batch = max(1, args.tts_batch_size // args.negative_tts_batch_divisor)
    log(
        "Piper negative TTS batch: "
        f"{negative_tts_batch} (tts_batch_size={args.tts_batch_size}, "
        f"divisor={args.negative_tts_batch_divisor})"
    )

    if not args.skip_generate:
        run([sys.executable, str(train_py), "--training_config", str(config_path), "--generate_clips"], env=env)
        generate_custom_negative_clips(args, config, output_dir, model_name)
    reviewed_dir, reviewed_count = stage_reviewed_negative_clips(
        output_dir,
        model_name,
        negative_training_dir,
    )
    if not args.skip_augment:
        normalize_generated_clip_sample_rates(output_dir, model_name)
        run([sys.executable, str(train_py), "--training_config", str(config_path), "--augment_clips"], env=env)

    custom_train_dir = clip_root / "custom_negative_train"
    custom_feature = clip_root / "custom_negative_features_train.npy"
    if config.get("weighted_custom_negative_phrases"):
        custom_samples = build_weighted_feature_source(
            config,
            custom_train_dir,
            custom_feature,
        )
        if custom_samples <= 0:
            raise SystemExit("Custom negative phrases were configured but produced no training clips")
        config["feature_data_files"]["custom_negative"] = str(custom_feature)

    if reviewed_count:
        reviewed_feature = clip_root / "reviewed_negative_features_train.npy"
        reviewed_samples = build_weighted_feature_source(
            config,
            reviewed_dir,
            reviewed_feature,
            augmentation_rounds=args.reviewed_negative_augmentation_rounds,
        )
        config["feature_data_files"]["reviewed_negative"] = str(reviewed_feature)
        config["batch_n_per_class"]["reviewed_negative"] = min(
            args.reviewed_negative_batch,
            reviewed_samples,
        )

    # Persist paths and weights for the distinct hard-negative sources before training.
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if not args.skip_train:
        missing_features = [path for path in generated_feature_files(output_dir, model_name) if not path.exists()]
        if missing_features and args.skip_augment:
            log("Generated feature files are missing; running augmentation before training:")
            for path in missing_features:
                log(f"  {path}")
            normalize_generated_clip_sample_rates(output_dir, model_name)
            run([sys.executable, str(train_py), "--training_config", str(config_path), "--augment_clips"], env=env)

        expected_onnx = output_dir / f"{model_name}.onnx"
        quarantine_existing_training_exports(expected_onnx)
        run_training_with_cpu_fallback(
            [sys.executable, str(train_py), "--training_config", str(config_path), "--train_model"],
            env=env,
            expected_onnx=expected_onnx,
        )

    expected_onnx = output_dir / f"{model_name}.onnx"
    if not expected_onnx.is_file():
        raise SystemExit(f"openWakeWord did not produce {expected_onnx}")

    export_dir = Path(args.export_dir).resolve()
    onnx_path = sync_artifacts(output_dir, export_dir, model_name, metadata)
    calibration: dict[str, Any] | None = None
    if onnx_path and not args.skip_calibration:
        calibration = run_calibration(
            args,
            onnx_path,
            model_name,
            export_dir / f"{model_name}.json",
            sample_partition=0,
        )

    retries_used = 0
    cumulative_triggered_paths: list[str] = []
    while (
        onnx_path
        and calibration is not None
        and not bool(calibration.get("deployment_ready"))
        and retries_used < max(0, args.calibration_retries)
    ):
        retry_config = build_corrective_retry(
            args,
            config,
            output_dir,
            model_name,
            calibration,
            previous_triggered_paths=cumulative_triggered_paths,
            retry_number=retries_used + 1,
        )
        if retry_config is None:
            log("Calibration failed, but no eligible generated false triggers were available for retry")
            break

        retries_used += 1
        cumulative_triggered_paths = list(
            (retry_config.get("quality_retry") or {}).get(
                "source_triggered_negative_paths"
            )
            or []
        )
        history_dir = clip_root / "quality_attempts"
        history_dir.mkdir(parents=True, exist_ok=True)
        source_model = output_dir / f"{model_name}.onnx"
        shutil.move(
            str(source_model),
            str(history_dir / f"{model_name}.attempt-{retries_used}.onnx"),
        )
        calibration_path = clip_root / f"{model_name}.calibration.json"
        if calibration_path.exists():
            shutil.copy2(
                calibration_path,
                history_dir / f"{model_name}.calibration.attempt-{retries_used}.json",
            )

        retry_path = output_dir / f"{model_name}.quality-retry-{retries_used}.yaml"
        retry_path.write_text(yaml.safe_dump(retry_config, sort_keys=False), encoding="utf-8")
        log(
            f"Calibration retry {retries_used}/{args.calibration_retries}: "
            f"negative weight={retry_config['max_negative_weight']}, "
            f"hard-negative batches={retry_config['batch_n_per_class']}"
        )
        run_training_with_cpu_fallback(
            [sys.executable, str(train_py), "--training_config", str(retry_path), "--train_model"],
            env=env,
            expected_onnx=source_model,
        )
        if not source_model.exists():
            raise SystemExit("openWakeWord corrective retry did not produce an ONNX model")

        metadata["quality_retries_used"] = retries_used
        metadata["retry_config"] = retry_config
        onnx_path = sync_artifacts(output_dir, export_dir, model_name, metadata)
        calibration = run_calibration(
            args,
            onnx_path,
            model_name,
            export_dir / f"{model_name}.json",
            sample_partition=retries_used,
        )

    metadata["quality_retries_used"] = retries_used
    if args.train_verifier and onnx_path:
        train_verifier(args, onnx_path, model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
