#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import wave
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


def collect_wavs(paths: list[str], *, limit: int, seed: int) -> list[Path]:
    wavs: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        candidates = [path] if path.is_file() else sorted(path.rglob("*.wav")) if path.exists() else []
        for candidate in candidates:
            if candidate.suffix.lower() != ".wav":
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                wavs.append(resolved)
    wavs.sort(key=lambda item: str(item))
    if limit > 0 and len(wavs) > limit:
        rng = random.Random(seed)
        wavs = sorted(rng.sample(wavs, limit), key=lambda item: str(item))
    return wavs


def wav_duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = float(wav.getframerate() or 0)
            frames = float(wav.getnframes() or 0)
            return frames / rate if rate > 0 else 0.0
    except Exception:
        return 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float((ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight))


def max_consecutive(values: list[float], threshold: float) -> int:
    best = 0
    current = 0
    for value in values:
        if float(value) >= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def threshold_values(minimum: float, maximum: float, step: float) -> list[float]:
    values: list[float] = []
    current = minimum
    while current <= maximum + 1e-9:
        values.append(round(current, 4))
        current += step
    if not values or values[-1] < maximum:
        values.append(round(maximum, 4))
    return values


def score_values(raw_values: Any) -> list[float]:
    if hasattr(raw_values, "tolist"):
        raw_values = raw_values.tolist()
    if isinstance(raw_values, dict):
        values: list[float] = []
        for nested_value in raw_values.values():
            values.extend(score_values(nested_value))
        return values
    if isinstance(raw_values, (list, tuple)):
        values = []
        for item in raw_values:
            values.extend(score_values(item))
        return values
    try:
        return [float(raw_values)]
    except Exception:
        return []


def prediction_score_series(result: Any) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}

    def add(label: Any, value: Any) -> None:
        label_text = str(label or "").strip()
        if not label_text:
            return
        values = score_values(value)
        if values:
            series.setdefault(label_text, []).extend(values)

    if isinstance(result, dict):
        for raw_label, raw_values in result.items():
            add(raw_label, raw_values)
    elif isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, dict):
                for raw_label, raw_values in item.items():
                    add(raw_label, raw_values)
            else:
                add("score", item)
    return series


def score_wavs(model: Any, wavs: list[Path], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ndx, wav_path in enumerate(wavs, start=1):
        if ndx == 1 or ndx % 50 == 0 or ndx == len(wavs):
            print(f"Scoring {label} clips: {ndx}/{len(wavs)}", flush=True)
        reset_model(model)
        result = model.predict_clip(str(wav_path))
        reset_model(model)
        best_label = ""
        best_frames: list[float] = []
        best_score = 0.0
        for raw_label, values in prediction_score_series(result).items():
            max_score = max(values) if values else 0.0
            if max_score >= best_score:
                best_label = str(raw_label)
                best_frames = values
                best_score = float(max_score)
        rows.append(
            {
                "path": str(wav_path),
                "label": best_label,
                "max_score": best_score,
                "duration_s": wav_duration_s(wav_path),
                "frames": best_frames,
            }
        )
    return rows


def reset_model(model: Any) -> None:
    try:
        model.reset()
    except Exception:
        pass


def activation_count(rows: list[dict[str, Any]], threshold: float, patience: int) -> int:
    return sum(1 for row in rows if max_consecutive(row.get("frames") or [], threshold) >= patience)


def build_metrics(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    thresholds: list[float],
    patience: int,
) -> list[dict[str, Any]]:
    negative_hours = sum(float(row.get("duration_s") or 0.0) for row in negatives) / 3600.0
    metrics: list[dict[str, Any]] = []
    for threshold in thresholds:
        false_clips = activation_count(negatives, threshold, patience)
        true_clips = activation_count(positives, threshold, patience)
        metrics.append(
            {
                "threshold": threshold,
                "patience": patience,
                "false_positive_clips": false_clips,
                "false_positive_rate": false_clips / len(negatives) if negatives else 0.0,
                "false_positives_per_hour": false_clips / negative_hours if negative_hours > 0 else None,
                "positive_clips_detected": true_clips,
                "positive_recall": true_clips / len(positives) if positives else None,
            }
        )
    return metrics


def choose_threshold(
    metrics: list[dict[str, Any]],
    *,
    max_false_positive_rate: float,
    min_positive_recall: float,
    fallback_threshold: float,
) -> tuple[float, str]:
    candidates = [
        row
        for row in metrics
        if float(row.get("false_positive_rate") or 0.0) <= max_false_positive_rate
        and (row.get("positive_recall") is None or float(row.get("positive_recall") or 0.0) >= min_positive_recall)
    ]
    if candidates:
        return float(candidates[0]["threshold"]), "meets_false_positive_and_recall_targets"

    safe_candidates = [
        row for row in metrics if float(row.get("false_positive_rate") or 0.0) <= max_false_positive_rate
    ]
    if safe_candidates:
        return float(safe_candidates[0]["threshold"]), "meets_false_positive_target_but_recall_is_low"

    return float(fallback_threshold), "no_threshold_met_false_positive_target"


def choose_confirmation_threshold(
    metrics: list[dict[str, Any]],
    *,
    min_positive_recall: float,
    fallback_threshold: float,
) -> tuple[float, str]:
    """Choose OWW's less-conservative threshold after an MWW candidate.

    A standalone detector must constrain false wakes itself. A confirmation
    detector has already been gated by MWW, so reusing the standalone policy
    can needlessly reject genuine speech. Choose the highest threshold that
    still preserves the requested positive recall.
    """
    candidates = [
        row
        for row in metrics
        if row.get("positive_recall") is not None
        and float(row.get("positive_recall") or 0.0) >= min_positive_recall
    ]
    if candidates:
        selected = max(candidates, key=lambda row: float(row.get("threshold") or 0.0))
        return float(selected["threshold"]), "meets_dual_confirmation_recall_target"
    return float(fallback_threshold), "no_threshold_met_dual_confirmation_recall_target"


def trim_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                "path": row.get("path"),
                "label": row.get("label"),
                "max_score": row.get("max_score"),
                "duration_s": row.get("duration_s"),
            }
        )
    compact.sort(key=lambda item: float(item.get("max_score") or 0.0), reverse=True)
    return compact[:20]


def update_metadata(path: Path, calibration: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        except Exception:
            payload = {}
    payload["calibration"] = calibration
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate an openWakeWord ONNX model against positive and negative WAV clips.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--framework", choices=("onnx", "tflite"), default="onnx")
    parser.add_argument("--positive-dir", action="append", default=[])
    parser.add_argument("--negative-dir", action="append", default=[])
    parser.add_argument("--positive-limit", type=int, default=200)
    parser.add_argument("--negative-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-threshold", type=float, default=0.50)
    parser.add_argument("--max-threshold", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--min-positive-recall", type=float, default=0.70)
    parser.add_argument("--fallback-threshold", type=float, default=0.95)
    parser.add_argument("--confirmation-min-positive-recall", type=float, default=0.80)
    parser.add_argument("--confirmation-fallback-threshold", type=float, default=0.90)
    parser.add_argument("--output")
    parser.add_argument("--metadata-json")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    positive_wavs = collect_wavs(args.positive_dir, limit=max(0, args.positive_limit), seed=args.seed)
    negative_wavs = collect_wavs(args.negative_dir, limit=max(0, args.negative_limit), seed=args.seed)
    if not negative_wavs:
        print("WARNING: no negative WAV clips found; using conservative fallback threshold", flush=True)

    from openwakeword.model import Model

    model = Model(wakeword_models=[str(model_path)], inference_framework=args.framework)
    positives = score_wavs(model, positive_wavs, "positive") if positive_wavs else []
    negatives = score_wavs(model, negative_wavs, "negative") if negative_wavs else []

    thresholds = threshold_values(args.min_threshold, args.max_threshold, args.threshold_step)
    metrics = build_metrics(positives, negatives, thresholds, max(1, args.patience))
    recommended_threshold, reason = choose_threshold(
        metrics,
        max_false_positive_rate=max(0.0, args.max_false_positive_rate),
        min_positive_recall=max(0.0, min(1.0, args.min_positive_recall)),
        fallback_threshold=max(args.min_threshold, min(args.max_threshold, args.fallback_threshold)),
    )
    confirmation_min_recall = max(0.0, min(1.0, args.confirmation_min_positive_recall))
    recommended_confirmation_threshold, confirmation_reason = choose_confirmation_threshold(
        metrics,
        min_positive_recall=confirmation_min_recall,
        fallback_threshold=max(
            args.min_threshold,
            min(args.max_threshold, args.confirmation_fallback_threshold),
        ),
    )

    negative_scores = [float(row.get("max_score") or 0.0) for row in negatives]
    positive_scores = [float(row.get("max_score") or 0.0) for row in positives]
    recommended_row = next((row for row in metrics if float(row["threshold"]) == recommended_threshold), {})
    recommended_confirmation_row = next(
        (
            row
            for row in metrics
            if float(row["threshold"]) == recommended_confirmation_threshold
        ),
        {},
    )
    recommended_recall = recommended_row.get("positive_recall") if recommended_row else None
    recall_ready = recommended_recall is None or float(recommended_recall or 0.0) >= max(0.0, min(1.0, args.min_positive_recall))
    calibration = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "framework": args.framework,
        "recommended_threshold": recommended_threshold,
        "recommended_patience": max(1, args.patience),
        "recommendation_reason": reason,
        "recommended_confirmation_threshold": recommended_confirmation_threshold,
        "recommended_confirmation_patience": max(1, args.patience),
        "confirmation_recommendation_reason": confirmation_reason,
        "confirmation_min_positive_recall": confirmation_min_recall,
        "confirmation_ready": bool(
            recommended_confirmation_row
            and recommended_confirmation_row.get("positive_recall") is not None
            and float(recommended_confirmation_row.get("positive_recall") or 0.0)
            >= confirmation_min_recall
        ),
        "deployment_ready": bool(
            recommended_row
            and int(recommended_row.get("false_positive_clips") or 0) == 0
            and recommended_threshold < args.max_threshold
            and recall_ready
        ),
        "positive_clip_count": len(positives),
        "negative_clip_count": len(negatives),
        "positive_score_summary": {
            "max": max(positive_scores) if positive_scores else 0.0,
            "median": median(positive_scores) if positive_scores else 0.0,
            "p05": percentile(positive_scores, 5),
        },
        "negative_score_summary": {
            "max": max(negative_scores) if negative_scores else 0.0,
            "median": median(negative_scores) if negative_scores else 0.0,
            "p95": percentile(negative_scores, 95),
            "p99": percentile(negative_scores, 99),
        },
        "recommended_metrics": recommended_row,
        "recommended_confirmation_metrics": recommended_confirmation_row,
        "threshold_metrics": metrics,
        "top_negative_clips": trim_rows(negatives),
        "top_positive_clips": trim_rows(positives),
    }

    if args.output:
        Path(args.output).expanduser().resolve().write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    if args.metadata_json:
        update_metadata(Path(args.metadata_json).expanduser().resolve(), calibration)

    print(json.dumps(calibration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
