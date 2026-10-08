#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("OWW_DATA_DIR", str(ROOT_DIR))).resolve()
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static"))).resolve()
PERSONAL_DIR = Path(os.environ.get("OWW_PERSONAL_DIR", str(DATA_DIR / "personal_samples"))).resolve()
NEGATIVE_DIR = Path(os.environ.get("OWW_NEGATIVE_DIR", str(DATA_DIR / "negative_samples"))).resolve()
CAPTURED_DIR = Path(os.environ.get("OWW_CAPTURED_DIR", str(DATA_DIR / "captured_audio"))).resolve()
TRIM_HISTORY_DIR = Path(os.environ.get("OWW_TRIM_HISTORY_DIR", str(DATA_DIR / "trim_history"))).resolve()
TRAINED_DIR = Path(
    os.environ.get("OWW_TRAINED_DIR", os.environ.get("OWW_EXPORT_DIR", str(DATA_DIR / "trained_wake_words")))
).resolve()
LOG_DIR = Path(os.environ.get("OWW_LOG_DIR", str(DATA_DIR / "logs"))).resolve()
TRAIN_SCRIPT = Path(os.environ.get("TRAIN_SCRIPT", str(ROOT_DIR / "train_openwakeword.sh"))).resolve()

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2
MAX_LOG_LINES = int(os.environ.get("OWW_MAX_LOG_LINES", "1200"))

for directory in (STATIC_DIR, PERSONAL_DIR, NEGATIVE_DIR, CAPTURED_DIR, TRIM_HISTORY_DIR, TRAINED_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="openWakeWord Trainer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STATE_LOCK = threading.Lock()
TRAIN_PROC: subprocess.Popen[str] | None = None
STATE: dict[str, Any] = {
    "training": {
        "running": False,
        "exit_code": None,
        "log_lines": [],
        "log_path": None,
        "safe_word": None,
        "started_at": None,
        "finished_at": None,
    }
}

_silero_vad_model = None
_silero_vad_utils = None
_SILERO_VAD_LOCK = threading.Lock()
VAD_SELECTION_PAD_START_S = 0.08
VAD_SELECTION_PAD_END_S = 0.08


def safe_name(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "wakeword"


def _audio_dir(kind: str) -> Path:
    if kind == "personal":
        return PERSONAL_DIR
    if kind == "negative":
        return NEGATIVE_DIR
    if kind == "captured":
        return CAPTURED_DIR
    raise HTTPException(status_code=404, detail="Unknown audio collection")


def _resolve_child(directory: Path, name: str) -> Path:
    candidate = Path(name or "").name
    if not candidate or candidate != (name or ""):
        raise HTTPException(status_code=400, detail="Invalid file path")
    path = (directory / candidate).resolve()
    if path.parent != directory.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")
    return path


def _audio_sidecar_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def _load_sidecar_json(audio_path: Path) -> dict[str, Any]:
    sidecar = _audio_sidecar_path(audio_path)
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_sidecar_json(audio_path: Path, payload: dict[str, Any]) -> None:
    _audio_sidecar_path(audio_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _wav_item(path: Path, directory: Path) -> dict[str, Any]:
    stat = path.stat()
    meta = _load_sidecar_json(path)
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "url": f"/api/audio/{directory.name}/{path.name}",
        "trimmed": bool(meta.get("trimmed")),
        "source_file": meta.get("source_file") or "",
    }


def _list_wavs(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True):
        rows.append(_wav_item(path, directory))
    rows.sort(key=lambda item: item.get("trimmed", False))
    return rows


def _list_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(TRAINED_DIR.glob("*")):
        if path.suffix.lower() not in {".onnx", ".data", ".pkl", ".json"}:
            continue
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": f"/api/artifacts/{path.name}",
            }
        )
    return rows


def _inspect_wav_bytes(data: bytes) -> dict[str, Any] | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = (frames / rate) if rate else 0.0
            return {
                "sample_rate": rate,
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
                "frames": frames,
                "duration_s": round(duration, 3),
            }
    except Exception:
        return None


def _is_target_wav_info(info: dict[str, Any] | None) -> bool:
    return bool(
        info
        and info.get("sample_rate") == TARGET_SAMPLE_RATE
        and info.get("channels") == TARGET_CHANNELS
        and info.get("sample_width") == TARGET_SAMPLE_WIDTH
        and int(info.get("frames") or 0) > 0
    )


def _target_wav_bytes(data: bytes, original_name: str) -> bytes:
    if _is_target_wav_info(_inspect_wav_bytes(data)):
        return data
    suffix = Path(original_name or "audio.wav").suffix.lower() or ".wav"
    with tempfile.TemporaryDirectory(prefix="oww_trim_") as tmpdir:
        src = Path(tmpdir) / f"source{suffix}"
        dest = Path(tmpdir) / "target.wav"
        src.write_bytes(data)
        _convert_audio(src, dest)
        return dest.read_bytes()


def _is_target_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getframerate() == TARGET_SAMPLE_RATE
                and wav.getnchannels() == TARGET_CHANNELS
                and wav.getsampwidth() == TARGET_SAMPLE_WIDTH
            )
    except Exception:
        return False


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _load_silero_vad():
    global _silero_vad_model, _silero_vad_utils
    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils
    with _SILERO_VAD_LOCK:
        if _silero_vad_model is not None:
            return _silero_vad_model, _silero_vad_utils
        import torch
        import silero_vad

        model = silero_vad.load_silero_vad()
        model.eval()
        _silero_vad_model = model
        _silero_vad_utils = {"torch": torch}
        return model, _silero_vad_utils


def _detect_speech_segments(wav_bytes: bytes) -> list[dict[str, float]]:
    model, utils = _load_silero_vad()
    torch = utils["torch"]
    import numpy as np
    from silero_vad.utils_vad import get_speech_timestamps

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    timestamps = get_speech_timestamps(
        torch.from_numpy(samples),
        model,
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=150,
        min_silence_duration_ms=100,
        return_seconds=True,
    )
    return [{"start": round(ts["start"], 3), "end": round(ts["end"], 3)} for ts in timestamps]


def _convert_audio(source: Path, dest: Path) -> None:
    if source.suffix.lower() == ".wav" and _is_target_wav(source):
        shutil.copy2(source, dest)
        return

    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise HTTPException(
            status_code=400,
            detail="ffmpeg is required for non-16k mono PCM WAV uploads",
        )

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(dest),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {result.stderr.strip()}")


def _write_raw_pcm_wav(blob: bytes, dest: Path, sample_rate: int) -> None:
    if sample_rate != TARGET_SAMPLE_RATE:
        ffmpeg = _ffmpeg_path()
        if not ffmpeg:
            raise HTTPException(status_code=400, detail="ffmpeg is required to resample raw captured audio")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as tmp:
            tmp.write(blob)
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                str(tmp_path),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                str(dest),
            ]
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                raise HTTPException(status_code=400, detail=f"Raw audio conversion failed: {result.stderr.strip()}")
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_SAMPLE_WIDTH)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(blob)


async def _save_upload(file: UploadFile, directory: Path, prefix: str) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.wav").suffix.lower() or ".wav"
    safe_prefix = safe_name(Path(file.filename or prefix).stem)[:50] or prefix
    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_prefix}_{uuid.uuid4().hex[:8]}.wav"
    dest = directory / out_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        _convert_audio(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"name": dest.name, "url": f"/api/audio/{directory.name}/{dest.name}"}


def _append_log(line: str, log_path: Path | None = None) -> None:
    clean = line.rstrip("\n")
    with STATE_LOCK:
        logs = STATE["training"]["log_lines"]
        logs.append(clean)
        if len(logs) > MAX_LOG_LINES:
            del logs[: len(logs) - MAX_LOG_LINES]
    if log_path:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(clean + "\n")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>openWakeWord Trainer</h1><p>static/index.html is missing.</p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> JSONResponse:
    with STATE_LOCK:
        training = dict(STATE["training"])
    return JSONResponse(
        {
            "training": training,
            "counts": {
                "personal": len(list(PERSONAL_DIR.glob("*.wav"))),
                "negative": len(list(NEGATIVE_DIR.glob("*.wav"))),
                "captured": len(list(CAPTURED_DIR.glob("*.wav"))),
                "artifacts": len(_list_artifacts()),
            },
            "artifacts": _list_artifacts(),
            "defaults": {
                "samples": int(os.environ.get("OWW_DEFAULT_SAMPLES", "20000")),
                "validation_samples": int(os.environ.get("OWW_DEFAULT_VALIDATION_SAMPLES", "2000")),
                "steps": int(os.environ.get("OWW_DEFAULT_STEPS", "50000")),
                "tts_batch_size": int(os.environ.get("OWW_DEFAULT_TTS_BATCH", "50")),
                "negative_tts_batch_divisor": int(os.environ.get("OWW_NEGATIVE_TTS_DIVISOR", "7")),
                "augmentation_batch_size": int(os.environ.get("OWW_DEFAULT_AUG_BATCH", "16")),
                "target_fp_per_hour": float(os.environ.get("OWW_DEFAULT_TARGET_FP", "0.2")),
                "max_negative_weight": int(os.environ.get("OWW_DEFAULT_MAX_NEGATIVE_WEIGHT", "1500")),
            },
        }
    )


@app.get("/api/samples/{kind}")
def list_samples(kind: str) -> JSONResponse:
    return JSONResponse({"items": _list_wavs(_audio_dir(kind))})


@app.post("/api/samples/{kind}/upload")
async def upload_samples(kind: str, files: list[UploadFile] = File(...)) -> JSONResponse:
    directory = _audio_dir(kind)
    saved = [await _save_upload(file, directory, kind) for file in files]
    return JSONResponse({"saved": saved})


@app.delete("/api/samples/{kind}/{name}")
def delete_sample(kind: str, name: str) -> JSONResponse:
    path = _resolve_child(_audio_dir(kind), name)
    path.unlink(missing_ok=True)
    with contextlib.suppress(Exception):
        path.with_suffix(".json").unlink()
    return JSONResponse({"ok": True})


@app.post("/api/samples/{kind}/{name}/vad")
def vad_segments(kind: str, name: str) -> JSONResponse:
    directory = _audio_dir(kind)
    path = _resolve_child(directory, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    wav_bytes = _target_wav_bytes(path.read_bytes(), path.name)
    try:
        all_segments = _detect_speech_segments(wav_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VAD failed: {exc}") from exc

    filtered = [segment for segment in all_segments if (segment["end"] - segment["start"]) >= 0.25]
    if not filtered:
        return JSONResponse({"ok": True, "segments": [], "segment_count": 0})

    segment = filtered[0]
    info = _inspect_wav_bytes(wav_bytes) or {}
    duration_s = float(info.get("duration_s") or 0.0)
    start = max(0.0, round(segment["start"] - VAD_SELECTION_PAD_START_S, 3))
    end = round(segment["end"] + VAD_SELECTION_PAD_END_S, 3)
    if duration_s > 0:
        end = min(duration_s, end)
    if end <= start:
        end = start + 0.001
    return JSONResponse({"ok": True, "segments": [{"start": start, "end": end}], "segment_count": 1})


@app.post("/api/samples/trim")
async def trim_sample_upload(
    file: UploadFile = File(...),
    kind: str = Form(...),
    source_file: str = Form(...),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
) -> JSONResponse:
    directory = _audio_dir(kind)
    source_path = _resolve_child(directory, source_file)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    data = _target_wav_bytes(data, file.filename or "trimmed.wav")

    TRIM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    backup_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{kind}_{source_path.name}"
    backup_path = TRIM_HISTORY_DIR / backup_name
    shutil.copy2(source_path, backup_path)
    source_sidecar = _audio_sidecar_path(source_path)
    if source_sidecar.exists():
        shutil.copy2(source_sidecar, _audio_sidecar_path(backup_path))

    previous_sidecar = _load_sidecar_json(source_path)
    source_path.write_bytes(data)
    sidecar = {
        **previous_sidecar,
        "trimmed": True,
        "source_file": previous_sidecar.get("source_file") or source_path.name,
        "source_kind": kind,
        "trim_start_s": float(start_time) if start_time else None,
        "trim_end_s": float(end_time) if end_time else None,
        "undo_backup_file": backup_name,
    }
    _write_sidecar_json(source_path, sidecar)

    return JSONResponse({"ok": True, "item": _wav_item(source_path, directory), "message": f"Trimmed {source_path.name}"})


@app.post("/api/samples/revert")
def revert_trim(kind: str = Form(...), name: str = Form(...)) -> JSONResponse:
    directory = _audio_dir(kind)
    path = _resolve_child(directory, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    sidecar = _load_sidecar_json(path)
    backup_name = str(sidecar.get("undo_backup_file") or "")
    if not backup_name:
        raise HTTPException(status_code=400, detail="No trim backup found for this sample")
    backup_path = _resolve_child(TRIM_HISTORY_DIR, backup_name)
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Trim backup file missing")

    shutil.copy2(backup_path, path)
    backup_sidecar = _audio_sidecar_path(backup_path)
    if backup_sidecar.exists():
        shutil.copy2(backup_sidecar, _audio_sidecar_path(path))
    else:
        _audio_sidecar_path(path).unlink(missing_ok=True)

    backup_path.unlink(missing_ok=True)
    backup_sidecar.unlink(missing_ok=True)
    return JSONResponse({"ok": True, "item": _wav_item(path, directory), "message": f"Reverted {path.name}"})


@app.delete("/api/samples/{kind}")
def clear_samples(kind: str) -> JSONResponse:
    directory = _audio_dir(kind)
    for path in directory.glob("*.wav"):
        path.unlink(missing_ok=True)
    for path in directory.glob("*.json"):
        path.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.get("/api/audio/{collection}/{name}")
def get_audio(collection: str, name: str) -> FileResponse:
    directory_by_name = {
        PERSONAL_DIR.name: PERSONAL_DIR,
        NEGATIVE_DIR.name: NEGATIVE_DIR,
        CAPTURED_DIR.name: CAPTURED_DIR,
    }
    directory = directory_by_name.get(collection)
    if directory is None:
        raise HTTPException(status_code=404, detail="Unknown audio collection")
    path = _resolve_child(directory, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/api/captured/{name}/approve")
def approve_captured(name: str) -> JSONResponse:
    src = _resolve_child(CAPTURED_DIR, name)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Captured clip not found")
    dest = PERSONAL_DIR / src.name
    shutil.move(str(src), str(dest))
    src_sidecar = _audio_sidecar_path(src)
    if src_sidecar.exists():
        shutil.move(str(src_sidecar), str(_audio_sidecar_path(dest)))
    return JSONResponse({"ok": True, "name": dest.name})


@app.post("/api/captured/{name}/false_wake")
def false_wake_captured(name: str) -> JSONResponse:
    src = _resolve_child(CAPTURED_DIR, name)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Captured clip not found")
    dest = NEGATIVE_DIR / src.name
    shutil.move(str(src), str(dest))
    src_sidecar = _audio_sidecar_path(src)
    if src_sidecar.exists():
        shutil.move(str(src_sidecar), str(_audio_sidecar_path(dest)))
    return JSONResponse({"ok": True, "name": dest.name})


@app.post("/api/upload_captured_audio_raw")
async def upload_captured_audio_raw(request: Request) -> JSONResponse:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="No audio body received")

    sample_rate = int(request.query_params.get("sample_rate", request.headers.get("x-sample-rate", TARGET_SAMPLE_RATE)))
    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_captured_{uuid.uuid4().hex[:8]}.wav"
    dest = CAPTURED_DIR / out_name
    _write_raw_pcm_wav(body, dest, sample_rate)
    return JSONResponse({"ok": True, "name": dest.name, "url": f"/api/audio/{CAPTURED_DIR.name}/{dest.name}"})


@app.post("/api/train")
async def start_training(request: Request) -> JSONResponse:
    global TRAIN_PROC

    payload = await request.json()
    phrase = str(payload.get("phrase") or "").strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="Wake phrase is required")

    with STATE_LOCK:
        if STATE["training"]["running"]:
            raise HTTPException(status_code=409, detail="Training is already running")

    safe_word = safe_name(phrase)
    log_path = LOG_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_word}.log"

    cmd = [
        str(TRAIN_SCRIPT),
        phrase,
        "--samples",
        str(int(payload.get("samples") or 20000)),
        "--validation-samples",
        str(int(payload.get("validation_samples") or 2000)),
        "--steps",
        str(int(payload.get("steps") or 50000)),
        "--tts-batch-size",
        str(int(payload.get("tts_batch_size") or 50)),
        "--negative-tts-batch-divisor",
        str(int(payload.get("negative_tts_batch_divisor") or 7)),
        "--augmentation-batch-size",
        str(int(payload.get("augmentation_batch_size") or 16)),
        "--target-fp-per-hour",
        str(float(payload.get("target_fp_per_hour") or 0.2)),
        "--max-negative-weight",
        str(int(payload.get("max_negative_weight") or 1500)),
    ]

    for item in payload.get("custom_negative_phrases") or []:
        phrase_item = str(item or "").strip()
        if phrase_item:
            cmd.extend(["--custom-negative-phrase", phrase_item])

    if payload.get("train_verifier"):
        cmd.append("--train-verifier")
    if payload.get("force_cpu"):
        cmd.append("--force-cpu")

    with STATE_LOCK:
        STATE["training"] = {
            "running": True,
            "exit_code": None,
            "log_lines": [],
            "log_path": str(log_path),
            "safe_word": safe_word,
            "started_at": time.time(),
            "finished_at": None,
        }

    def worker() -> None:
        global TRAIN_PROC
        _append_log(f"$ {' '.join(cmd)}", log_path)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            TRAIN_PROC = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert TRAIN_PROC.stdout is not None
            for line in TRAIN_PROC.stdout:
                _append_log(line, log_path)
            exit_code = TRAIN_PROC.wait()
        except Exception as exc:
            _append_log(f"ERROR: {exc}", log_path)
            exit_code = 1
        finally:
            TRAIN_PROC = None
            with STATE_LOCK:
                STATE["training"]["running"] = False
                STATE["training"]["exit_code"] = exit_code
                STATE["training"]["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"ok": True, "log_path": str(log_path), "safe_word": safe_word})


@app.post("/api/train/stop")
def stop_training() -> JSONResponse:
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        TRAIN_PROC.terminate()
        return JSONResponse({"ok": True, "message": "Training process terminated"})
    return JSONResponse({"ok": True, "message": "No training process is running"})


@app.get("/api/train/log")
def training_log() -> JSONResponse:
    with STATE_LOCK:
        training = dict(STATE["training"])
        lines = list(training.get("log_lines") or [])
    return JSONResponse({"training": training, "lines": lines})


@app.get("/api/artifacts")
def artifacts() -> JSONResponse:
    return JSONResponse({"items": _list_artifacts()})


@app.get("/api/artifacts/{name}")
def get_artifact(name: str) -> FileResponse:
    path = _resolve_child(TRAINED_DIR, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)
