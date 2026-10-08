<div align="center">
  <a href="https://taterassistant.com">
    <img src="images/tater-repo-logo.png" alt="openWakeWord Trainer" width="460"/>
  </a>
</div>
<h3 align="center">
  <a href="https://taterassistant.com">taterassistant.com</a>
</h3>

# openWakeWord Trainer for Apple Silicon and NVIDIA

Train custom openWakeWord models from a local web UI or CLI, with an ONNX classifier for Tater Echo firmware and optional verifier training from real positive and false-wake clips.

This project mirrors the shape of the Tater microWakeWord trainers, but the model strategy is different:

- openWakeWord runs a shared mel/embedding backbone and a small wake-word classifier.
- The best portable classifier artifact is ONNX.
- Real captured samples are most useful as a second-stage verifier for reducing false wakes in your actual room, mic, and voice setup.
- Full synthetic model training is still happiest on Linux/NVIDIA. Apple Silicon can run the UI, curate samples, test ONNX models, and run smaller native training jobs when dependencies cooperate.

## Why This Path

The official openWakeWord docs still recommend the automated training notebook for production-ish models, and describe the quick Colab as convenient but weaker in some deployments. The automated flow generates target/adversarial clips, augments them with room/background audio, trains against large negative feature sets, and produces the ONNX classifier used by Tater.

Useful upstream references:

- [openWakeWord README](https://github.com/dscripka/openWakeWord#training-new-models)
- [official automatic training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
- [official training config](https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)

## Quick Start: Web UI

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8791
```

The web UI stores:

```text
personal_samples/       real positive wake-word clips
negative_samples/       reviewed false wakes and hard negatives
captured_audio/         inbox for device uploads
trained_wake_words/     exported .onnx, .pkl, and metadata files
logs/                   training logs
```

## Train From The CLI

```bash
./train_openwakeword.sh "hey tater"
```

Python 3.10+ is required for training because openWakeWord 0.6.0 declares `python_requires >=3.10`; Python 3.11 is recommended. Set `OWW_PYTHON_BIN=/path/to/python3.11` if needed.

On Apple Silicon:

```bash
brew install python@3.11
rm -rf .venv
OWW_PYTHON_BIN=/opt/homebrew/bin/python3.11 ./train_openwakeword.sh "hey tater"
```

Balanced defaults:

```bash
./train_openwakeword.sh "hey tater" \
  --samples 20000 \
  --validation-samples 2000 \
  --steps 50000 \
  --custom-negative-phrase "potato"
```

Adversarial negative clips default to `tts_batch_size // 7`, matching upstream openWakeWord. On Apple Silicon MPS you can lower the divisor for larger, faster negative batches:

```bash
./train_openwakeword.sh "hey tater" \
  --tts-batch-size 50 \
  --negative-tts-batch-divisor 4
```

Fast smoke test:

```bash
OWW_NEGATIVE_FEATURES=skip ./train_openwakeword.sh "hey tater" \
  --samples 1000 \
  --validation-samples 300 \
  --steps 3000
```

Best-result run:

```bash
OWW_NEGATIVE_FEATURES=full ./train_openwakeword.sh "hey tater" \
  --samples 50000 \
  --validation-samples 5000 \
  --steps 80000 \
  --target-fp-per-hour 0.2 \
  --max-negative-weight 1500
```

The first full run can download many GB of data, including the openWakeWord negative feature file. Keep the `data/` directory around so later runs reuse it. Background audio downloads try the streamed FMA dataset first; if HuggingFace's FMA loader cannot stream on your machine, the trainer generates local 16 kHz fallback background clips and continues. Set `OWW_BACKGROUND_ALLOW_FULL_FMA=1` only if you want to allow the full 7.2 GiB FMA small download.

## NVIDIA Docker

Build:

```bash
docker build -f dockerfile -t openwakeword-trainer .
```

Run:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=8g \
  -p 8791:8791 \
  -v "$(pwd)/data":/data \
  openwakeword-trainer
```

Open:

```text
http://localhost:8791
```

The Docker image follows the smaller microWakeWord trainer pattern: it ships only the app, Python, and system tools. UI and training dependencies are installed into `/data/.recorder-venv` and `/data/.venv` on first run, so rebuilding the image stays small and the heavy Python stack is cached in your mounted `data/` directory.

By default Docker sets `OWW_TORCH_CUDA=cu124`, so the training venv installs the CUDA 12.4 PyTorch wheels and the upstream trainer should select `cuda:0` when run with `--gpus all`. For a CPU-only container, pass `-e OWW_FORCE_CPU=1 -e OWW_TORCH_CUDA=`.

If training fails with `Unexpected bus error encountered in worker`, Docker's shared-memory mount is too small. This is separate from free space in `/data`; PyTorch DataLoader workers use `/dev/shm` while passing batches between worker processes. Keep `--shm-size=8g` on the run command, or use `--ipc=host` on a trusted local machine. The image also defaults to `OWW_TRAIN_NUM_WORKERS=2` and `OWW_TRAIN_PREFETCH_FACTOR=2` to avoid overfilling shared memory; raise those only after confirming training stays real-time and stable.

## Apple Silicon Notes

Native Apple training is supported as a best-effort path. It uses the same scripts, creates `.venv`, installs the Python stack, and runs the upstream openWakeWord trainer. The launcher patches the vendored trainer so PyTorch selects Apple MPS when available, streams false-positive validation data in bounded batches instead of creating one multi-gigabyte MPS tensor, and lets `OWW_DEFAULT_NEGATIVE_BATCH` reduce the normal training batch on unified-memory Macs. Piper sample generation uses `OWW_PIPER_DEVICE=auto` so it prefers CUDA/MPS before falling back to CPU. Set `OWW_FORCE_CPU=1`, `OWW_ENABLE_MPS=0`, `OWW_PIPER_DEVICE=cpu`, or enable `force CPU` in the UI if an operation falls back poorly.

Because upstream automated training historically targeted Linux/Piper, NVIDIA Docker is still the recommended path for the big final run.

Apple is excellent for:

- running the local trainer UI
- collecting and reviewing clips
- testing ONNX models
- training verifier models from personal/negative clips
- short smoke training runs on CPU or MPS

## Captured Audio Endpoint

Devices can upload raw 16 kHz mono signed 16-bit PCM to:

```text
POST /api/upload_captured_audio_raw
```

Uploaded clips land in `captured_audio/`. In the UI, approve good wake-word examples into `personal_samples/` or mark false wakes into `negative_samples/`.

## Verifier Training

After a base `.onnx` model exists, train a verifier from your reviewed clips:

```bash
.venv/bin/python scripts/train_verifier.py \
  --base-model trained_wake_words/hey_tater.onnx \
  --positive-dir personal_samples \
  --negative-dir negative_samples \
  --output trained_wake_words/hey_tater_verifier.pkl
```

This follows openWakeWord's recommended second-stage filter idea: keep the general wake-word model broad, then use real clips to make your deployment less trigger-happy.

## Test A Model

```bash
.venv/bin/python scripts/test_model.py \
  --model trained_wake_words/hey_tater.onnx \
  --wav personal_samples/example.wav
```

For a deployment check, calibrate the model against reviewed false wakes and generated negative test clips:

```bash
.venv/bin/python scripts/calibrate_model.py \
  --model trained_wake_words/hey_tater.onnx \
  --positive-dir personal_samples \
  --negative-dir negative_samples \
  --negative-dir output/hey_tater/hey_tater/negative_test \
  --metadata-json trained_wake_words/hey_tater.json
```

The trainer runs this automatically after a successful ONNX export when negative clips are available. It reports separate policies because the model serves two different jobs: `recommended_threshold` and `recommended_patience` are the stricter standalone OWW settings, while `recommended_confirmation_threshold` and `recommended_confirmation_patience` preserve genuine-wake recall when OWW confirms an earlier MWW detection. Do not force the standalone threshold onto the dual-model confirmation path.

For live microphone testing, use the upstream openWakeWord examples after setup:

```bash
.venv/bin/python vendor/openwakeword/examples/detect_from_microphone.py \
  --model-path trained_wake_words/hey_tater.onnx
```

## Output

Successful training syncs artifacts into:

```text
trained_wake_words/<model>.onnx
trained_wake_words/<model>.json         # local metadata
trained_wake_words/<model>_verifier.pkl # optional
```

## Important Caveats

- openWakeWord's published pretrained/custom training flow is strongest for English.
- Threshold tuning matters. The default runtime threshold of `0.5` is a starting point, not a promise.
- If a custom model triggers repeatedly on silence or room audio, immediately raise the runtime threshold near `0.95`, increase patience to `3` or `4`, and collect those false-wake clips into `negative_samples/` before retraining or training a verifier.
- Reviewed false wakes are valuable. Use them for verifier training and as custom negative phrases when they are phrase-like.
- Big negative datasets improve false-positive behavior, but they cost disk and time.

## Credits

Built around:

- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
- the existing Tater microWakeWord trainer workflow
