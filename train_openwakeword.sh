#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./train_openwakeword.sh "hey tater" [options]

Common options are passed through to scripts/train_openwakeword.py:
  --samples 20000
  --validation-samples 2000
  --steps 50000
  --negative-tts-batch-divisor 7
  --custom-negative-phrase "phrase"
  --train-verifier

Environment:
  OWW_NEGATIVE_FEATURES=full|skip   default: full
  OWW_DOWNLOAD_BACKGROUND=1|0       default: 1
  OWW_DOWNLOAD_RIRS=1|0             default: 1
  OWW_PIPER_DEVICE=auto|mps|cuda|cpu default: auto
  OWW_NEGATIVE_TTS_DIVISOR=7        lower is faster but uses more memory
  OWW_FORCE_CPU=1                   disable CUDA/MPS visibility
  OWW_DATA_DIR=/data                persistent Docker data root
  OWW_TORCH_CUDA=cu124              install CUDA PyTorch wheels in the training venv
EOF
  exit 1
fi

if [[ -n "${OWW_DATA_DIR:-}" ]]; then
  DATA_ROOT="$OWW_DATA_DIR"
  ASSET_DIR="${OWW_ASSET_DIR:-$DATA_ROOT}"
else
  DATA_ROOT="$ROOT_DIR"
  ASSET_DIR="${OWW_ASSET_DIR:-$ROOT_DIR/data}"
fi
VENDOR_DIR="${OWW_VENDOR_DIR:-$DATA_ROOT/vendor}"
OPENWAKEWORD_DIR="${OWW_OPENWAKEWORD_DIR:-$VENDOR_DIR/openwakeword}"
PIPER_DIR="${OWW_PIPER_DIR:-$VENDOR_DIR/piper-sample-generator}"
OUTPUT_ROOT="${OWW_OUTPUT_ROOT:-$DATA_ROOT/output}"
EXPORT_DIR="${OWW_EXPORT_DIR:-${OWW_TRAINED_DIR:-$DATA_ROOT/trained_wake_words}}"
POSITIVE_DIR="${OWW_PERSONAL_DIR:-$DATA_ROOT/personal_samples}"
NEGATIVE_DIR="${OWW_NEGATIVE_DIR:-$DATA_ROOT/negative_samples}"
VENV_DIR="${OWW_VENV_DIR:-$DATA_ROOT/.venv}"
PIPER_REPO_URL="${OWW_PIPER_REPO_URL:-https://github.com/TaterTotterson/piper-sample-generator.git}"
if [[ -n "${OWW_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$OWW_PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.10)"
elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.11"
elif [[ -x /opt/homebrew/bin/python3.10 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.10"
else
  PYTHON_BIN=""
fi
PY="$VENV_DIR/bin/python"
MPLCONFIGDIR="${MPLCONFIGDIR:-$DATA_ROOT/.cache/matplotlib}"
PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:pkg_resources is deprecated as an API:UserWarning}"
TRAIN_REQUIREMENTS_KEY="$(cksum "$ROOT_DIR/requirements-train.txt" | awk '{print $1 ":" $2}')"
TRAIN_DEPS_KEY="cpu:${TRAIN_REQUIREMENTS_KEY}"
if [[ -n "${OWW_TORCH_CUDA:-}" && "${OWW_FORCE_CPU:-0}" != "1" ]]; then
  TRAIN_DEPS_KEY="${OWW_TORCH_VERSION:-2.6.0}+${OWW_TORCH_CUDA}:${TRAIN_REQUIREMENTS_KEY}"
fi

mkdir -p \
  "$ASSET_DIR" \
  "$VENDOR_DIR" \
  "$OUTPUT_ROOT" \
  "$EXPORT_DIR" \
  "$POSITIVE_DIR" \
  "$NEGATIVE_DIR" \
  "$DATA_ROOT/logs" \
  "$MPLCONFIGDIR"
export MPLCONFIGDIR PYTHONWARNINGS OWW_OPENWAKEWORD_DIR="$OPENWAKEWORD_DIR" OWW_PIPER_DIR="$PIPER_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ Training requires Python 3.10+ because openWakeWord 0.6.0 declares python_requires >=3.10."
  echo "   Install one, then rerun. On Apple Silicon:"
  echo "     brew install python@3.11"
  echo "   Or set:"
  echo "     OWW_PYTHON_BIN=/path/to/python3.11 ./train_openwakeword.sh ..."
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$PYTHON_VERSION" in
  3.10|3.11|3.12) : ;;
  *)
    echo "❌ Training requires Python 3.10+; found $PYTHON_VERSION at $PYTHON_BIN."
    echo "   Install Homebrew Python 3.11 and rerun:"
    echo "     brew install python@3.11"
    echo "     rm -rf .venv"
    echo "     ./train_openwakeword.sh \"hey tater\" ..."
    exit 1
    ;;
esac

if [[ ! -x "$PY" ]]; then
  echo "Creating training venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

VENV_PYTHON_VERSION="$("$PY" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
case "$VENV_PYTHON_VERSION" in
  3.10|3.11|3.12) : ;;
  *)
    echo "❌ Existing training venv uses Python $VENV_PYTHON_VERSION, but openWakeWord needs Python 3.10+."
    echo "   Remove the old venv after installing Python 3.11:"
    echo "     rm -rf .venv"
    echo "     ./train_openwakeword.sh \"hey tater\" ..."
    exit 1
    ;;
esac

install_cuda_torch() {
  local torch_version torch_cuda
  torch_version="${OWW_TORCH_VERSION:-2.6.0}"
  torch_cuda="${OWW_TORCH_CUDA:-}"
  if [[ -z "$torch_cuda" || "${OWW_FORCE_CPU:-0}" == "1" ]]; then
    return
  fi

  echo "Installing CUDA PyTorch wheels: torch ${torch_version}+${torch_cuda}"
  "$PY" -m pip install --index-url "https://download.pytorch.org/whl/${torch_cuda}" \
    "torch==${torch_version}+${torch_cuda}" \
    "torchaudio==${torch_version}+${torch_cuda}"
}

write_filtered_train_requirements() {
  local filtered="$VENV_DIR/requirements-train.no-torch.txt"
  "$PY" - "$ROOT_DIR/requirements-train.txt" "$filtered" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
skip = {"torch", "torchaudio"}
lines = []

for line in source.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
    if match and match.group(1).lower() in skip:
        continue
    lines.append(line)

dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(dest)
PY
}

if [[ ! -f "$VENV_DIR/.train_deps_installed" || "$(cat "$VENV_DIR/.train_deps_key" 2>/dev/null || true)" != "$TRAIN_DEPS_KEY" ]]; then
  "$PY" -m pip install -U pip setuptools wheel
  if [[ -n "${OWW_TORCH_CUDA:-}" && "${OWW_FORCE_CPU:-0}" != "1" ]]; then
    install_cuda_torch
    FILTERED_REQUIREMENTS="$(write_filtered_train_requirements)"
    "$PY" -m pip install -r "$FILTERED_REQUIREMENTS"
  else
    "$PY" -m pip install -r requirements-train.txt
  fi

  touch "$VENV_DIR/.train_deps_installed"
  printf "%s\n" "$TRAIN_DEPS_KEY" > "$VENV_DIR/.train_deps_key"
else
  echo "Reusing existing training venv"
fi

if ! "$PY" - <<'PY' >/dev/null 2>&1
import pkg_resources
PY
then
  echo "Repairing training venv: installing setuptools with pkg_resources support"
  "$PY" -m pip install "setuptools==80.9.0"
fi

if [[ ! -d "$OPENWAKEWORD_DIR/.git" ]]; then
  git clone https://github.com/dscripka/openWakeWord.git "$OPENWAKEWORD_DIR"
else
  git -C "$OPENWAKEWORD_DIR" pull --ff-only origin main || true
fi

"$PY" scripts/patch_openwakeword_device.py "$OPENWAKEWORD_DIR"

is_current_piper_layout() {
  [[ -f "$PIPER_DIR/piper_sample_generator/__main__.py" && -d "$PIPER_DIR/piper_train" ]]
}

refresh_piper_checkout() {
  local backup_dir model_cache piper_parent
  piper_parent="$(dirname "$PIPER_DIR")"
  backup_dir="$PIPER_DIR.backup.$(date +%Y%m%d%H%M%S)"
  model_cache="$piper_parent/.piper-model-cache"

  echo "Refreshing stale piper-sample-generator checkout from $PIPER_REPO_URL"
  mkdir -p "$model_cache" "$piper_parent"
  if [[ -d "$PIPER_DIR/models" ]]; then
    cp -R "$PIPER_DIR/models/." "$model_cache/" 2>/dev/null || true
  fi

  if [[ -d "$PIPER_DIR" ]]; then
    mv "$PIPER_DIR" "$backup_dir"
    echo "Backed up previous Piper checkout to $backup_dir"
  fi

  if ! git clone "$PIPER_REPO_URL" "$PIPER_DIR"; then
    if [[ -d "$backup_dir" && ! -d "$PIPER_DIR" ]]; then
      mv "$backup_dir" "$PIPER_DIR"
    fi
    echo "❌ Failed to refresh piper-sample-generator from $PIPER_REPO_URL"
    exit 1
  fi
  if [[ -d "$model_cache" ]]; then
    mkdir -p "$PIPER_DIR/models"
    cp -R "$model_cache/." "$PIPER_DIR/models/" 2>/dev/null || true
  fi
}

if [[ ! -d "$PIPER_DIR/.git" ]]; then
  refresh_piper_checkout
else
  current_piper_origin="$(git -C "$PIPER_DIR" remote get-url origin 2>/dev/null || true)"
  if [[ "$current_piper_origin" != "$PIPER_REPO_URL" ]]; then
    echo "Updating piper-sample-generator origin: $PIPER_REPO_URL"
    git -C "$PIPER_DIR" remote set-url origin "$PIPER_REPO_URL"
  fi

  if ! is_current_piper_layout; then
    refresh_piper_checkout
  elif git -C "$PIPER_DIR" diff --quiet && git -C "$PIPER_DIR" diff --cached --quiet; then
    git -C "$PIPER_DIR" pull --ff-only origin master || true
  else
    echo "Skipping piper-sample-generator pull; local trainer patches are already applied"
  fi
fi

"$PY" scripts/patch_piper_generator.py "$PIPER_DIR"
# The Linux PyPI metadata for openwakeword requires tflite-runtime even when
# only its ONNX training/export path is used.  Python 3.12 (including the
# Blackwell image) has no matching tflite-runtime wheel, so install the
# vendored source without resolving that unused backend.  Every dependency
# needed by Tater's ONNX path is installed explicitly above.
"$PY" -m pip install --no-build-isolation --no-deps -e "$OPENWAKEWORD_DIR"
if ! "$PY" -m pip install -e "$PIPER_DIR"; then
  echo "WARNING: editable piper-sample-generator install failed; retrying vendored fork without dependency resolution"
  "$PY" -m pip install --no-build-isolation --force-reinstall --no-deps -e "$PIPER_DIR"
fi

check_piper_generator() {
  local quiet="${1:-0}"
  "$PY" - "$PIPER_DIR" "$quiet" <<'PY'
import sys
from pathlib import Path

piper_root = Path(sys.argv[1]).resolve()
quiet = sys.argv[2] == "1"
sys.path.insert(0, str(piper_root))

try:
    from generate_samples import generate_samples  # noqa: F401
    import piper_sample_generator
    import piper_train  # noqa: F401
except Exception as exc:
    if not quiet:
        print(f"Piper sample generator import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"Piper sample generator ready: {piper_sample_generator.__file__}")
PY
}

if ! check_piper_generator 1 >/dev/null 2>&1; then
  echo "Repairing piper-sample-generator editable install"
  "$PY" -m pip install --no-build-isolation --force-reinstall --no-deps -e "$PIPER_DIR"
  check_piper_generator
else
  check_piper_generator
fi

DOWNLOAD_ARGS=()
if [[ "${OWW_NEGATIVE_FEATURES:-full}" == "full" ]]; then
  DOWNLOAD_ARGS+=(--negative-features full)
else
  DOWNLOAD_ARGS+=(--negative-features skip)
fi

if [[ "${OWW_DOWNLOAD_BACKGROUND:-1}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--background-hours "${OWW_BACKGROUND_HOURS:-1}")
else
  DOWNLOAD_ARGS+=(--skip-background)
fi

if [[ "${OWW_DOWNLOAD_RIRS:-1}" == "1" ]]; then
  :
else
  DOWNLOAD_ARGS+=(--skip-rirs)
fi

"$PY" scripts/download_assets.py --data-dir "$ASSET_DIR" "${DOWNLOAD_ARGS[@]}"

TRAIN_ARGS=(
  --data-dir "$ASSET_DIR"
  --output-root "$OUTPUT_ROOT"
  --export-dir "$EXPORT_DIR"
  --openwakeword-dir "$OPENWAKEWORD_DIR"
  --piper-dir "$PIPER_DIR"
  --positive-dir "$POSITIVE_DIR"
  --negative-dir "$NEGATIVE_DIR"
  "$@"
)
if [[ "${OWW_FORCE_CPU:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--force-cpu)
fi

"$PY" scripts/train_openwakeword.py "${TRAIN_ARGS[@]}"
