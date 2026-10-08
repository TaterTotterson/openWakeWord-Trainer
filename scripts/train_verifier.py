#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an openWakeWord custom verifier from reviewed clips.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--negative-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    positive_clips = sorted(str(path) for path in Path(args.positive_dir).glob("*.wav"))
    negative_clips = sorted(str(path) for path in Path(args.negative_dir).glob("*.wav"))
    if not positive_clips:
        raise SystemExit("No positive WAV clips found")
    if not negative_clips:
        raise SystemExit("No negative WAV clips found")

    import openwakeword

    trainer = getattr(openwakeword, "train_custom_verifier", None)
    if trainer is None:
        try:
            from openwakeword.utils import train_custom_verifier as trainer
        except Exception as exc:
            raise SystemExit(f"Could not import train_custom_verifier: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_model = Path(args.base_model).expanduser().resolve()
    trainer(
        positive_reference_clips=positive_clips,
        negative_reference_clips=negative_clips,
        output_path=str(output),
        model_name=str(base_model),
        inference_framework="onnx",
    )
    print(f"Saved verifier: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
