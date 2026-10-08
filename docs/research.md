# Research Notes

Checked May 2026 while scaffolding this trainer.

## openWakeWord

The official repository describes openWakeWord as a wake-word framework with pretrained models and custom model training. Its training docs still present two paths:

- a simple Colab notebook for a quick model
- the more detailed automatic training notebook for better customization and generally stronger models

The automatic notebook writes a YAML config, runs `openwakeword/train.py --generate_clips`, then `--augment_clips`, then `--train_model`. It expects synthetic positive/adversarial clips, room impulse responses, background audio, false-positive validation features, and a large negative feature file.

Sources:

- https://github.com/dscripka/openWakeWord
- https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
- https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml

## Model Artifact Choice

ONNX is the only published classifier target here. It works naturally with openWakeWord's Python runtime and with the ONNX Runtime deployment used by Tater Echo firmware. Upstream may create temporary TFLite files while training, but this trainer does not publish them.

## Personal Samples

openWakeWord's full model training is primarily synthetic-positive plus large negative-data training. For real deployment clips, the upstream docs recommend custom verifier models as a second-stage filter. This trainer keeps the microWakeWord-style personal/false-wake sample workflow and uses those clips for verifier training after the base ONNX model exists.

Source:

- https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md

## Piper

The old notebooks clone `piper-sample-generator`. Current piper-sample-generator also exists as an installable Python package and supports normal Piper voices plus the LibriTTS generator model. This repo still vendors the source checkout for compatibility with openWakeWord's config field, while installing the package in the training environment.

Source:

- https://github.com/rhasspy/piper-sample-generator
- https://pypi.org/project/piper-sample-generator/

## Apple vs NVIDIA

NVIDIA/Linux remains the best final-training path because the original automated training flow and Piper dependencies were Linux-first and because large runs benefit heavily from CUDA.

The upstream trainer is PyTorch-based and selects `cuda:0` when CUDA is available, otherwise CPU. This repo applies a small local patch after cloning openWakeWord so Apple Silicon can choose PyTorch MPS when available. MPS should be treated as an acceleration path to smoke-test and iterate locally; CUDA is still the safer target for the final high-sample run.
