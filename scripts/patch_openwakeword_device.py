#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEVICE_ORIGINAL = "self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
DEVICE_PATCHED = """if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif os.environ.get("OWW_ENABLE_MPS", "1") == "1" and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')"""

NEGATIVE_BATCH_ORIGINAL = 'batch_size=config["tts_batch_size"]//7,'
NEGATIVE_BATCH_PATCHED = (
    'batch_size=max(1, int(config["tts_batch_size"])//'
    'max(1, int(config.get("negative_tts_batch_divisor", 7)))),'
)

FEATURE_SKIP_ORIGINAL = (
    'if not os.path.exists(os.path.join(feature_save_dir, "positive_features_train.npy")) '
    'or args.overwrite is True:'
)
FEATURE_SKIP_PATCHED = (
    'if args.overwrite is True or not all(os.path.exists(os.path.join(feature_save_dir, name)) '
    'for name in ["positive_features_train.npy", "negative_features_train.npy", '
    '"positive_features_test.npy", "negative_features_test.npy"]):'
)

STORE_TRUE_DEFAULT_ORIGINAL = '        default="False",'
STORE_TRUE_DEFAULT_PATCHED = "        default=False,"

TRAIN_LOADER_ORIGINAL = '''        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)'''
TRAIN_LOADER_PATCHED = '''        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        if sys.platform == "darwin":
            n_cpus = 0
        n_cpus = int(os.environ.get("OWW_TRAIN_NUM_WORKERS", n_cpus))
        prefetch_factor = max(1, int(os.environ.get("OWW_TRAIN_PREFETCH_FACTOR", "2")))
        train_loader_kwargs = {"batch_size": None, "num_workers": n_cpus}
        if n_cpus > 0:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              **train_loader_kwargs)'''

TRAIN_LOADER_PATCHED_LEGACY = '''        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        if sys.platform == "darwin":
            n_cpus = 0
        n_cpus = int(os.environ.get("OWW_TRAIN_NUM_WORKERS", n_cpus))
        train_loader_kwargs = {"batch_size": None, "num_workers": n_cpus}
        if n_cpus > 0:
            train_loader_kwargs["prefetch_factor"] = 16
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              **train_loader_kwargs)'''

VALIDATION_LOADER_ORIGINAL = '''        X_val_fp = np.load(config["false_positive_validation_data_path"])
        X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model
        X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=len(X_val_fp_labels)
        )'''
VALIDATION_LOADER_PATCHED = '''        class FalsePositiveValidationDataset(torch.utils.data.IterableDataset):
            def __init__(self, features, window_size, batch_size):
                self.features = features
                self.window_size = window_size
                self.batch_size = batch_size

            def __iter__(self):
                n_windows = max(0, self.features.shape[0] - self.window_size)
                for start in range(0, n_windows, self.batch_size):
                    stop = min(n_windows, start + self.batch_size)
                    source = self.features[start:stop+self.window_size-1]
                    windows = np.lib.stride_tricks.sliding_window_view(
                        source,
                        window_shape=self.window_size,
                        axis=0,
                    ).swapaxes(1, 2)
                    # Copy only this bounded batch from the memory-mapped
                    # source. The original implementation expanded every
                    # overlapping window into a multi-gigabyte array.
                    windows = np.array(windows, dtype=np.float32, copy=True)
                    yield (
                        torch.from_numpy(windows),
                        torch.zeros(windows.shape[0], dtype=torch.float32),
                    )

        X_val_fp_source = np.load(config["false_positive_validation_data_path"], mmap_mode="r")
        X_val_fp = torch.utils.data.DataLoader(
            FalsePositiveValidationDataset(
                X_val_fp_source,
                input_shape[0],
                max(1, int(config.get("validation_batch_size", 2048))),
            ),
            batch_size=None,
            num_workers=0,
        )'''

RESAMPLE_ORIGINAL = '''            if clip_sr != sr:
                raise ValueError("Error! Clip does not have the correct sample rate!")'''
RESAMPLE_PATCHED = '''            if clip_sr != sr:
                clip_data = torchaudio.functional.resample(clip_data, orig_freq=clip_sr, new_freq=sr)
                clip_sr = sr'''

OUTPUT_TYPE_REPLACEMENTS = (
    (
        '''                mode="per_batch"
            ),''',
        '''                mode="per_batch",
                output_type="tensor"
            ),''',
    ),
    (
        '''            torch_audiomentations.BandStopFilter(p=augmentation_probabilities["BandStopFilter"], mode="per_batch"),''',
        '''            torch_audiomentations.BandStopFilter(
                p=augmentation_probabilities["BandStopFilter"],
                mode="per_batch",
                output_type="tensor"
            ),''',
    ),
    (
        '''            torch_audiomentations.Gain(max_gain_in_db=0, p=augmentation_probabilities["Gain"]),''',
        '''            torch_audiomentations.Gain(
                max_gain_in_db=0,
                p=augmentation_probabilities["Gain"],
                output_type="tensor"
            ),''',
    ),
    (
        '''        ])
    else:
        augment2 = torch_audiomentations.Compose([''',
        '''        ], output_type="tensor")
    else:
        augment2 = torch_audiomentations.Compose([''',
    ),
    (
        '''        ])

    # Iterate through all clips and augment them''',
        '''        ], output_type="tensor")

    # Iterate through all clips and augment them''',
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch vendored openWakeWord training for Apple MPS and local trainer settings."
    )
    parser.add_argument("openwakeword_dir")
    args = parser.parse_args()

    train_py = Path(args.openwakeword_dir).resolve() / "openwakeword" / "train.py"
    if not train_py.exists():
        raise SystemExit(f"openWakeWord train.py not found: {train_py}")

    text = train_py.read_text(encoding="utf-8")
    changed = False

    if "OWW_ENABLE_MPS" in text:
        print(f"MPS device patch already present: {train_py}", flush=True)
    else:
        if DEVICE_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream device-selection line in {train_py}")
        text = text.replace(DEVICE_ORIGINAL, DEVICE_PATCHED)
        changed = True
        print(f"Patched openWakeWord device selection for MPS: {train_py}", flush=True)

    if "negative_tts_batch_divisor" in text:
        print(f"Negative TTS batch divisor patch already present: {train_py}", flush=True)
    else:
        if NEGATIVE_BATCH_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream negative batch-size line in {train_py}")
        text = text.replace(NEGATIVE_BATCH_ORIGINAL, NEGATIVE_BATCH_PATCHED)
        changed = True
        print(f"Patched openWakeWord negative TTS batch divisor: {train_py}", flush=True)

    if "positive_features_test.npy\", \"negative_features_test.npy" in text:
        print(f"Feature completeness patch already present: {train_py}", flush=True)
    else:
        if FEATURE_SKIP_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream feature-skip line in {train_py}")
        text = text.replace(FEATURE_SKIP_ORIGINAL, FEATURE_SKIP_PATCHED)
        changed = True
        print(f"Patched openWakeWord feature completeness check: {train_py}", flush=True)

    if STORE_TRUE_DEFAULT_ORIGINAL in text:
        text = text.replace(STORE_TRUE_DEFAULT_ORIGINAL, STORE_TRUE_DEFAULT_PATCHED)
        changed = True
        print(f"Patched openWakeWord store_true defaults: {train_py}", flush=True)
    else:
        print(f"openWakeWord store_true defaults already patched: {train_py}", flush=True)

    if "OWW_TRAIN_PREFETCH_FACTOR" in text:
        print(f"Mac DataLoader worker patch already present: {train_py}", flush=True)
    elif TRAIN_LOADER_PATCHED_LEGACY in text:
        text = text.replace(TRAIN_LOADER_PATCHED_LEGACY, TRAIN_LOADER_PATCHED)
        changed = True
        print(f"Updated openWakeWord training DataLoader worker prefetch patch: {train_py}", flush=True)
    else:
        if TRAIN_LOADER_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream training DataLoader block in {train_py}")
        text = text.replace(TRAIN_LOADER_ORIGINAL, TRAIN_LOADER_PATCHED)
        changed = True
        print(f"Patched openWakeWord training DataLoader workers for macOS spawn: {train_py}", flush=True)

    if "FalsePositiveValidationDataset" in text:
        print(f"Memory-safe validation loader patch already present: {train_py}", flush=True)
    else:
        if VALIDATION_LOADER_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream false-positive validation loader in {train_py}")
        text = text.replace(VALIDATION_LOADER_ORIGINAL, VALIDATION_LOADER_PATCHED)
        changed = True
        print(f"Patched openWakeWord validation loader to stream bounded batches: {train_py}", flush=True)

    if changed:
        train_py.write_text(text, encoding="utf-8")

    data_py = Path(args.openwakeword_dir).resolve() / "openwakeword" / "data.py"
    if not data_py.exists():
        raise SystemExit(f"openWakeWord data.py not found: {data_py}")

    data_text = data_py.read_text(encoding="utf-8")
    data_changed = False
    if "torchaudio.functional.resample(clip_data" in data_text:
        print(f"Augmentation resample patch already present: {data_py}", flush=True)
    else:
        if RESAMPLE_ORIGINAL not in data_text:
            raise SystemExit(f"Could not find upstream sample-rate check in {data_py}")
        data_text = data_text.replace(RESAMPLE_ORIGINAL, RESAMPLE_PATCHED)
        data_changed = True
        print(f"Patched openWakeWord augmentation sample-rate resampling: {data_py}", flush=True)

    if data_text.count('output_type="tensor"') >= 11:
        print(f"Torch audiomentations output type patch already present: {data_py}", flush=True)
    else:
        if 'output_type="tensor"' in data_text:
            raise SystemExit(f"Partial torch audiomentations output type patch found in {data_py}")
        for original, patched in OUTPUT_TYPE_REPLACEMENTS:
            if original not in data_text:
                raise SystemExit(f"Could not find upstream torch audiomentations pattern in {data_py}")
            data_text = data_text.replace(original, patched)
        data_changed = True
        print(f"Patched torch audiomentations output_type compatibility: {data_py}", flush=True)

    if data_changed:
        data_py.write_text(data_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
