# 3. Freq2Clean — Training and inference

Freq2Clean fuses, in the frequency domain, the **temporally averaged** noisy video (high spatial SNR) with a **denoiser** output (fast transients). This section trains the fusion module and runs it.

## ✅ Prerequisites

In the dataset folder you need three files:

```
dataset/zenodo/synthetic/
├── x.tif            # noisy            (Section 1)
├── gt.tif           # clean target     (Section 1)
└── deepcad.tiff     # denoiser output  (Section 2)
```

## 📍 How data location is specified

You never hard-code paths. Instead:

- **`dataset_name`** — a key from `DATASETS` ([`src/dataset/dataset.py`](../src/dataset/dataset.py)); it resolves the folder above and auto-downloads `x.tif`/`gt.tif` if missing.
- **`denoiser_name` + `denoiser_variant`** — select the denoised file as `<denoiser_name><denoiser_variant>.tiff` inside that folder (e.g. `deepcad` + `_15` → `deepcad_15.tiff`).

## 🔀 Frequency transform: `fft1d` (preferred) vs `dct3d`

Set `frequency_transform` in the config:

- **`dft1d`** *(preferred)* — a 1-D DFT along time, per pixel. Fast, few parameters, robust. Use large `patch_t`.
- **`dct3d`** *(optional)* — a 3-D DCT over space+time patches. Many more parameters, so use a **small `patch_t`** to keep it tractable.

## 🏋️ Training

Edit [`train_config.json`](train_config.json), then run:

```sh
python freq2clean_train.py
```

Key fields:

```jsonc
{
  "dataset_name": "synthetic",      // key in DATASETS
  "denoiser_name": "deepcad",       // <name> of the .tiff baseline
  "denoiser_variant": "_15",        // suffix, "" if none
  "frequency_transform": "dft1d",   // "dft1d" (preferred) or "dct3d"
  "patch_t": 3000,                  // use a SMALL value for dct3d
  "patch_xy": 64,
  "avg_win": 1024                   // temporal-averaging window
}
```

Outputs go to `trainings/<timestamp>-<dataset>_<denoiser>/`, containing `cfg.json`, checkpoints under `pth/`, the learned mask, and loss plots.

## 🚀 Inference

```sh
python freq2clean_test.py \
  --checkpoint <trainings-subfolder> \   # e.g. dft1d  (folder under trainings/)
  --dataset    synthetic \               # key in DATASETS
  --denoiser   deepcad \                 # <name> of the baseline .tiff
  --variant    _15 \                     # optional suffix (default "")
  --batch_size 1
```

This reports PSNR3D / SSIM3D and writes snapshots + `metrics.json` under `results/<dataset>/<denoiser>/`. To also export the cleaned video as a `.tiff`, set `SAVE_TIFF = True` near the top of [`freq2clean_test.py`](freq2clean_test.py).

> The frequency transform is read from the checkpoint's `cfg.json`, so it always matches what you trained.

## 🔬 Exploring the fusion (optional)

[`A-frequency_fusion/`](A-frequency_fusion) contains notebooks that explore DFT1D / DCT3D / wavelet fusion and run a grid search over the fusion coefficients, used as a non-learned reference for the trained masks.
