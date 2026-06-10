# 2. SOTA denoisers — Producing the baseline

Freq2Clean runs *after* a denoiser. This section denoises the noisy recording (`x.tif`) downloaded in [Section 1](../1-eda) and produces the baseline that Freq2Clean fuses with the temporal average.

The recommended denoiser is **DeepCAD-RT**. Other denoisers are optional and follow the same idea.

## 🎯 The output Freq2Clean expects

Whatever denoiser you use, save its denoised video as a TIFF **next to** `x.tif` and `gt.tif`, named `<denoiser><variant>.tiff`:

```
dataset/zenodo/synthetic/
├── x.tif              # noisy   (from Section 1)
├── gt.tif             # clean   (from Section 1)
└── deepcad.tiff       # denoised baseline  <-- produced here
```

The `<variant>` is an optional suffix (e.g. `deepcad_15.tiff`) to keep several denoised versions of the same recording (different hyper-parameters). Section 3 selects the file via `--denoiser deepcad --variant _15`.

---

## 🧠 DeepCAD-RT (recommended)

### 1. Place the training data

DeepCAD trains on a folder of noisy TIFF stacks under `2-sota/dataset/<name>/`. Create that folder and link (or copy) the noisy recording into it:

```sh
mkdir -p dataset/synthetic && cd dataset/synthetic
ln -s ../../../dataset/zenodo/synthetic/x.tif x.tif
cd ../..
```

### 2. Train

Edit the config at the top of [`deepcad_train.py`](deepcad_train.py) — set `datasets_path` to the folder above and adjust `n_epochs`, `patch_*`, etc. Then:

```sh
python deepcad_train.py          # or: sbatch deepcad_train.sh  (SLURM)
```

Checkpoints are written to `pth/<dataset>_<timestamp>/`.

### 3. Inference

Edit [`deepcad_test.py`](deepcad_test.py): set `datasets_path` and `denoise_model` to the trained folder name from step 2. Then:

```sh
python deepcad_test.py
```

The denoised stack is written under `results/`. Move/rename it next to `x.tif`/`gt.tif` as `deepcad.tiff` (see *output* section above) so Freq2Clean can pick it up.

---

## 🔁 Other denoisers (optional)

Each notebook/script denoises `x.tif` and can produce a `<denoiser>.tiff` baseline the same way. Run them only if you want to compare baselines.

| Denoiser            | How to run                                             |
| ------------------- | ------------------------------------------------------ |
| Temporal averaging  | `2.1-Temporal_Averaging.ipynb`                         |
| BM3D / BM4D         | `python bm_test.py` (notebook `2.2`)                   |
| K-SVD               | `2.3-K_SVD.ipynb`                                       |
| Noise2Void          | `python noise2void_train.py` → `python noise2void_test.py` |
| Noise2Noise         | `python noise2noise_train.py` → `python noise2noise_test.py` |
| TeD                 | `TeD/` — see [`TeD/README.md`](TeD/README.md)          |
| SRDTrans            | `SRDTrans/` — see [`srd_test.sh`](srd_test.sh)         |
| FAST                | `FAST/train.py` / `FAST/test.py`                       |
| DeepVIDv2           | `DeepVIDv2/scripts/train.py` / `inference.py`          |

> The configurable parameters (dataset, frames, patch size) live at the top of each script or in the first notebook cell.
