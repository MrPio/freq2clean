# 4. Segmentation — Validating Freq2Clean downstream

This section runs segmentation networks on the **cleaned** videos and compares the masks against those obtained from the ground truth. The goal is to show that Freq2Clean segmentations match the ground-truth segmentations more closely than the raw denoiser does, without distorting temporal dynamics.

## 📂 Expected data layout

Each network reads `dataset/<dataset>/<folder>/data.tiff`. Put one `data.tiff` per version you want to compare:

```
dataset/synthetic/
├── gt/data.tiff           # clean ground truth
├── x/data.tiff            # noisy input
├── deepcad_15/data.tiff   # denoiser baseline
└── fft_15/data.tiff       # Freq2Clean output (fft1d)   <-- from Section 3
```

`fft_*` = Freq2Clean output, `deepcad_*` = denoiser baseline; the suffix is the averaging window. Edit the `folders` list at the top of each `*_run.py` to match the versions you created.

## 🧫 The three segmentation networks

All three share the same interface — pass the dataset name as the only argument:

```sh
python cellpose_run.py  synthetic
python stardist_run.py  synthetic
python cellsam_run.py   synthetic
```

For every folder they segment `num_frames` evenly-spaced frames and save the masks (`*_mask.npy`) plus a preview PNG under `<net>_results/<dataset>/<folder>/`.

| Network   | Script             | Note                                                         |
| --------- | ------------------ | ------------------------------------------------------------ |
| Cellpose  | `cellpose_run.py`  | Uses the `cpsam` pretrained model.                           |
| StarDist  | `stardist_run.py`  | Uses `2D_versatile_fluo`.                                    |
| CellSAM   | `cellsam_run.py`   | Loads credentials from a `.env` (DeepCell access token).     |

## 📈 Analysis

- **Suite2p** (`suite2p_run.py`) is the calcium-imaging ROI/signal pipeline used by the cross-contamination and temporal-dynamics analyses (it is configured in-script, not via a CLI argument).
- Notebooks visualize and quantify the results:
  - `4.1.x` — visualize masks per network (Suite2p / Cellpose / StarDist / CellSAM).
  - `4.2.x` — analyze ROI cross-contamination (Cellpose, Suite2p).
  - `4.3` — temporal-dynamics preservation.
  - `4.4` — compare segmentation metrics across versions.
