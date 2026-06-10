# 1. EDA — Downloading and visualizing datasets

This section downloads the calcium-imaging datasets and visualizes their noisy/clean frames and SNR.

## 📦 Available datasets

All datasets are declared in [`src/dataset/dataset.py`](../src/dataset/dataset.py) inside the `DATASETS` dictionary. Each entry knows where to download the **noisy** recording (`x.tif`) and its **clean** ground truth (`gt.tif`).

| Key (`DATASETS[...]`)          | Notebook                                  | Source        |
| ------------------------------ | ----------------------------------------- | ------------- |
| `synthetic` ⭐                  | `1.1-Synthetic.ipynb`                     | NAOMi (Zenodo) |
| `zebrafish`                    | `1.2-Zebrafish.ipynb`                     | DeepCAD-RT    |
| `neutrophils`                  | `1.3-Neutrophils.ipynb`                   | DeepCAD-RT    |
| `mouse_neuronal_populations`   | `1.4-Mouse_Neuronal_Populations.ipynb`    | DeepCAD-RT    |
| `mouse_dendritic_spines`       | `1.5.1-Mouse_Dendritic_Spines_50mW.ipynb` | DeepCAD-RT    |
| `mouse_dendritic_spines_115mw` | `1.5.2-..._115mW.ipynb`                   | DeepCAD-RT    |

⭐ **`synthetic`** is the main dataset: it ships with a true clean ground truth and is the one used to train and benchmark Freq2Clean.

## ⬇️ How to download

Downloads are lazy: calling `download()` fetches the files only if they are missing, and saves them as `x.tif` (noisy) and `gt.tif` (clean) inside the dataset folder.

```python
import sys; sys.path.append("..")
from src import DATASETS

meta = DATASETS["synthetic"]   # pick any key from the table above
meta.download()                # -> dataset/zenodo/synthetic/{x.tif, gt.tif}
```

Running any of the `1.x` notebooks does this for you in the first cell and then plots sample frames, SNR, and metrics.

## ➕ Adding your own dataset

Add a new `DatasetMetadata` entry to `DATASETS` in [`src/dataset/dataset.py`](../src/dataset/dataset.py). If you already have the files locally, just place `x.tif` (and optionally `gt.tif`) in a folder and point `dir` to it — leave `urls=None` to skip downloading.
