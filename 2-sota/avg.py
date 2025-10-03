import sys
from PIL import Image
from csbdeep.utils import normalize

sys.path.append("..")
from src import *

out_dir = mkdir("avgs")
for dataset, y_path in tqdm(
    [
        (
            "synthetic",
            "results/DataFolderIs_synthetic_202509211503_ModelFolderIs_synthetic_202509211433/E_10_Iter_1248/xf_E_10_Iter_1248_output.tif",
        ),
        (
            "zebrafish",
            "results/DataFolderIs_zebrafish_202509211737_ModelFolderIs_zebrafish_202509211717/E_10_Iter_1812/xf_E_10_Iter_1812_output.tif",
        ),
        (
            "neutrophils",
            "results/DataFolderIs_neutrophils_202509211945_ModelFolderIs_neutrophils_202509211933/E_10_Iter_1200/xf_E_10_Iter_1200_output.tif",
        ),
    ]
):
    meta = DATASETS[dataset]
    x, y, gt = (Recording(_, max_frames=None).np for _ in [meta.x, y_path, meta.gt])
    norm = lambda arr: np.clip(normalize(arr, 0, 99.9), 0, 1) * 255
    Image.fromarray(norm(np.mean(x, axis=0))).convert("RGB").save(out_dir / f"{dataset}_x.png")
    Image.fromarray(norm(gt[gt.shape[0] // 2])).convert("RGB").save(out_dir / f"{dataset}_gt.png")
    Image.fromarray(norm(y[y.shape[0] // 2])).convert("RGB").save(out_dir / f"{dataset}_y.png")
