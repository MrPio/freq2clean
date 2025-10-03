import sys

sys.path.append("..")
from src import *

out_dir = mkdir("avgs")
for dataset, y_path in tqdm(
    [
        # (
        #     "synthetic",
        #     "results/DataFolderIs_synthetic_202509211503_ModelFolderIs_synthetic_202509211433/E_10_Iter_1248/xf_E_10_Iter_1248_output.tif",
        # ),
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
    metrics_path = out_dir / f"{dataset}.csv"
    df = pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")
    meta = DATASETS[dataset]
    clog("Loading x, y, gt...")
    x, y, gt = (Recording(_, max_frames=None).normalized for _ in [meta.x, y_path, meta.gt])
    clog("Averaging x...")
    x = Recording(x).avg_fast(2048)

    clog("Computing metrics...")
    df.loc["2048"] = [psnr(x, gt), ssim3d(x, gt, step=4)]
    df.loc["DeepCad"] = [psnr(y, gt[: y.shape[0]]), ssim3d(y, gt[: y.shape[0]], step=4)]
    df.to_csv(metrics_path)
