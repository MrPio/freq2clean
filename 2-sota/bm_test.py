import sys
from pathlib import Path
import bm3d, bm4d
from PIL import Image
from csbdeep.utils import normalize

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

# Args
dataset = "synthetic"
frame = 100
patch_ts = [8, 16, 32, 64]
σs = [0.05, 0.1, 0.15, 0.2]
stage = bm3d.BM3DStages.ALL_STAGES

# Init
OUT_DIR = mkdir("bm_results")
METRICS_PATH = OUT_DIR / "metrics.csv"
clog("red:Loading Dataset...")
meta = DATASETS[dataset]
x, gt = (Recording(_, max_frames=frame + max(patch_ts)).normalized for _ in [meta.x, meta.gt])
df = (
    pd.read_csv(METRICS_PATH, index_col="suffx")
    if METRICS_PATH.exists()
    else pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")
)


def save(res):
    clog("yellow:Rendering and evaluating...")
    for suffx, img in res.items():
        if suffx != "gt":
            df.loc[suffx] = [p := psnr(img, res["gt"]), s := ssim(img, res["gt"])]
            df.to_csv(METRICS_PATH)
        Image.fromarray(np.clip(normalize(img, 0.25, 99.9), 0, 1) * 255).convert("RGB").save(OUT_DIR / f"{suffx}.png")


results = {"x": x[frame], "gt": gt[frame]}
clog("Running", "blue:BM3D")
for σ in tqdm(σs, desc="σ", leave=False):
    suffx = f"bm3d_{dataset}_frames{frame}_σ{σ}_stage{stage}"
    if suffx in df.index:
        continue
    results[suffx] = bm3d.bm3d(
        z=x[frame],
        sigma_psd=σ,
        stage_arg=stage,
    )
    save(results)

clog("Running", "blue:BM4D")
for patch_t in tqdm(patch_ts, desc="patch_t"):
    for σ in tqdm(σs, desc="σ", leave=False):
        suffx = f"bm4d_{dataset}_frames{frame}_t{patch_t}_σ{σ}_stage{stage}"
        if suffx in df.index:
            continue
        results[suffx] = bm4d.bm4d(
            z=x[frame - patch_t // 2 : frame + patch_t // 2],
            sigma_psd=σ,
            stage_arg=stage,
        )[patch_t // 2]
        save(results)
