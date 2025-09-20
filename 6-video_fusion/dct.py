import sys
from pathlib import Path
from skimage.exposure import match_histograms
from tqdm import trange
from scipy.fftpack import dctn, idctn

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

METRICS_PATH = Path("dct_syntethic_metrics.csv")
MAX_FRAMES = None

# Init
cprint("red:Loading Dataset...", f"[{print_mem()}]", f"[{elapsed()}s]")
ds_dir = DATASETS["synthetic"].dir
x_path = ds_dir / "noise_1Q_-5.52dBSNR_490x490x6000.tif"
y_path = ds_dir / "deepcad_E_10_test.tif"
gt_path = ds_dir / "clean_30Hz_490x490x6000.tif"
x, y, gt = (Recording(_, max_frames=MAX_FRAMES) for _ in [x_path, y_path, gt_path])
RES_DIR = FILE_DIR / "results/synthetic/"
RES_DIR.mkdir(exist_ok=True)
means = {}


def test(frames, win, s0, δs, t0, δt, ssim3d_step=4, save=False):
    suffx = f"frame{frames}_win{win}_s0{s0}_δs{δs}_t0{t0}_δt{δt}"
    cprint(f"magenta:RUN --> {suffx}", f"[{print_mem()}]", f"[{elapsed()}s]")

    df = (
        pd.read_csv(METRICS_PATH, index_col="suffx")
        if METRICS_PATH.exists()
        else pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")
    )
    if "deepcad" not in df.index:
        # cprint("red:Initializing metrics...", f"[{print_mem()}]", f"[{elapsed()}s]")
        # psnr_ = psnr3d(gt, y, data_range=1_520)  # 1_520 is the 99.9% Quantile of GT
        # ssim_ = ssim3d(gt.np[::ssim3d_step], y.np[::ssim3d_step])
        # df.loc["deepcad"] = [psnr_, ssim_]
        # df.to_csv(METRICS_PATH)
        # cprint(
        #     "\tDeepCAD --> PSNR3D=",
        #     f"cyan:{psnr_:.2f}",
        #     "SSIM3D=",
        #     f"cyan:{ssim_:.2f}",
        #     f"[{print_mem()}]",
        #     f"[{elapsed()}s]",
        # )
        df.loc["deepcad"] = [32.30381747535838, 0.5577632784843445]

    # Mask
    T, H, W = frames, *x.np.shape[1:]
    w_t = np.arange(T)[:, None, None]
    w_y = np.arange(H)[None, :, None]
    w_x = np.arange(W)[None, None, :]
    w_r = np.sqrt(w_y**2 + w_x**2)
    w_s = np.clip((w_r - (s0 - δs)) / (2 * δs), 0, 1)  # grows from 0→1 around s0
    w_t = np.clip((w_t - (t0 - δt)) / (2 * δt), 0, 1)  # grows from 0→1 around t0
    W = w_s * (1 - w_t)

    if win not in means:
        cprint("blue:Computing Avg...", f"[{print_mem()}]", f"[{elapsed()}s]")
        means[win] = x.avg_fast(win)

    def dct_fusion(vox_x, vox_y):
        dct_x = dctn(vox_x, type=2, norm="ortho")
        dct_y = dctn(vox_y, type=2, norm="ortho")
        merged = W * dct_x + (1 - W) * dct_y
        return idctn(merged, norm="ortho")

    fused = np.empty_like(y.np)
    for i in trange(x.frames // frames, desc="DCT fusion...", colour="cyan"):
        start = i * frames
        end = start + frames
        fused[start:end] = dct_fusion(means[win][start:end], y.np[start:end])

    # Metrics
    if save:
        cprint("yellow:Saving results...", f"[{print_mem()}]", f"[{elapsed()}s]")
        np.save(RES_DIR / f"dct_fused_{suffx}.npy", fused)

    cprint("yellow:Computing PSNR3D...", f"[{print_mem()}]", f"[{elapsed()}s]")
    psnr_ = psnr3d(gt.np[:end], fused[:end], data_range=1_520)  # 1_520 is the 99.9% Quantile of GT
    cprint("yellow:Computing SSIM3D...", f"[{print_mem()}]", f"[{elapsed()}s]")
    ssim_ = ssim3d(gt.np[:end:ssim3d_step], fused[:end:ssim3d_step])

    df.loc[suffx] = [psnr_, ssim_]
    df.to_csv(METRICS_PATH)
    cprint("\tPSNR3D=", f"cyan:{psnr_:.2f}", "SSIM3D=", f"cyan:{ssim_:.2f}", f"[{print_mem()}]", f"[{elapsed()}s]")


# Weights
# for s0, δs, t0, δt in tqdm(
#     [
#         (8, 24, 0, 64),
#         (16, 64, 0, 32),
#         (48, 128, -6, 16),
#         (64, 196, -6, 8),

#         (64, 128, -6, 16),
#         (128, 64, -6, 16),
#         (64, 128, -6, 32),
#         (128, 64, -6, 32),

#         (48, 96, -6, 16),
#         (36, 72, -6, 16),
#         (36, 72, -6, 12),
#         (36, 72, -6, 8),
#         (48, 96, 0, 4),
#     ]
# ):
#     test(frames=300, win=512, s0=s0, δs=δs, t0=t0, δt=δt)

# Frames
# s0, δs, t0, δt = (48, 128, -6, 16)
# for frames in tqdm([6000,1000]):
#     test(frames=frames, win=512, s0=s0, δs=δs, t0=t0, δt=δt)

# Avgs
# s0, δs, t0, δt = (48, 128, -6, 16)
# frames = 600
# for win in tqdm([1, 4, 16, 64, 256, 1024, 2048, 4096, 6000]):
#     test(frames=frames, win=win, s0=s0, δs=δs, t0=t0, δt=δt)


s0, δs, t0, δt = (36, 72, -6, 16)
frames = 3000
win = 6000
test(frames=frames, win=win, s0=s0, δs=δs, t0=t0, δt=δt, save=True)
