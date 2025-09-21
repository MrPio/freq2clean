import sys
from pathlib import Path
from time import time_ns
from skimage.exposure import match_histograms
from tqdm import trange
import cupy as cp

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

deepcad_suffx = "15-150-2"
METRICS_PATH = Path(f"fft_syntethic_metrics_patcht{deepcad_suffx}.csv")

# Init
cprint("red:Loading Dataset...", f"[{print_mem()}]", f"[{elapsed()}s]")
ds_dir = DATASETS["synthetic"].dir
x_path = ds_dir / "noise_1Q_-5.52dBSNR_490x490x6000.tif"

# y_path = ds_dir / "deepcad_E_10_test_patcht_30_test_150.tif"
y_path = "/leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/2-denoise/results/DataFolderIs_synthetic_202509211422_ModelFolderIs_synthetic_202509211352/E_10_Iter_1200/xf_E_10_Iter_1200_output.tif"

gt_path = ds_dir / "clean_30Hz_490x490x6000.tif"
x, y, gt = (Recording(_, max_frames=None) for _ in [x_path, y_path, gt_path])
RES_DIR = FILE_DIR / "results/synthetic/"
RES_DIR.mkdir(exist_ok=True)

# Freq0 as Averaged Frame
cprint("blue:Computing Freq0...", f"[{print_mem()}]", f"[{elapsed()}s]")
x_mean = np.mean(x.np, axis=0)


def test(frames, alphas, ssim3d_step=4, save=False):
    suffx = f"frame{frames}_alphas{'-'.join(map(str,alphas))}"
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
        df.loc["deepcad"] = [0, 0]

    def fft_fusion(vox):
        # FFT
        fft = np.fft.fft(vox, axis=0)
        fft_abs = np.abs(fft)
        fft_angle = np.angle(fft, deg=True)
        del fft

        # Video Fusion
        global x_mean
        freq0 = match_histograms(x_mean, fft_abs[0])
        fft_abs[0] = alphas[0] * freq0 + (1 - alphas[0]) * fft_abs[0]
        X = fft_abs * np.exp(1j * np.deg2rad(fft_angle))
        fused = np.fft.ifft(X, axis=0).real
        return fused

    def fft_fusion_optimized(vox, eps=1e-12):
        X = np.fft.rfft(vox, axis=0)
        mag = np.abs(X)

        global x_mean
        freq0 = match_histograms(x_mean, mag[0])
        mag0_new = alphas[0] * freq0 + (1.0 - alphas[0]) * mag[0]
        mag0 = mag[0]

        zero_mask = mag0 < eps
        nonzero_mask = ~zero_mask
        if nonzero_mask.any():
            scale = mag0_new[nonzero_mask] / mag0[nonzero_mask]
            X[0][nonzero_mask] *= scale
        if zero_mask.any():
            # set new complex values with zero phase
            X0 = X[0]
            X0[zero_mask] = mag0_new[zero_mask].astype(X.dtype)
            X[0] = X0
        fused = np.fft.irfft(X, n=vox.shape[0], axis=0)
        return fused

    def fft_fusion_gpu(
        vox: np.ndarray,
        eps: float = 1e-12
    ) -> cp.ndarray:
        vox_gpu=cp.asarray(vox)
        x_mean_gpu=cp.asarray(x_mean)
        
        X = cp.fft.rfft(vox_gpu, axis=0)
        mag = cp.abs(X)
        mag0 = mag[0]

        freq0_matched = match_histograms(x_mean_gpu, mag0)
        mag0_new = alphas[0] * freq0_matched + (1.0 - alphas[0]) * mag0

        zero_mask = mag0 < eps
        nonzero_mask = ~zero_mask
        if cp.any(nonzero_mask):
            scale = mag0_new[nonzero_mask] / mag0[nonzero_mask]
            X[0, nonzero_mask] *= scale
        if cp.any(zero_mask):
            X[0, zero_mask] = mag0_new[zero_mask].astype(X.dtype)
        return cp.fft.irfft(X, n=vox_gpu.shape[0], axis=0).get()
    
    
    fused = np.empty_like(y.np)
    for i in trange(x.frames // frames, desc="FFT fusion...", colour="cyan"):
        start = i * frames
        end = start + frames
        fused[start:end] = fft_fusion_gpu(y.np[start:end])
    # Metrics
    if save:
        cprint("yellow:Saving results...", f"[{print_mem()}]", f"[{elapsed()}s]")
        np.save(RES_DIR / f"ftt_fused_{suffx}_{deepcad_suffx}.npy", fused)

    cprint("yellow:Computing PSNR3D...", f"[{print_mem()}]", f"[{elapsed()}s]")
    psnr_ = psnr3d(gt.np[:end], fused[:end], data_range=1_520)  # 1_520 is the 99.9% Quantile of GT
    cprint("yellow:Computing SSIM3D...", f"[{print_mem()}]", f"[{elapsed()}s]")
    ssim_ = ssim3d(gt.np[:end:ssim3d_step], fused[:end:ssim3d_step])

    df.loc[suffx] = [psnr_, ssim_]
    df.to_csv(METRICS_PATH)
    cprint("\tPSNR3D=", f"cyan:{psnr_:.2f}", "SSIM3D=", f"cyan:{ssim_:.2f}", f"[{print_mem()}]", f"[{elapsed()}s]")


# Alpha test
# FRAMES = 1_000
# for alpha in tqdm([0.1 * i for i in range(1, 11)]):
#     test(frames=FRAMES, alphas=[alpha])

# Alphas test
# for n, s in tqdm([(n, s) for n in [1, 2, 3, 4, 5] for s in [1, 0.75, 0.5, 0.25]]):
#     test(alphas=[ALPHA] + (np.linspace(1, 0, n) * s).tolist())

# Frames test
# ALPHAS = [ALPHA]
#     for frames in tqdm([20, 50, 100, 300, 600, 1200, 3000, 6000]):
#     test(frames=frames, alphas=ALPHAS)

# BEST
test(frames=3_000, alphas=[0.85], ssim3d_step=4, save=True)
