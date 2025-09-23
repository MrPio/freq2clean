import sys
from pathlib import Path
from time import time_ns
from typing import Literal
from skimage.exposure import match_histograms
from tqdm import trange

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

denoiser_name: Literal["deepcad", "noise2noise", "noise2void"] = "deepcad"
denoiser_suffx = "300-150 norm"
dataset = "synthetic"
y_path = "../bkp/2-denoise/results/DataFolderIs_synthetic_202509211545_ModelFolderIs_synthetic_202509211437/E_10_Iter_1296/xf_E_10_Iter_1296_output.tif"

# Init
METRICS_PATH = Path(f"fft_{dataset}_metrics_{denoiser_name}_{denoiser_suffx}.csv")
clog("red:Loading Dataset...")
metadata = DATASETS[dataset]
x, y, gt = (Recording(_, max_frames=None) for _ in [metadata.x, y_path, metadata.gt])
x.np = x.np[: y.frames, : y.np.shape[1], : y.np.shape[2]]
gt.np = gt.np[: y.frames, : y.np.shape[1], : y.np.shape[2]]
RES_DIR = FILE_DIR / f"results/{dataset}/"
RES_DIR.mkdir(exist_ok=True, parents=True)

# Freq0 as Averaged Frame
clog("blue:Computing Freq0...")
x_mean = np.mean(x.np, axis=0)


def test(frames, alphas, ssim3d_step=4, save=False):
    suffx = f"frame{frames}_alphas{'-'.join(map(str,alphas))}"
    clog(f"magenta:RUN --> {suffx}")

    df = (
        pd.read_csv(METRICS_PATH, index_col="suffx")
        if METRICS_PATH.exists()
        else pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")
    )
    if denoiser_name not in df.index:
        clog("red:Initializing metrics...")
        psnr_ = psnr3d(gt, y, data_range=metadata.data_range)
        ssim_ = ssim3d(
            Recording(gt.np[::ssim3d_step]).normalized,
            Recording(y.np[::ssim3d_step]).normalized,
        )
        df.loc[denoiser_name] = [psnr_, ssim_]
        df.to_csv(METRICS_PATH)
        clog(f"\t{denoiser_name} --> PSNR3D=", f"cyan:{psnr_:.2f}", "SSIM3D=", f"cyan:{ssim_:.2f}")
        # df.loc[denoiser_name] = [0, 0]

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

    def fft_fusion_gpu(vox: np.ndarray, eps=1e-12) -> np.ndarray:
        vox_gpu = cp.asarray(vox)
        x_mean_gpu = cp.asarray(x_mean)

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
        fused[start:end] = (fft_fusion_gpu if CUPY_AVAILABLE else fft_fusion_optimized)(y.np[start:end])

    # Metrics
    if save:
        clog("yellow:Saving results...")
        np.save(RES_DIR / f"ftt_{dataset}_{suffx}{denoiser_name}_{denoiser_suffx}.npy", fused)

    clog("yellow:Computing PSNR3D...")
    psnr_ = psnr3d(gt.np[:end], fused[:end], data_range=metadata.data_range)
    clog("yellow:Computing SSIM3D...")
    ssim_ = ssim3d(
        Recording(gt.np[:end:ssim3d_step]).normalized,
        Recording(fused[:end:ssim3d_step]).normalized,
    )

    df.loc[suffx] = [psnr_, ssim_]
    df.to_csv(METRICS_PATH)
    clog("\tPSNR3D=", f"cyan:{psnr_:.2f}", "SSIM3D=", f"cyan:{ssim_:.2f}")


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
