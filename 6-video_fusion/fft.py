import sys
from pathlib import Path
from time import time_ns
from skimage.exposure import match_histograms

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

METRICS_PATH = Path("fft_syntethic_metrics.csv")


def test(frames=600, avg=6000, alphas=[1], ssim3d_step=4):
    suffx = f"frame{frames}_avg{avg}_alphas{'-'.join(map(str,alphas))}"
    cprint(f"magenta:RUN --> {suffx}", f"[{print_mem()}]", f"[{elapsed()}s]")

    # Init
    cprint("red:Loading Dataset...", f"[{print_mem()}]", f"[{elapsed()}s]")
    ds_dir = DATASETS["synthetic"].dir
    x_path = ds_dir / "noise_1Q_-5.52dBSNR_490x490x6000.tif"
    y_path = ds_dir / "deepcad_E_10_test.tif"
    gt_path = ds_dir / "clean_30Hz_490x490x6000.tif"
    x, y, gt = (Recording(_, max_frames=frames) for _ in [x_path, y_path, gt_path])
    RES_DIR = FILE_DIR / "results/synthetic/"
    RES_DIR.mkdir(exist_ok=True)

    df = (
        pd.read_csv(METRICS_PATH, index_col="suffx")
        if METRICS_PATH.exists()
        else pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")
    )
    if "deepcad" not in df.index:
        cprint("red:Initializing metrics...", f"[{print_mem()}]", f"[{elapsed()}s]")
        psnr_ = psnr3d(gt, y, data_range=1_520)  # 1_520 is the 99.9% Quantile of GT
        ssim_ = ssim3d(gt.np[::ssim3d_step], y.np[::ssim3d_step])
        df.loc["deepcad"] = [psnr_, ssim_]
        df.to_csv(METRICS_PATH)
        cprint(
            "\tDeepCAD --> PSNR3D=",
            f"cyan:{psnr_:.2f}",
            "SSIM3D=",
            f"cyan:{ssim_:.2f}",
            f"[{print_mem()}]",
            f"[{elapsed()}s]",
        )

    # FFT
    cprint("green:Computing FFT(X)...", f"[{print_mem()}]", f"[{elapsed()}s]")
    fft_x = np.fft.fft(x.np, axis=0)
    fft_abs_x = np.abs(fft_x)
    fft_angle_x = np.angle(fft_x, deg=True)
    del fft_x

    cprint("green:Computing FFT(y)...", f"[{print_mem()}]", f"[{elapsed()}s]")
    fft_y = np.fft.fft(y.np, axis=0)
    fft_abs_y = np.abs(fft_y)
    fft_angle_y = np.angle(fft_y, deg=True)
    del fft_y

    # Freq0 as Averaged Frame
    cprint("blue:Computing Freq0...", f"[{print_mem()}]", f"[{elapsed()}s]")
    x_mean = np.mean(Recording(x_path, max_frames=avg).np, axis=0)
    x_mean = match_histograms(x_mean, fft_abs_x[0])
    alpha = 1.0

    # Video Fusion
    cprint("blue:Computing Video Fusion...", f"[{print_mem()}]", f"[{elapsed()}s]")
    fft_abs = fft_abs_y
    fft_abs[0] = alphas[0] * x_mean + (1 - alphas[0]) * fft_abs_y[0]
    for i, a in enumerate(alphas[1:]):
        fft_abs[i + 1] = a * fft_abs_x[i + 1] + (1 - a) * fft_abs_y[i + 1]

    X = fft_abs * np.exp(1j * np.deg2rad(fft_angle_y))
    cprint("blue:Computing IFFT...", f"[{print_mem()}]", f"[{elapsed()}s]")
    fused = np.fft.ifft(X, axis=0).real
    del X, fft_abs_x, fft_angle_x, fft_abs_y, fft_angle_y

    # Metrics
    cprint("yellow:Saving results...", f"[{print_mem()}]", f"[{elapsed()}s]")
    # np.save(RES_DIR / f"fused_{suffx}.npy", fused)

    cprint("yellow:Computing PSNR3D...", f"[{print_mem()}]", f"[{elapsed()}s]")
    psnr_ = psnr3d(gt, fused, data_range=1_520)  # 1_520 is the 99.9% Quantile of GT

    cprint("yellow:Computing SSIM3D...", f"[{print_mem()}]", f"[{elapsed()}s]")
    ssim_ = ssim3d(gt.np[::ssim3d_step], fused[::ssim3d_step])

    df.loc[suffx] = [psnr_, ssim_]
    df.to_csv(METRICS_PATH)
    cprint("\tPSNR3D=", f"cyan:{psnr_:.2f}", "SSIM3D=", f"cyan:{ssim_:.2f}", f"[{print_mem()}]", f"[{elapsed()}s]")


# for alpha in tqdm([0.8, 0.6, 0.4, 0.2]):
#     test(alphas=[alpha])

ALPHA = 0.65
# for n, s in tqdm([(n, s) for n in [1, 2, 3, 4, 5] for s in [1, 0.75, 0.5, 0.25]]):
#     test(alphas=[ALPHA] + (np.linspace(1, 0, n) * s).tolist())

ALPHAS = [ALPHA]
# for avg in tqdm([64, 256, 1024, 4096, 6000]):
#     test(avg=avg, alphas=ALPHAS)

AVG = 6000
for frames in tqdm([100, 300, 1200, 2400, 4800, 6000]):
    test(frames=frames, avg=AVG, alphas=ALPHAS)
