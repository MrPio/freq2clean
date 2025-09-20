import sys
from pathlib import Path
from time import time_ns
from skimage.exposure import match_histograms

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

# %% Params
FRAMES = 600
AVG = 1_024
ALPHAS = [1]
SSIM3D_STEP = 4
METRICS_PATH = Path("fft_syntethic_metrics.csv")
suffx = f"frame{FRAMES}_avg{AVG}_alphas{'-'.join(map(str,ALPHAS))}"
cprint(f"RUN --> {suffx}", f"[{print_mem()}]", f"[{eta()}s]")

# %% Init
cprint("red:Loading Dataset...", f"[{print_mem()}]", f"[{eta()}s]")
ds_dir = DATASETS["synthetic"].dir
x_path = ds_dir / "noise_1Q_-5.52dBSNR_490x490x6000.tif"
y_path = ds_dir / "deepcad_E_10_test.tif"
gt_path = ds_dir / "clean_30Hz_490x490x6000.tif"
x, y, gt = (Recording(_, max_frames=FRAMES) for _ in [x_path, y_path, gt_path])
RES_DIR = FILE_DIR / "results/synthetic/"
RES_DIR.mkdir(exist_ok=True)

# %% FFT
cprint("green:Computing FFT(X)...", f"[{print_mem()}]", f"[{eta()}s]")
fft_x = np.fft.fft(x.np, axis=0)
fft_abs_x = np.abs(fft_x)
fft_angle_x = np.angle(fft_x, deg=True)
del fft_x

cprint("green:Computing FFT(y)...", f"[{print_mem()}]", f"[{eta()}s]")
fft_y = np.fft.fft(y.np, axis=0)
fft_abs_y = np.abs(fft_y)
fft_angle_y = np.angle(fft_y, deg=True)
del fft_y

# %% Freq0 as Averaged Frame
cprint("blue:Computing Freq0...", f"[{print_mem()}]", f"[{eta()}s]")
x_mean = np.mean(Recording(x_path, max_frames=AVG).np, axis=0)
x_mean = match_histograms(x_mean, fft_abs_x[0])
alpha = 1.0

# %% Video Fusion
cprint("blue:Computing Video Fusion...", f"[{print_mem()}]", f"[{eta()}s]")
fft_abs = fft_abs_y
fft_abs[0] = ALPHAS[0] * x_mean + (1 - ALPHAS[0]) * fft_abs_y[0]
for i, a in enumerate(ALPHAS[1:]):
    fft_abs[i + 1] = a * fft_abs_x[i + 1] + (1 - a) * fft_abs_y[i + 1]

X = fft_abs * np.exp(1j * np.deg2rad(fft_angle_y))
cprint("blue:Computing IFFT...", f"[{print_mem()}]", f"[{eta()}s]")
fused = np.fft.ifft(X, axis=0).real
del X, fft_abs_x, fft_angle_x, fft_abs_y, fft_angle_y

# %% Metrics
cprint("yellow:Saving results...", f"[{print_mem()}]", f"[{eta()}s]")
# np.save(RES_DIR / f"fused_{suffx}.npy", fused)

if METRICS_PATH.exists():
    df = pd.read_csv(METRICS_PATH, index_col="suffx")
else:
    df = pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")

cprint("yellow:Computing PSNR3D...", f"[{print_mem()}]", f"[{eta()}s]")
psnr_ = psnr3d(gt, fused, data_range=1_520)  # 1_520 is the 99.9% Quantile of GT

cprint("yellow:Computing SSIM3D...", f"[{print_mem()}]", f"[{eta()}s]")
ssim_ = ssim3d(gt.np[::SSIM3D_STEP], fused[::SSIM3D_STEP])

df.loc[suffx] = [psnr_, ssim_]
df.to_csv(METRICS_PATH)
cprint("PSNR3D=", psnr_, "SSIM3D=", ssim_, f"[{print_mem()}]", f"[{eta()}s]")
