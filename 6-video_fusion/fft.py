import sys
from pathlib import Path
from skimage.exposure import match_histograms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

#%% Params
FRAMES = 600
AVG = None

#%% Init
ds_dir=DATASETS["synthetic"].dir
x_path = ds_dir / "noise_1Q_-5.52dBSNR_490x490x6000.tif"
y_path = ds_dir / "deepcad_E_10_test.tif"
gt_path = ds_dir / "clean_30Hz_490x490x6000.tif"
x, y, gt = (Recording(_, max_frames=FRAMES) for _ in [x_path, y_path, gt_path])
RES_DIR = Path("results/synthetic/")
RES_DIR.mkdir(exist_ok=True)

#%% FFT
fft_x = np.fft.fft(x.np, axis=0)
fft_abs_x = np.abs(fft_x)
fft_angle_x = np.angle(fft_x, deg=True)
del fft_x

fft_y = np.fft.fft(y.np, axis=0)
fft_abs_y = np.abs(fft_y)
fft_angle_y = np.angle(fft_y, deg=True)
del fft_y

dt = 1 / 30  # Synthetic is 30Hz
freqs = np.fft.fftfreq(x.np.shape[0], d=dt)
rfft_abs_x = fft_abs_x[freqs >= 0]
rfft_abs_y = fft_abs_y[freqs >= 0]
rfft_angle_x = fft_angle_x[freqs >= 0]
rfft_angle_y = fft_angle_y[freqs >= 0]

#%% Freq0 as Averaged Frame
x_mean = np.mean(Recording(x_path, max_frames=AVG).np, axis=0)
# Rescale according to rfft_abs_x[0] distribution
x_mean = (x_mean - np.min(x_mean)) / (np.max(x_mean) - np.min(x_mean))
x_mean = (x_mean * (np.max(rfft_abs_x[0]) - np.min(rfft_abs_x[0]))) + np.min(rfft_abs_x[0])

x_mean = match_histograms(x_mean, rfft_abs_x[0])