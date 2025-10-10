import sys
from skimage.exposure import match_histograms

sys.path.append("..")
from src import *

# Args
dataset = "oabf_astro"
fft_file = "ftt_oabf_astro_frame3000_alphas0.85deepcad_theirs.npy"
FRAMES = 3000
AVG_WIN = 2048
EXPORT_FRAMES = [0, 100, 1000, 2000]
OUT_DIR = mkdir("frames")

# Init
meta = DATASETS[dataset]
cprint(f"blue:Loading x,y...")
x, y = (Recording(_, max_frames=FRAMES) for _ in [meta.x, meta.dir / "y.tiff"])
cprint(f"blue:Loading fft...")
fft = Recording(np.load(f"results/{dataset}/{fft_file}"))
cprint(f"blue:Loading gt...")
gt = {frame: x.avg_frame(frame=frame, window=AVG_WIN) for frame in EXPORT_FRAMES}
cprint(f"blue:Matching hist...")
fft_matched = match_histograms(fft.np, y.np)

# Export
imshow(
    {
        k: v
        for frame in EXPORT_FRAMES
        for k, v in {
            f"Raw ({frame})": x.np[frame],
            f"DeepCAD ({frame})": y.np[frame],
            f"FFT-fused ({frame})": fft_matched[frame],
            f"Pseudo-GT ({frame})": gt[frame],
        }.items()
    },
    size=8,
    zoom=2,
    cols=4,
    path=OUT_DIR / f"{dataset}.png",
)
clog(f"green:Exported dataset {dataset} to {OUT_DIR.stem}/")
