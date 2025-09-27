import sys
import suite2p

sys.path.append("..")
from src import DATASETS, clog, Recording

# Args
dataset = "synthetic"
folder = "fft"
max_frames = 500

# Init
meta = DATASETS[dataset]
n_time, Ly, Lx = meta.shape
ops = suite2p.default_ops()
ops["fs"] = meta.hz
ops["tau"] = 1.25
ops["soma_crop"] = False
db = {"data_path": [c]}

if max_frames:
    clog(f"Cutting TIFF to {max_frames} frames...")
    Recording(db["data_path"][0] + "/data.tiff", max_frames=max_frames).save(db["data_path"][0] + "/data.tiff")

clog("light_red:Running pipeline...")
output_ops = suite2p.run_s2p(ops=ops, db=db)
