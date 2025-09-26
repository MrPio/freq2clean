import sys
import suite2p

sys.path.append("..")
from src import DATASETS, clog

# Args
dataset = "synthetic"
folder = "fft_15"

# Init
meta = DATASETS[dataset]
n_time, Ly, Lx = meta.shape
ops = suite2p.default_ops()
# ops['threshold_scaling'] = 2.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
ops["fs"] = meta.hz  # sampling rate of recording, determines binning for cell detection
ops["tau"] = 1.25  # timescale of gcamp to use for deconvolution
db = {"data_path": [f"dataset/{dataset}/{folder}"]}

clog("light_red:Running pipeline...")
output_ops = suite2p.run_s2p(ops=ops, db=db)
