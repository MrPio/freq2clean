import sys
from typing import Any

sys.path.append("..")
from src import *

# Args
max_frames = 600
dataset = "synthetic"
variant = 150
segment_net = "cellpose"

# Init
meta = DATASETS[dataset]

clog(f"blue:Loading recordings...")
x = Recording(meta.x, max_frames=max_frames).np
y_dcad = Recording(f"dataset/{dataset}/deepcad_{variant}/data.tiff", max_frames=max_frames).np
y_f2c = Recording(f"dataset/{dataset}/fft_{variant}/data.tiff", max_frames=max_frames).np
gt = Recording(meta.gt, max_frames=max_frames).np

clog(f"blue:Loading masks...")
mask_dcad = np.load(f"{segment_net}_results/{dataset}/deepcad_{variant}/cpsam_mask.npy")
mask_f2c = np.load(f"{segment_net}_results/{dataset}/fft_{variant}/cpsam_mask.npy")
mask_gt = np.load(f"{segment_net}_results/{dataset}/gt/cpsam_mask.npy")


class ROI:
    def __init__(self, label, coords: tuple[np.array, np.array], mask):
        self.label = int(label)
        self.coords = np.vstack(coords).T
        self.ys, self.xs = coords
        self.centroid = (self.ys.mean(), self.xs.mean())
        self.area = self.ys.size
        self.mask = mask


# Funs
def mask2rois(mask: np.ndarray) -> list[ROI]:
    rois = []
    H, W = mask.shape
    for label in np.unique(mask):
        ys, xs = np.nonzero(mask == label)
        if ys.size == 0 or label == 0:
            continue
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[ys, xs] = True
        rois.append(ROI(label, (ys, xs), roi_mask))
    return rois


def get_traces(vid: np.ndarray, rois: list[ROI]) -> np.ndarray:
    T, H, W = vid.shape
    traces = np.zeros((len(rois), T), dtype=np.float32)
    for i, r in enumerate(rois):
        traces[i] = vid[:, r.ys, r.xs].mean(axis=(1, 2))
    return traces


# Run analysis
rois_dcad = mask2rois(mask_dcad[-1])
rois_f2c = mask2rois(mask_f2c[-1])

traces_dcad = get_traces(y_dcad, rois_dcad)
traces_f2c = get_traces(y_f2c, rois_f2c)

print(traces_dcad)
print(traces_dcad.shape)

clog(f"ROI={rois_dcad}")
# out_deep = analyze_pairwise_contamination(movie_deep, rois_deep, pixel_size_um=1.0, do_neuropil=False, max_dist_um=200)
# out_freq = analyze_pairwise_contamination(movie_freq, rois_freq, pixel_size_um=1.0, do_neuropil=False, max_dist_um=200)
# print('Demo shapes: corr matrices', out_deep['corr'].shape, out_freq['corr'].shape)
# print('Demo auc50 deep, freq', out_deep['auc_50um'], out_freq['auc_50um'])
# plot_median_vs_distance(out_deep['summary'], out_freq['summary'])
