import sys

sys.path.append("..")
from src import Path, np, pd, clog, cprint

# Args
dataset = sys.argv[1]
denoisers = [
    "gt",
    "deepcad_15",
    "fft_15",
    "deepcad_30",
    "fft_30",
    "deepcad_75",
    "fft_75",
    "deepcad_150",
    "fft_150",
    "deepcad_300",
    "fft_300",
]
model = "cpsam"  # ["cpsam", "cyto3", "nuclei"]

# Init
METRICS_PATH = Path(f"cellpose_metrics.csv")


def create_mask(stat_item, ly, lx):
    """Converts a suite2p ROI 'stat' item into a 2D boolean mask."""
    mask = np.zeros((ly, lx), dtype=bool)
    mask[stat_item["ypix"], stat_item["xpix"]] = True
    return mask


def iou(mask1, mask2):
    """Calculates the Intersection over Union (IoU) for two boolean masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return 0.0
    return intersection / union


def compare(test: str, gt: str = "gt"):
    """
    Compares two suite2p segmentations and computes performance metrics.

    Args:
        gt_dir (str): Path to the ground truth suite2p output directory.
        test_dir (str): Path to the test (e.g., denoised) suite2p output directory.
        iou_threshold (float): The IoU threshold to consider a match a true positive.
    """
    SUFFX = f"{dataset}_{test}"
    test_path = Path(f"cellpose_results/{dataset}/{test}/{model}_mask.npy")
    gt_path = Path(f"cellpose_results/{dataset}/{gt}/{model}_mask.npy")

    df = (
        pd.read_csv(METRICS_PATH, index_col="suffx")
        if METRICS_PATH.exists()
        else pd.DataFrame(columns=["suffx", "ROI GT", "ROIs", "IoU"]).set_index("suffx")
    )
    mask_test = np.load(test_path)
    mask_gt = np.load(gt_path)
    n_gt = mask_gt.max()
    n_test = mask_test.max()
    mask_test = mask_test.astype(bool)
    mask_gt = mask_gt.astype(bool)
    cprint(f"Found", n_gt, "ROIs in Ground Truth and", n_test, "ROIs in Test.")

    tot_iou = np.mean([iou(mask_test[i], mask_gt[i]) for i in range(mask_gt.shape[0])])
    clog(f"IoU=", f"blue:{tot_iou:.3f}")
    df.loc[SUFFX] = [n_gt, n_test, tot_iou]
    df.to_csv(METRICS_PATH)


for denoiser in denoisers:
    clog("Processing", f"red:{denoiser}")
    compare(denoiser)
