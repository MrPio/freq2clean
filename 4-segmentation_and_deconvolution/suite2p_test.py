import sys
from scipy.optimize import linear_sum_assignment

sys.path.append("..")
from src import Path, np, trange, pd, clog, cprint

# Args
dataset = "synthetic"
denoiser = "fft_15"
FAST_IOU=False

# Init
METRICS_PATH = Path(f"suite2p_matrics.csv")
SUFFX = f"{dataset}_metrics_{denoiser}"


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


def compare(test: str, gt: str = "gt", iou_threshold=0.5):
    """
    Compares two suite2p segmentations and computes performance metrics.

    Args:
        gt_dir (str): Path to the ground truth suite2p output directory.
        test_dir (str): Path to the test (e.g., denoised) suite2p output directory.
        iou_threshold (float): The IoU threshold to consider a match a true positive.
    """
    test_dir = Path(f"dataset/{dataset}/{test}/suite2p/plane0")
    gt_dir = Path(f"dataset/{dataset}/{gt}/suite2p/plane0")

    df = (
        pd.read_csv(METRICS_PATH, index_col="suffx")
        if METRICS_PATH.exists()
        else pd.DataFrame(
            columns=["suffx", "IoU Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1-Score", "Average IoU"]
        ).set_index("suffx")
    )

    clog("Load suite2p outputs...")
    stat_gt = np.load(gt_dir / "stat.npy", allow_pickle=True)
    ops_gt = np.load(gt_dir / "ops.npy", allow_pickle=True).item()
    stat_test = np.load(test_dir / "stat.npy", allow_pickle=True)
    ops_test = np.load(test_dir / "ops.npy", allow_pickle=True).item()

    clog("Ensure dimensions match...")
    ly, lx = ops_gt["Ly"], ops_gt["Lx"]
    assert ly == ops_test["Ly"] and lx == ops_test["Lx"]

    n_gt = len(stat_gt)
    n_test = len(stat_test)
    cprint(f"Found", n_gt, "ROIs in Ground Truth and", n_test, "ROIs in Test.")

    if FAST_IOU:
        gt_mask = np.logical_or.reduce([create_mask(_, ly, lx) for _ in stat_gt])
        test_mask = np.logical_or.reduce([create_mask(_, ly, lx) for _ in stat_test])
        tot_iou = iou(test_mask, gt_mask)
        clog(f"IoU={tot_iou:.3f}")
    else:
        clog("Building IoU matrix...")
        iou_matrix = np.zeros((n_gt, n_test))
        for i in trange(n_gt):
            mask_gt = create_mask(stat_gt[i], ly, lx)
            for j in range(n_test):
                mask_test = create_mask(stat_test[j], ly, lx)
                iou_matrix[i, j] = iou(mask_gt, mask_test)

        clog("Match ROIs using the Hungarian algorithm...")
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        clog("Computing metrics...")
        matched_ious = iou_matrix[row_ind, col_ind]
        true_positives = np.sum(matched_ious >= iou_threshold)
        false_negatives = n_gt - true_positives
        false_positives = n_test - true_positives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou_matched = np.mean(matched_ious[matched_ious >= iou_threshold]) if true_positives > 0 else 0.0

        # --- Print Results ---
        print("\n--- Segmentation Performance ---")
        print(f"IoU Threshold: {iou_threshold}")
        print("---------------------------------")
        print(f"True Positives (TP):  {true_positives}")
        print(f"False Positives (FP): {false_positives}")
        print(f"False Negatives (FN): {false_negatives}")
        print("---------------------------------")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1_score:.3f}")
        print(f"Average IoU of matched ROIs: {avg_iou_matched:.3f}")
        print("---------------------------------\n")
        df.loc[SUFFX] = [
            iou_threshold,
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1_score,
            avg_iou_matched,
        ]
        df.to_csv(METRICS_PATH)


compare(denoiser)
