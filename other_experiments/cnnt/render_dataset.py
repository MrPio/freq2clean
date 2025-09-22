import sys
from pathlib import Path

sys.path.append(str(Path("..").resolve()))
from src import imshow, cprint, tqdm, Recording, np

dataset_dir = Path("dataset/Denoising/")
out_dir = dataset_dir / "previews"
out_dir.mkdir(exist_ok=True)
subdir = "Training"

datasets = [_ for _ in dataset_dir.glob("*") if _.is_dir()]
cprint("The available datasets are:", *[f"rand:{_.stem}" for _ in datasets], sep="\n")
dataset = "Lysosome"

cprint(f"Processing", f"cyan:{dataset}")
x_dir, gt_dir = (dataset_dir / dataset / subdir / _ for _ in ["Raw", "GT"])
xs, gts = (sorted(_.glob("*.tif")) for _ in [x_dir, gt_dir])

gt_full = np.concatenate([Recording(_, max_frames=None).np for _ in tqdm(gts[:6]]), axis=0)
Recording(gt_full).render(f"{dataset}_gt.mp4", codec="libx264")
