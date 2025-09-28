import sys
from pathlib import Path

sys.path.append("..")
from src import imshow, cprint, tqdm, Recording

dataset_dir = Path("dataset/Denoising/")
out_dir = dataset_dir / "previews"
out_dir.mkdir(exist_ok=True)
subdir = "Training"

datasets = [_ for _ in dataset_dir.glob("*") if _.is_dir()]
cprint("The available datasets are:", *[f"rand:{_.stem}" for _ in datasets], sep="\n")

for dataset in tqdm(datasets):
    cprint(f"Processing", f"cyan:{dataset}")
    x_dir, gt_dir = (dataset / subdir / _ for _ in ["Raw", "GT"])
    xs, gts = (sorted(_.glob("*")) for _ in [x_dir, gt_dir])
    for x, gt in tqdm(zip(xs, gts)):
        try:
            imshow(
                {
                    "Noisy": Recording(x, max_frames=1).np,
                    "Ground Truth": Recording(gt, max_frames=1).np,
                },
                size=12,
                path=out_dir / f"{dataset.stem}-{x.stem}.png",
            )
        except:
            continue
