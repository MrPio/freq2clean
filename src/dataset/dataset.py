import torch
from torchvision import transforms
import random
from pathlib import Path
from tqdm import tqdm
from ..utils import download, cprint

_ROOT_DIR = Path(__file__).parents[2]


class DatasetMetadata:
    def __init__(
        self,
        dir,
        shape=None,
        data_range=2**16 - 1,  # uint16
        x="x.tif",
        gt: str | None = "gt.tif",
        hz=30,
        urls: tuple[str, str] | None = None,
    ):
        self.dir = _ROOT_DIR / dir
        self.shape = shape
        self.data_range = data_range
        self.x = self.dir / x
        self.labeled = gt != None
        self.hz = hz
        self.urls = urls
        if gt:
            self.gt = self.dir / gt

    def download(self):
        if not self.urls:
            raise Exception("No download URLs available for this dataset!")

        paths = (self.x, self.gt if hasattr(self, "gt") else None)
        for path, url in (pbar := tqdm(zip(paths, self.urls), leave=False)):
            # Lazy
            if path and url:
                if path.exists():
                    cprint(f"green:{path.stem}", "already exists!")
                else:
                    path.parent.mkdir(exist_ok=True, parents=True)
                    pbar.set_description(f"Downloading [{path.stem}]...")
                    download(url, path.resolve())


DATASETS = {
    "oabf_astro": DatasetMetadata(
        dir="dataset/oabf/astro",
        data_range=14_207,
        gt=None,
        hz=7,
    ),
    "oabf_vpm": DatasetMetadata(
        dir="dataset/oabf/vpm",
        gt=None,
    ),
    "oabf_resonant_neuro": DatasetMetadata(
        dir="dataset/oabf/resonant_neuro",
        gt=None,
    ),
    "synthetic": DatasetMetadata(
        dir="dataset/zenodo/synthetic",
        shape=(6000, 490, 490),
        data_range=1_520,
        hz=30,
        urls=(
            "https://zenodo.org/records/6254739/files/noise_1Q_-5.52dBSNR_490x490x6000.tif?download=1",
            "https://zenodo.org/records/6254739/files/clean_30Hz_490x490x6000.tif?download=1",
        ),
    ),
    "zebrafish": DatasetMetadata(
        dir="dataset/zenodo/zebrafish",
        shape=(9800, 400, 485),
        data_range=32_767,
        hz=15,
        urls=(
            "https://zenodo.org/records/6293696/files/01_ZebrafishMul_GCaMP6s_485x400x9800_lowSNR.tif?download=1",
            "https://zenodo.org/records/6293696/files/01_ZebrafishMul_GCaMP6s_485x400x9800_highSNR.tif?download=1",
        ),
    ),
    "neutrophils": DatasetMetadata(
        dir="dataset/zenodo/neutrophils",
        data_range=49_978,
        urls=(
            "https://zenodo.org/records/6296569/files/02_neutrophil_0.465umPerPixel_512x512x5706_lowSNR.tif?download=1",
            "https://zenodo.org/records/6296569/files/02_neutrophil_0.465umPerPixel_512x512x5706_highSNR.tif?download=1",
        ),
    ),
    "mouse_neuronal_populations": DatasetMetadata(
        dir="dataset/zenodo/mouse_neuronal_populations_5",
        shape=(6500, 490, 490),
        data_range=47_939,
        hz=30,
        urls=(
            "https://zenodo.org/records/6299096/files/05_MouseNeurons_GCaMP6f_100umdepth_50mWpower_30Hz_lowSNR_MCRound1.tif?download=1",
            "https://zenodo.org/records/6299096/files/05_MouseNeurons_GCaMP6f_100umdepth_50mWpower_30Hz_highSNR_MCRound1.tif?download=1",
        ),
    ),
    "mouse_dendritic_spines": DatasetMetadata(
        dir="dataset/zenodo/mouse_dendritic_spines",
        shape=(6500, 492, 978),
        data_range=30_666,
        hz=30,
        urls=(
            "https://zenodo.org/records/6275571/files/1_spine_GCaMP6f_50mWpower_978x492x6500_lowSNR.tif?download=1",
            "https://zenodo.org/records/6275571/files/1_spine_GCaMP6f_50mWpower_978x492x6500_highSNR.tif?download=1",
        ),
    ),
    "mouse_dendritic_spines_115mw": DatasetMetadata(
        dir="dataset/zenodo/mouse_dendritic_spines_115mw",
        shape=(6500, 432, 944),
        data_range=62_523,
        hz=30,
        urls=(
            "https://zenodo.org/records/6275571/files/2_spine_GCaMP6f_115mWpower_944x432x6500_lowSNR.tif?download=1",
            "https://zenodo.org/records/6275571/files/2_spine_GCaMP6f_115mWpower_944x432x6500_highSNR.tif?download=1",
        ),
    ),
}


def random_transform(input, target):
    """
    (From DeepCAD-RT)
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target (C,W,H) : the input and target patch before data augmentation
    Return:
        input, target (C,W,H) : the input and target patch after data augmentation
    """
    p_trans = random.randrange(0, 8)
    if p_trans == 0:
        pass
    elif p_trans == 1:  # left rotate 90
        input = torch.rot90(input, k=1, dims=(1, 2))
        target = torch.rot90(target, k=1, dims=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = torch.rot90(input, k=2, dims=(1, 2))
        target = torch.rot90(target, k=2, dims=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = torch.rot90(input, k=3, dims=(1, 2))
        target = torch.rot90(target, k=3, dims=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = torch.flip(input, dims=(2,))
        target = torch.flip(target, dims=(2,))
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = torch.flip(input, dims=(2,))
        input = torch.rot90(input, k=1, dims=(1, 2))
        target = torch.flip(target, dims=(2,))
        target = torch.rot90(target, k=1, dims=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = torch.flip(input, dims=(2,))
        input = torch.rot90(input, k=2, dims=(1, 2))
        target = torch.flip(target, dims=(2,))
        target = torch.rot90(target, k=2, dims=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = torch.flip(input, dims=(2,))
        input = torch.rot90(input, k=3, dims=(1, 2))
        target = torch.flip(target, dims=(2,))
        target = torch.rot90(target, k=3, dims=(1, 2))
    return input, target


def npy2tensor(max_value):
    return transforms.Compose(
        [
            # numpy uint16 H×W -> torch.Tensor 1×H×W [0,1]
            transforms.Lambda(lambda x: torch.from_numpy(x).float() / float(max_value)),
            transforms.Lambda(lambda x: torch.clip(x, min=0, max=1)),
            transforms.Lambda(lambda t: t.unsqueeze(0) if t.ndim == 2 else t),
            # normalize  [0,1] -> [–1,1]
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )
