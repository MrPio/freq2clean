import sys
import tifffile as tiff
from pathlib import Path
from tqdm import trange

for i in trange(1, len(sys.argv)):
    file = Path(sys.argv[i])
    tiff.imwrite(
        file.with_name(f"{file.stem}_f32.tiff"),
        tiff.imread(file).astype("float32"),
    )
