import functools
from pathlib import Path
from typing import List, Tuple
import numpy as np
import sys
from moviepy.editor import VideoFileClip
from PIL import Image

input_path = Path(str(sys.argv[1]))
output_path = Path(input_path).parent / f"{Path(input_path).stem}_green{Path(input_path).suffix}"
BITRATE = 10_000


GREEN_GRADIENT = (
    (0, (0, 0, 0)),
    ((int(sys.argv[2]) if len(sys.argv) > 2 else 166), (98, 255, 67)),
    (255, (255, 255, 255)),
)

# Generating LUT
x_coords = np.array([s[0] for s in GREEN_GRADIENT])
colors = np.array([s[1] for s in GREEN_GRADIENT], dtype=np.float32)
lut_inputs = np.arange(256)  # The 256 possible input values
output_channels = []
for i in range(3):
    interpolated_channel = np.interp(lut_inputs, x_coords, colors[:, i])
    output_channels.append(interpolated_channel)
lut = np.stack(output_channels, axis=-1)
lut = np.clip(lut, 0, 255).astype(np.uint8)


def gradient(frame: np.ndarray) -> np.ndarray:
    return lut[frame[:, :, 0]]


if input_path.suffix in [".jpg", ".png"]:
    img = np.array(Image.open(input_path).convert("RGB"))
    img = gradient(img)
    Image.fromarray(img).save(output_path)
else:
    clip = VideoFileClip(str(input_path))
    green_clip = clip.fl_image(gradient)
    green_clip.write_videofile(
        str(output_path),
        codec="libx265",
        bitrate=f"{BITRATE}k",
        verbose=False,
        logger="bar",
    )
