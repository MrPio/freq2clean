from pathlib import Path
from PIL import Image
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array
from tqdm import tqdm
from IPython.display import Video

""" 
The `Editor` class holds the basic video processing operations used to compare multiple videos.

Note: Ensure you have Magik installed.
"""


class Editor:
    def __init__(self):
        pass

    def compose(
        self,
        array: list[dict[str, Path | str]],
        output: Path | str,
        fontsize=28,
        bitrate=4500,
        font="SourceCodeVF-Black",
    ):
        """Create a composed video from a grid of videos.
        Args:
            font (str): You can list installed fonts with `convert -list font`
        """
        clips = []
        for videos_row in array:
            row = []
            if isinstance(videos_row, list):
                videos_row = {Path(_).stem: _ for _ in videos_row}
            for title, video in videos_row.items():
                clip = VideoFileClip(video)
                text = TextClip(title, fontsize=fontsize, color="white", font=font)
                text = text.set_position(("center", "top")).set_duration(clip.duration)
                row.append(CompositeVideoClip([clip, text]))
            clips.append(row)
        clips_array(clips).write_videofile(output, bitrate=f"{bitrate}k")
        return Video(output)
