from pathlib import Path
from PIL import Image
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array, concatenate_videoclips
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
        fontsize=26,
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

    def alternate(
        self,
        videos: dict[str : Path | str] | list[Path | str],
        output: Path | str,
        fontsize=26,
        bitrate=4500,
        font="SourceCodeVF-Black",
        delta=1.0,
    ):
        """
        Create a video alternating every second between videos.
        """
        if isinstance(videos, list):
            videos = {Path(_).stem: _ for _ in videos}
        clips = [VideoFileClip(str(p)) for _, p in videos.items()]
        titles = [t for t, _ in videos.items()]
        duration = min(map(lambda _: _.duration, clips))
        sequence = []
        for i in range(int(duration / delta)):
            idx = i % len(clips)
            base_clip = clips[idx].subclip(i, i + delta)
            title = (
                TextClip(titles[idx], fontsize=fontsize, color="white", font=font)
                .set_position(("center", "top"))
                .set_duration(delta)
            )
            sequence.append(CompositeVideoClip([base_clip, title]))
        concatenate_videoclips(sequence, method="compose").write_videofile(str(output), bitrate=f"{bitrate}k")
        return Video(output)
