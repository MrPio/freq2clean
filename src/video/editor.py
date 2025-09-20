from pathlib import Path
from PIL import Image
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array, concatenate_videoclips
import moviepy.video.fx.all as vfx
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
        codec="libx265",
        duration=None,
        zoom=None,
        speed=None,
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
                clip = VideoFileClip(str(video))
                if speed:
                    clip = vfx.speedx(clip, factor=speed)
                if duration:
                    clip = clip.set_duration(duration)
                if zoom:
                    w, h = clip.size
                    new_w = int(w / zoom)
                    new_h = int(h / zoom)
                    clip = vfx.crop(clip, width=new_w, height=new_h, x_center=w / 2, y_center=h / 2)
                text = TextClip(title, fontsize=fontsize / (zoom**0.5), color="white", font=font)
                text = text.set_position(("center", "top")).set_duration(duration if duration else clip.duration)
                row.append(CompositeVideoClip([clip, text]))
            clips.append(row)
        clips_array(clips).write_videofile(str(output), bitrate=f"{bitrate}k", codec=codec)
        return Video(output)

    def alternate(
        self,
        videos: dict[str : Path | str] | list[Path | str],
        output: Path | str,
        fontsize=26,
        bitrate=4500,
        font="SourceCodeVF-Black",
        delta=1.0,
        codec="libx265",
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
        concatenate_videoclips(sequence, method="compose").write_videofile(
            str(output), bitrate=f"{bitrate}k", codec=codec
        )
        return Video(output)
