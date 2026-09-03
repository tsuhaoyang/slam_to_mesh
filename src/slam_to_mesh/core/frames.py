"""Video → frames for the photogrammetry input path.

Extracts evenly-spaced frames from a video so they can be treated as a
multi-view image set (see ``docs/spec_photogrammetry.md``). Prefers system
**ffmpeg** (fast, robust); falls back to imageio if ffmpeg is unavailable.

Video frames are typically lower quality than deliberate photos (motion blur,
compression), so capture should be slow and well-lit — documented for users.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _video_duration_seconds(video: Path) -> float | None:
    """Probe duration via ffprobe, if available."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video),
            ],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError):
        return None


def extract_frames(video: str | Path, out_dir: str | Path, n: int = 40) -> int:
    """Extract ~*n* evenly-spaced frames from *video* into *out_dir*.

    Writes ``frame_00001.png`` … Returns the number of frames written. Uses
    ffmpeg's ``fps`` filter computed from duration when known, else a frame-count
    approach; falls back to imageio.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, int(n))

    ffmpeg = _ffmpeg_bin()
    if ffmpeg is not None:
        count = _extract_ffmpeg(ffmpeg, video, out_dir, n)
        if count > 0:
            return count
    return _extract_imageio(video, out_dir, n)


def _extract_ffmpeg(ffmpeg: str, video: Path, out_dir: Path, n: int) -> int:
    """Evenly sample n frames with ffmpeg. Returns frames written."""
    duration = _video_duration_seconds(video)
    pattern = str(out_dir / "frame_%05d.png")
    if duration and duration > 0:
        # fps that yields ~n frames over the whole clip.
        fps = max(n / duration, 1e-3)
        vf = f"fps={fps}"
    else:
        # Unknown duration: grab up to n frames at a modest fps.
        vf = "fps=2"
    cmd = [
        ffmpeg, "-y", "-i", str(video),
        "-vf", vf, "-vsync", "vfr",
        "-frames:v", str(n),
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    frames = sorted(out_dir.glob("frame_*.png"))
    if proc.returncode != 0 and not frames:
        return 0
    return len(frames)


def _extract_imageio(video: Path, out_dir: Path, n: int) -> int:
    """Fallback frame extraction via imageio (optional dependency)."""
    try:
        import imageio.v3 as iio
    except Exception as e:
        raise RuntimeError(
            "no frame extractor available: install ffmpeg or imageio"
        ) from e

    frames = iio.imread(str(video), index=None)  # (T, H, W, C)
    total = len(frames)
    if total == 0:
        return 0
    step = max(1, total // n)
    written = 0
    for i in range(0, total, step):
        if written >= n:
            break
        iio.imwrite(str(out_dir / f"frame_{written + 1:05d}.png"), frames[i])
        written += 1
    return written
