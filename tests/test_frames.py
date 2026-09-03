"""Tests for core.frames (video → frames)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from slam_to_mesh.core.frames import extract_frames, is_video_file


def test_is_video_file():
    assert is_video_file("a.mp4") is True
    assert is_video_file("a.MOV") is True
    assert is_video_file("a.png") is False
    assert is_video_file("a.zip") is False


def _make_test_video(path: Path, seconds: int = 3, rate: int = 15) -> bool:
    """Create a synthetic test video with ffmpeg; return False if unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    r = subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={seconds}:size=320x240:rate={rate}",
            str(path),
        ],
        capture_output=True, check=False,
    )
    return r.returncode == 0 and path.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_frames_count(tmp_path: Path):
    vid = tmp_path / "v.mp4"
    assert _make_test_video(vid)
    out = tmp_path / "frames"
    n = extract_frames(vid, out, n=15)
    assert n > 0
    assert n <= 15
    files = sorted(out.glob("frame_*.png"))
    assert len(files) == n


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_frames_creates_dir(tmp_path: Path):
    vid = tmp_path / "v.mp4"
    assert _make_test_video(vid, seconds=2)
    out = tmp_path / "nested" / "frames"
    n = extract_frames(vid, out, n=8)
    assert out.is_dir()
    assert n > 0
