#!/usr/bin/env python3
"""
LangBuster v0.1 — Video Decoder Module
Streaming decode for mp4, webm, gif → Frame sequences.
Reuses existing libraries; no codec implementation.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, List, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, GifImagePlugin
except ImportError:
    Image = None


@dataclass
class VideoMetadata:
    """Immutable metadata about source video."""
    path: str
    sha256: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    format: str
    codec: str
    has_audio: bool = False


@dataclass
class Frame:
    """Single decoded frame as spatial state (FLOW-compatible)."""
    index: int                    # 0-based frame index
    timestamp: float              # seconds from start
    rgb: np.ndarray               # (H, W, 3) uint8
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.rgb.shape

    @property
    def w(self) -> int:
        return self.rgb.shape[1]

    @property
    def h(self) -> int:
        return self.rgb.shape[0]

    def to_flow_channels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to FLOW R,G,B channels (float32 R,G ; uint8 B)."""
        arr = self.rgb.astype(np.float32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = self.rgb[:, :, 2].copy().astype(np.uint8)
        return r, g, b


class VideoDecoder:
    """
    Streaming video decoder with deterministic frame access.
    Supports: mp4, webm, gif (via OpenCV + PIL fallback).
    """

    def __init__(self, video_path: str | Path):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._metadata: Optional[VideoMetadata] = None
        self._cap = None
        self._gif_frames: Optional[List[Frame]] = None
        self._opened = False

    def _compute_sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def _open_cv2(self):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) required for mp4/webm. Install: pip install opencv-python")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {self.path}")
        return cap

    def _open_gif(self):
        if Image is None:
            raise RuntimeError("PIL required for GIF. Install: pip install pillow")
        img = Image.open(self.path)
        if not getattr(img, "is_animated", False):
            raise ValueError(f"Not an animated GIF: {self.path}")
        return img

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is not None:
            return self._metadata

        sha = self._compute_sha256()
        suffix = self.path.suffix.lower()

        if suffix == ".gif":
            gif = self._open_gif()
            w, h = gif.size
            frame_count = getattr(gif, "n_frames", 1)
            # GIF duration per frame in ms
            durations = []
            try:
                for i in range(frame_count):
                    gif.seek(i)
                    durations.append(gif.info.get("duration", 100))
            except EOFError:
                pass
            avg_duration = sum(durations) / len(durations) if durations else 100
            fps = 1000.0 / avg_duration if avg_duration > 0 else 10.0
            duration_sec = frame_count / fps

            self._metadata = VideoMetadata(
                path=str(self.path),
                sha256=sha,
                width=w,
                height=h,
                fps=fps,
                frame_count=frame_count,
                duration_sec=duration_sec,
                format="gif",
                codec="gif",
            )
        else:
            cap = self._open_cv2()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0.0
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
            cap.release()

            self._metadata = VideoMetadata(
                path=str(self.path),
                sha256=sha,
                width=w,
                height=h,
                fps=fps,
                frame_count=frame_count,
                duration_sec=duration_sec,
                format=suffix.lstrip("."),
                codec=codec or "unknown",
            )
        return self._metadata

    def frames(self) -> Iterator[Frame]:
        """Stream frames sequentially (generator)."""
        meta = self.metadata
        suffix = self.path.suffix.lower()

        if suffix == ".gif":
            gif = self._open_gif()
            for i in range(meta.frame_count):
                gif.seek(i)
                frame_rgb = np.array(gif.convert("RGB"), dtype=np.uint8)
                ts = i / meta.fps
                yield Frame(index=i, timestamp=ts, rgb=frame_rgb)
        else:
            cap = self._open_cv2()
            for i in range(meta.frame_count):
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts = i / meta.fps
                yield Frame(index=i, timestamp=ts, rgb=frame_rgb)
            cap.release()

    def frame_at(self, index: int) -> Frame:
        """Random access to specific frame (decodes up to index)."""
        if index < 0:
            raise IndexError("Frame index must be >= 0")
        meta = self.metadata
        if index >= meta.frame_count:
            raise IndexError(f"Frame {index} >= frame_count {meta.frame_count}")

        suffix = self.path.suffix.lower()
        if suffix == ".gif":
            gif = self._open_gif()
            gif.seek(index)
            frame_rgb = np.array(gif.convert("RGB"), dtype=np.uint8)
            ts = index / meta.fps
            return Frame(index=index, timestamp=ts, rgb=frame_rgb)
        else:
            cap = self._open_cv2()
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Failed to decode frame {index}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ts = index / meta.fps
            return Frame(index=index, timestamp=ts, rgb=frame_rgb)

    def frame_range(self, start: int, end: int) -> Iterator[Frame]:
        """Efficient range access (streaming)."""
        if start < 0 or end > self.metadata.frame_count or start >= end:
            raise ValueError(f"Invalid range [{start}, {end}) for {self.metadata.frame_count} frames")
        for i, frame in enumerate(self.frames()):
            if i < start:
                continue
            if i >= end:
                break
            yield frame


def decode_video(video_path: str | Path) -> VideoDecoder:
    """Factory function."""
    return VideoDecoder(video_path)


# Convenience for CLI / quick inspection
def video_info(video_path: str | Path) -> dict:
    dec = VideoDecoder(video_path)
    m = dec.metadata
    return {
        "path": str(m.path),
        "sha256": m.sha256,
        "width": m.width,
        "height": m.height,
        "fps": m.fps,
        "frame_count": m.frame_count,
        "duration_sec": m.duration_sec,
        "format": m.format,
        "codec": m.codec,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m langbuster.video <video_path>")
        sys.exit(1)
    info = video_info(sys.argv[1])
    import json
    print(json.dumps(info, indent=2))