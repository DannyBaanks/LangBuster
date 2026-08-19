#!/usr/bin/env python3
"""
LangBuster v0.1 — Temporal Module
Spatiotemporal state model + temporal semantics (DELTA, MOTION, PERSIST, etc.).
Extends FLOW's spatial computation to SPACE × TIME.
"""

from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator, Callable
from enum import IntEnum
import numpy as np

from .video import VideoDecoder, Frame, VideoMetadata


# ============================================================
# Temporal Instruction Set (extends FLOW's B-channel instructions)
# ============================================================

class TemporalOp(IntEnum):
    """Temporal operations encoded in extended instruction space."""
    # Frame navigation
    FRAME = 128        # Load frame t into working state
    NEXT  = 129        # Advance to frame t+1
    PREV  = 130        # Retreat to frame t-1
    JUMP  = 131        # Jump to absolute frame index

    # Temporal observation
    DELTA = 132        # Pixel difference: current - previous
    MOTION = 133       # Optical flow / motion vectors between frames
    TRACE  = 134       # Particle trajectory across frames

    # Temporal predicates
    PERSIST = 135      # Region/pixel persists across N frames
    APPEAR  = 136      # Region/pixel appears (was 0, now non-zero)
    DISAPPEAR = 137    # Region/pixel disappears (was non-zero, now 0)
    ENTER   = 138      # Particle enters region
    EXIT    = 139      # Particle exits region

    # Temporal control flow
    LOOP    = 140      # Loop over frame range
    WHILE   = 141      # While temporal condition holds
    ON_CHANGE = 142    # Trigger when change detected
    ON_EVENT  = 143    # Trigger on temporal event

    # Spatiotemporal queries
    REGION_T = 144     # Region at time t
    REGION_TT = 145    # Region across t0..t1
    PIXEL_T  = 146     # Pixel at (x,y,t)
    PIXEL_TT = 147     # Pixel trajectory (x,y,t0..t1)

    # Output / evidence
    EMIT    = 148      # Emit event to trace
    SNAPSHOT = 149     # Snapshot current spatiotemporal state


# Mnemonics for readability
TEMPORAL_MNEMONIC = {
    TemporalOp.FRAME: "FRAME",
    TemporalOp.NEXT: "NEXT",
    TemporalOp.PREV: "PREV",
    TemporalOp.JUMP: "JUMP",
    TemporalOp.DELTA: "DELTA",
    TemporalOp.MOTION: "MOTION",
    TemporalOp.TRACE: "TRACE",
    TemporalOp.PERSIST: "PERSIST",
    TemporalOp.APPEAR: "APPEAR",
    TemporalOp.DISAPPEAR: "DISAPPEAR",
    TemporalOp.ENTER: "ENTER",
    TemporalOp.EXIT: "EXIT",
    TemporalOp.LOOP: "LOOP",
    TemporalOp.WHILE: "WHILE",
    TemporalOp.ON_CHANGE: "ON_CHANGE",
    TemporalOp.ON_EVENT: "ON_EVENT",
    TemporalOp.REGION_T: "REGION_T",
    TemporalOp.REGION_TT: "REGION_TT",
    TemporalOp.PIXEL_T: "PIXEL_T",
    TemporalOp.PIXEL_TT: "PIXEL_TT",
    TemporalOp.EMIT: "EMIT",
    TemporalOp.SNAPSHOT: "SNAPSHOT",
}


# ============================================================
# Spatiotemporal State
# ============================================================

@dataclass
class FrameState:
    """
    Single frame as computational state (FLOW-compatible).
    Contains vector field (R,G) + scalar field (B) + trace layer.
    """
    frame: Frame
    field_r: np.ndarray        # (H, W) float32 - vx
    field_g: np.ndarray        # (H, W) float32 - vy
    field_b: np.ndarray        # (H, W) uint8 - mutable scalar/instructions
    field_trace: np.ndarray    # (H, W) uint8 - trace layer
    particles: List[Any] = field(default_factory=list)  # Particle refs

    @classmethod
    def from_frame(cls, frame: Frame, trace_enabled: bool = True) -> "FrameState":
        r, g, b = frame.to_flow_channels()
        return cls(
            frame=frame,
            field_r=r,
            field_g=g,
            field_b=b,
            field_trace=np.zeros((frame.h, frame.w), dtype=np.uint8) if trace_enabled else None,
        )

    def sample_vector(self, x: float, y: float) -> Tuple[float, float]:
        """Bilinear sample of vector field at continuous position."""
        h, w = self.field_r.shape
        x = max(0.0, min(float(w - 1), x))
        y = max(0.0, min(float(h - 1), y))
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        dx, dy = x - x0, y - y0

        def sample(field):
            v00 = field[y0, x0]
            v10 = field[y0, x1]
            v01 = field[y1, x0]
            v11 = field[y1, x1]
            return (v00 * (1 - dx) * (1 - dy) +
                    v10 * dx * (1 - dy) +
                    v01 * (1 - dx) * dy +
                    v11 * dx * dy)

        vx = (sample(self.field_r) / 127.5) - 1.0
        vy = (sample(self.field_g) / 127.5) - 1.0
        return vx, vy

    def sample_scalar(self, x: float, y: float) -> int:
        """Bilinear sample of scalar field (B channel)."""
        h, w = self.field_b.shape
        x = max(0.0, min(float(w - 1), x))
        y = max(0.0, min(float(h - 1), y))
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        dx, dy = x - x0, y - y0
        v00 = self.field_b[y0, x0]
        v10 = self.field_b[y0, x1]
        v01 = self.field_b[y1, x0]
        v11 = self.field_b[y1, x1]
        val = (v00 * (1 - dx) * (1 - dy) +
               v10 * dx * (1 - dy) +
               v01 * (1 - dx) * dy +
               v11 * dx * dy)
        return int(val + 0.5) & 0xFF


@dataclass
class VideoState:
    """
    Complete spatiotemporal state: video + current frame index + working memory.
    This is the core computational substrate of LangBuster.
    """
    decoder: VideoDecoder
    metadata: VideoMetadata
    frame_states: Dict[int, FrameState] = field(default_factory=dict)  # cached frames
    current_t: int = 0                    # current frame index
    previous_t: Optional[int] = None      # previous frame index
    working_memory: Dict[str, Any] = field(default_factory=dict)  # general purpose
    trace_log: List[Dict[str, Any]] = field(default_factory=list)  # temporal trace
    config: "LangBusterConfig" = None

    def __post_init__(self):
        if self.config is None:
            from .runtime import LangBusterConfig
            self.config = LangBusterConfig()

    @property
    def current(self) -> FrameState:
        """Get current frame state (loads if needed)."""
        return self.get_frame(self.current_t)

    @property
    def previous(self) -> Optional[FrameState]:
        """Get previous frame state."""
        if self.previous_t is None:
            return None
        return self.get_frame(self.previous_t)

    def get_frame(self, t: int) -> FrameState:
        """Get or create FrameState for frame index t."""
        if t < 0 or t >= self.metadata.frame_count:
            raise IndexError(f"Frame {t} out of range [0, {self.metadata.frame_count})")
        if t not in self.frame_states:
            frame = self.decoder.frame_at(t)
            self.frame_states[t] = FrameState.from_frame(frame, self.config.trace_enabled)
        return self.frame_states[t]

    def advance(self) -> bool:
        """Advance to next frame. Returns False if at end."""
        if self.current_t + 1 >= self.metadata.frame_count:
            return False
        self.previous_t = self.current_t
        self.current_t += 1
        return True

    def retreat(self) -> bool:
        """Retreat to previous frame."""
        if self.current_t - 1 < 0:
            return False
        self.current_t -= 1
        self.previous_t = self.current_t + 1 if self.current_t + 1 < self.metadata.frame_count else None
        return True

    def jump(self, t: int) -> bool:
        """Jump to absolute frame."""
        if t < 0 or t >= self.metadata.frame_count:
            return False
        self.previous_t = self.current_t
        self.current_t = t
        return True

    def delta(self, x: int, y: int) -> int:
        """Pixel delta: current - previous at (x,y)."""
        if self.previous is None:
            return 0
        cur = self.current.sample_scalar(x + 0.5, y + 0.5)
        prev = self.previous.sample_scalar(x + 0.5, y + 0.5)
        return (cur - prev) & 0xFF

    def delta_region(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Region delta as array."""
        if self.previous is None:
            return np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.int16)
        cur_b = self.current.field_b[y1:y2+1, x1:x2+1].astype(np.int16)
        prev_b = self.previous.field_b[y1:y2+1, x1:x2+1].astype(np.int16)
        return cur_b - prev_b

    def motion_vectors(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Compute motion vectors between previous and current (simple block matching)."""
        if self.previous is None:
            return np.zeros((y2 - y1 + 1, x2 - x1 + 1, 2), dtype=np.float32)
        # Simplified: use vector field difference as motion proxy
        cur_vx = self.current.field_r[y1:y2+1, x1:x2+1]
        cur_vy = self.current.field_g[y1:y2+1, x1:x2+1]
        prev_vx = self.previous.field_r[y1:y2+1, x1:x2+1]
        prev_vy = self.previous.field_g[y1:y2+1, x1:x2+1]
        motion = np.stack([cur_vx - prev_vx, cur_vy - prev_vy], axis=-1)
        return motion / 127.5  # normalize to [-1, 1]

    def persist_check(self, x1: int, y1: int, x2: int, y2: int, t0: int, t1: int, threshold: int = 10) -> bool:
        """Check if region persists (non-zero) across frame range [t0, t1]."""
        for t in range(t0, min(t1 + 1, self.metadata.frame_count)):
            fs = self.get_frame(t)
            region = fs.field_b[y1:y2+1, x1:x2+1]
            if np.count_nonzero(region) < threshold:
                return False
        return True

    def appear_check(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if region appears (was zero, now non-zero)."""
        if self.previous is None:
            return False
        prev_region = self.previous.field_b[y1:y2+1, x1:x2+1]
        cur_region = self.current.field_b[y1:y2+1, x1:x2+1]
        return np.all(prev_region == 0) and np.any(cur_region != 0)

    def disappear_check(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if region disappears (was non-zero, now zero)."""
        if self.previous is None:
            return False
        prev_region = self.previous.field_b[y1:y2+1, x1:x2+1]
        cur_region = self.current.field_b[y1:y2+1, x1:x2+1]
        return np.any(prev_region != 0) and np.all(cur_region == 0)

    def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit event to temporal trace."""
        self.trace_log.append({
            "t": self.current_t,
            "timestamp": self.current.frame.timestamp,
            "type": event_type,
            "payload": payload,
        })

    def snapshot(self) -> Dict[str, Any]:
        """Snapshot current spatiotemporal state."""
        return {
            "t": self.current_t,
            "timestamp": self.current.frame.timestamp,
            "field_b_hash": hashlib.sha256(self.current.field_b.tobytes()).hexdigest()[:16],
            "field_trace_hash": hashlib.sha256(self.current.field_trace.tobytes()).hexdigest()[:16] if self.current.field_trace is not None else None,
            "working_memory": dict(self.working_memory),
        }


# ============================================================
# Temporal Transition (state(t) → state(t+1))
# ============================================================

@dataclass
class TemporalTransition:
    """Represents a transition between two frame states."""
    t_from: int
    t_to: int
    delta_b: np.ndarray           # B field difference
    motion_field: np.ndarray      # (H, W, 2) motion vectors
    events: List[Dict[str, Any]]  # detected events

    @classmethod
    def compute(cls, state: VideoState, t_from: int, t_to: int) -> "TemporalTransition":
        fs_from = state.get_frame(t_from)
        fs_to = state.get_frame(t_to)

        delta_b = fs_to.field_b.astype(np.int16) - fs_from.field_b.astype(np.int16)
        motion = np.stack([fs_to.field_r - fs_from.field_r,
                           fs_to.field_g - fs_from.field_g], axis=-1) / 127.5

        events = []
        # Detect appeared regions
        appeared = (fs_from.field_b == 0) & (fs_to.field_b != 0)
        if np.any(appeared):
            ys, xs = np.where(appeared)
            for y, x in zip(ys[:10], xs[:10]):  # limit
                events.append({"type": "APPEAR", "x": int(x), "y": int(y), "value": int(fs_to.field_b[y, x])})

        # Detect disappeared regions
        disappeared = (fs_from.field_b != 0) & (fs_to.field_b == 0)
        if np.any(disappeared):
            ys, xs = np.where(disappeared)
            for y, x in zip(ys[:10], xs[:10]):
                events.append({"type": "DISAPPEAR", "x": int(x), "y": int(y), "old_value": int(fs_from.field_b[y, x])})

        return cls(t_from=t_from, t_to=t_to, delta_b=delta_b, motion_field=motion, events=events)


# ============================================================
# Configuration
# ============================================================

@dataclass
class LangBusterConfig:
    """Runtime configuration."""
    max_frames: int = 0              # 0 = all frames
    trace_enabled: bool = True
    trace_max_events: int = 10000
    cache_frames: int = 10           # LRU cache size for frame states
    motion_block_size: int = 8       # Block size for motion estimation
    deterministic: bool = True       # Deterministic execution
    seed: Optional[int] = None       # RNG seed


def create_video_state(video_path: str | Path, config: Optional[LangBusterConfig] = None) -> VideoState:
    """Factory: create VideoState from video file."""
    decoder = VideoDecoder(video_path)
    metadata = decoder.metadata
    cfg = config or LangBusterConfig()
    return VideoState(decoder=decoder, metadata=metadata, config=cfg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m langbuster.temporal <video_path>")
        sys.exit(1)

    state = create_video_state(sys.argv[1])
    print(f"Video: {state.metadata.path}")
    print(f"  {state.metadata.width}x{state.metadata.height} @ {state.metadata.fps:.2f}fps")
    print(f"  {state.metadata.frame_count} frames, {state.metadata.duration_sec:.2f}s")
    print(f"  SHA256: {state.metadata.sha256[:16]}...")

    # Test frame access
    fs0 = state.get_frame(0)
    print(f"\nFrame 0: {fs0.frame.index} @ {fs0.frame.timestamp:.3f}s")
    print(f"  Vector at center: {fs0.sample_vector(state.metadata.width/2, state.metadata.height/2)}")
    print(f"  Scalar at center: {fs0.sample_scalar(state.metadata.width/2, state.metadata.height/2)}")

    # Test transition
    if state.metadata.frame_count > 1:
        state.advance()
        trans = TemporalTransition.compute(state, 0, 1)
        print(f"\nTransition 0→1:")
        print(f"  Delta B stats: min={trans.delta_b.min()}, max={trans.delta_b.max()}, mean={trans.delta_b.mean():.2f}")
        print(f"  Events: {len(trans.events)}")