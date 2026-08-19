#!/usr/bin/env python3
"""
LangBuster v0.1 — Visual Observer
Spatiotemporal observation layer: pixel, region, trajectory, event queries.
Machine-readable output with full provenance.
"""

from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
from PIL import Image

from .video import VideoDecoder, Frame, VideoMetadata
from .temporal import VideoState, FrameState, TemporalTransition, create_video_state
from .trace import TemporalTrace, TemporalEvent
from .runtime import LangBusterVM, LangBusterConfig

OBSERVATION_SCHEMA = "langbuster.observation/1"
ENGINE_VERSION = "langbuster-0.1+observe-1"


# ============================================================
# Observation Data Structures
# ============================================================

@dataclass
class SpatiotemporalPixel:
    """Single pixel observation across space and time."""
    x: int
    y: int
    t: int
    rgba: List[int]                    # [R,G,B,A] at (x,y,t)
    vector: Tuple[float, float]        # (vx, vy) from R,G
    vector_magnitude: float
    vector_direction_rad: float
    scalar_b: int                      # B channel value
    trace_value: int                   # trace layer value
    delta_from_prev: Optional[int] = None      # B - B_prev
    motion_vector: Optional[Tuple[float, float]] = None  # motion from prev frame
    events_at: List[Dict[str, Any]] = field(default_factory=list)
    particle_visits: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = "pixel_st"
        d["provenance"]["schema"] = OBSERVATION_SCHEMA
        return d


@dataclass
class SpatiotemporalRegion:
    """Region observation across space and time."""
    x1: int
    y1: int
    x2: int
    y2: int
    t0: int
    t1: int
    width: int
    height: int
    frame_count: int
    pixels: List[SpatiotemporalPixel] = field(default_factory=list)  # sampled
    stats: Dict[str, Any] = field(default_factory=dict)
    events_in_region: List[Dict[str, Any]] = field(default_factory=list)
    particles_in_region: List[int] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = "region_st"
        d["provenance"]["schema"] = OBSERVATION_SCHEMA
        return d


@dataclass
class ParticleTrajectory:
    """Particle trajectory across frames (spatiotemporal path)."""
    pid: int
    spawn_t: int
    spawn_x: float
    spawn_y: float
    path: List[Dict[str, Any]] = field(default_factory=list)  # [{t, x, y, state, instr}]
    events: List[Dict[str, Any]] = field(default_factory=list)
    death_t: Optional[int] = None
    death_x: Optional[float] = None
    death_y: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = "particle_trajectory"
        d["provenance"]["schema"] = OBSERVATION_SCHEMA
        return d


@dataclass
class SpatiotemporalObservation:
    """Complete spatiotemporal observation container."""
    # Identity
    video_path: str
    video_sha256: str
    program_path: Optional[str] = None
    program_sha256: Optional[str] = None
    trace_path: Optional[str] = None
    trace_sha256: Optional[str] = None

    # Video metadata
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_sec: float = 0.0

    # Execution metadata
    seed: Optional[int] = None
    engine_version: str = ENGINE_VERSION
    observation_schema: str = OBSERVATION_SCHEMA
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Content
    pixels: List[SpatiotemporalPixel] = field(default_factory=list)
    regions: List[SpatiotemporalRegion] = field(default_factory=list)
    trajectories: List[ParticleTrajectory] = field(default_factory=list)
    events: List[TemporalEvent] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    # Global provenance
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"]["schema"] = self.observation_schema
        d["provenance"]["engine_version"] = self.engine_version
        return d

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# Visual Observer Engine
# ============================================================

class SpatiotemporalObserver:
    """
    Observes spatiotemporal state from VideoState + TemporalTrace.
    Provides queries for pixels, regions, trajectories, events.
    """

    def __init__(
        self,
        video_state: VideoState,
        trace: Optional[TemporalTrace] = None,
        program_path: Optional[str] = None,
    ):
        self.video_state = video_state
        self.trace = trace
        self.program_path = program_path

        self.metadata = video_state.metadata
        self.video_sha256 = self.metadata.sha256
        self.program_sha256 = None
        if program_path:
            self.program_sha256 = hashlib.sha256(Path(program_path).read_bytes()).hexdigest()

        self.observation = SpatiotemporalObservation(
            video_path=str(video_state.decoder.path),
            video_sha256=self.video_sha256,
            program_path=program_path,
            program_sha256=self.program_sha256,
            trace_path=str(getattr(trace, "_trace_path", "")) if trace else None,
            width=self.metadata.width,
            height=self.metadata.height,
            fps=self.metadata.fps,
            frame_count=self.metadata.frame_count,
            duration_sec=self.metadata.duration_sec,
            seed=video_state.config.seed,
        )

        # Build indices for fast queries
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices from trace events."""
        self._events_at_st: Dict[Tuple[int, int, int], List[Dict]] = {}  # (x,y,t) -> events
        self._visits_at_st: Dict[Tuple[int, int, int], List[Dict]] = {}
        self._events_by_type: Dict[str, List[Dict]] = {}
        self._particle_events: Dict[int, List[Dict]] = {}

        if self.trace:
            for ev in self.trace.events:
                # Index by type
                self._events_by_type.setdefault(ev.type, []).append(ev.to_dict())

                # Index by spacetime coordinates
                payload = ev.payload
                x = payload.get("x")
                y = payload.get("y")
                t = ev.t

                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    key = (int(round(x)), int(round(y)), t)
                    self._events_at_st.setdefault(key, []).append(ev.to_dict())

                    # Also index visits
                    if ev.type in ("PIXEL", "VECTOR", "REGION", "DELTA", "MOTION",
                                   "PARTICLE_MOVE", "PARTICLE_SPAWN", "PARTICLE_DEATH"):
                        self._visits_at_st.setdefault(key, []).append(ev.to_dict())

                # Index by particle
                pid = payload.get("pid")
                if isinstance(pid, int) and pid >= 0:
                    self._particle_events.setdefault(pid, []).append(ev.to_dict())

    def _sample_frame_state(self, t: int) -> FrameState:
        """Get or load FrameState for frame t."""
        return self.video_state.get_frame(t)

    def _make_pixel_st(self, x: int, y: int, t: int) -> SpatiotemporalPixel:
        """Create spatiotemporal pixel observation."""
        if not (0 <= x < self.metadata.width and 0 <= y < self.metadata.height and 0 <= t < self.metadata.frame_count):
            raise ValueError(f"Coordinates out of bounds: ({x},{y},{t})")

        fs = self._sample_frame_state(t)

        # Original RGB from video
        frame = self.video_state.decoder.frame_at(t)
        r, g, b = int(frame.rgb[y, x, 0]), int(frame.rgb[y, x, 1]), int(frame.rgb[y, x, 2])

        # Vector field at pixel center
        vx, vy = fs.sample_vector(x + 0.5, y + 0.5)
        mag = math.hypot(vx, vy)
        ang = math.atan2(vy, vx)

        # Scalar B
        scalar_b = fs.sample_scalar(x + 0.5, y + 0.5)

        # Trace layer
        trace_val = int(fs.field_trace[y, x]) if fs.field_trace is not None else 0

        # Delta from previous frame
        delta = None
        if t > 0:
            fs_prev = self._sample_frame_state(t - 1)
            prev_b = fs_prev.sample_scalar(x + 0.5, y + 0.5)
            delta = (scalar_b - prev_b) & 0xFF

        # Motion vector (from transition)
        motion = None
        if t > 0:
            trans = TemporalTransition.compute(self.video_state, t - 1, t)
            if 0 <= y < trans.motion_field.shape[0] and 0 <= x < trans.motion_field.shape[1]:
                mv = trans.motion_field[y, x]
                motion = (float(mv[0]), float(mv[1]))

        # Events at this spacetime coordinate
        events = self._events_at_st.get((x, y, t), [])

        # Particle visits
        visits = self._visits_at_st.get((x, y, t), [])

        return SpatiotemporalPixel(
            x=x, y=y, t=t,
            rgba=[r, g, b, 255],
            vector=(vx, vy),
            vector_magnitude=mag,
            vector_direction_rad=ang,
            scalar_b=scalar_b,
            trace_value=trace_val,
            delta_from_prev=delta,
            motion_vector=motion,
            events_at=events,
            particle_visits=visits,
            provenance={
                "source": "video_frame + frame_state + trace_indices",
                "video_sha256": self.video_sha256,
                "seed": self.video_state.config.seed,
                "schema": OBSERVATION_SCHEMA,
            }
        )

    # ---- Public Query API ----

    def pixel(self, x: int, y: int, t: int) -> SpatiotemporalPixel:
        """Observe single pixel at (x,y,t)."""
        return self._make_pixel_st(x, y, t)

    def pixel_trajectory(self, x: int, y: int, t0: int, t1: int) -> List[SpatiotemporalPixel]:
        """Pixel trajectory across time range [t0, t1]."""
        return [self._make_pixel_st(x, y, t) for t in range(t0, min(t1 + 1, self.metadata.frame_count))]

    def region(self, x1: int, y1: int, x2: int, y2: int, t: int, include_pixels: bool = True, max_pixels: int = 100) -> SpatiotemporalRegion:
        """Observe region at single frame t."""
        x1, x2 = sorted([max(0, x1), min(self.metadata.width - 1, x2)])
        y1, y2 = sorted([max(0, y1), min(self.metadata.height - 1, y2)])
        t = max(0, min(t, self.metadata.frame_count - 1))

        w, h = x2 - x1 + 1, y2 - y1 + 1
        pixels = []

        if include_pixels:
            # Sample pixels (stride if too many)
            stride = max(1, int(math.sqrt(w * h / max_pixels))) if w * h > max_pixels else 1
            for yy in range(y1, y2 + 1, stride):
                for xx in range(x1, x2 + 1, stride):
                    pixels.append(self._make_pixel_st(xx, yy, t))

        # Stats
        fs = self._sample_frame_state(t)
        region_b = fs.field_b[y1:y2+1, x1:x2+1]
        region_trace = fs.field_trace[y1:y2+1, x1:x2+1] if fs.field_trace is not None else None

        stats = {
            "scalar_b": {
                "mean": float(np.mean(region_b)),
                "std": float(np.std(region_b)),
                "min": int(np.min(region_b)),
                "max": int(np.max(region_b)),
                "nonzero": int(np.count_nonzero(region_b)),
            },
            "trace": {
                "nonzero": int(np.count_nonzero(region_trace)) if region_trace is not None else 0,
            },
            "vector_field": {},
        }

        # Vector field stats
        vx_vals, vy_vals = [], []
        for yy in range(y1, y2 + 1):
            for xx in range(x1, x2 + 1):
                vx, vy = fs.sample_vector(xx + 0.5, yy + 0.5)
                vx_vals.append(vx)
                vy_vals.append(vy)
        stats["vector_field"] = {
            "vx_mean": float(np.mean(vx_vals)),
            "vy_mean": float(np.mean(vy_vals)),
            "magnitude_mean": float(np.mean([math.hypot(vx, vy) for vx, vy in zip(vx_vals, vy_vals)])),
        }

        # Events in region at this frame
        events = []
        pids = set()
        for yy in range(y1, y2 + 1):
            for xx in range(x1, x2 + 1):
                key = (xx, yy, t)
                events.extend(self._events_at_st.get(key, []))
                for v in self._visits_at_st.get(key, []):
                    pid = v.get("payload", {}).get("pid")
                    if isinstance(pid, int):
                        pids.add(pid)

        return SpatiotemporalRegion(
            x1=x1, y1=y1, x2=x2, y2=y2, t0=t, t1=t,
            width=w, height=h, frame_count=1,
            pixels=pixels,
            stats=stats,
            events_in_region=events,
            particles_in_region=sorted(pids),
            provenance={
                "source": "video_frame + frame_state + trace_indices",
                "video_sha256": self.video_sha256,
                "seed": self.video_state.config.seed,
                "region": [x1, y1, x2, y2, t],
                "schema": OBSERVATION_SCHEMA,
            }
        )

    def region_trajectory(self, x1: int, y1: int, x2: int, y2: int, t0: int, t1: int, include_pixels: bool = False) -> SpatiotemporalRegion:
        """Region across time range [t0, t1]."""
        x1, x2 = sorted([max(0, x1), min(self.metadata.width - 1, x2)])
        y1, y2 = sorted([max(0, y1), min(self.metadata.height - 1, y2)])
        t0 = max(0, min(t0, self.metadata.frame_count - 1))
        t1 = max(t0, min(t1, self.metadata.frame_count - 1))

        w, h = x2 - x1 + 1, y2 - y1 + 1
        frame_count = t1 - t0 + 1
        all_pixels = []

        if include_pixels:
            # Sample a few frames
            for t in range(t0, t1 + 1, max(1, frame_count // 5)):
                all_pixels.extend(self.region(x1, y1, x2, y2, t, include_pixels=True, max_pixels=25).pixels)

        # Aggregate stats across time
        all_b = []
        all_trace = []
        all_vx, all_vy = [], []
        all_events = []
        all_pids = set()

        for t in range(t0, t1 + 1):
            fs = self._sample_frame_state(t)
            region_b = fs.field_b[y1:y2+1, x1:x2+1]
            all_b.append(region_b)
            if fs.field_trace is not None:
                all_trace.append(fs.field_trace[y1:y2+1, x1:x2+1])

            for yy in range(y1, y2 + 1):
                for xx in range(x1, x2 + 1):
                    vx, vy = fs.sample_vector(xx + 0.5, yy + 0.5)
                    all_vx.append(vx)
                    all_vy.append(vy)

                    key = (xx, yy, t)
                    all_events.extend(self._events_at_st.get(key, []))
                    for v in self._visits_at_st.get(key, []):
                        pid = v.get("payload", {}).get("pid")
                        if isinstance(pid, int):
                            all_pids.add(pid)

        all_b_arr = np.stack(all_b) if all_b else np.array([])
        stats = {
            "scalar_b": {
                "mean": float(np.mean(all_b_arr)) if all_b_arr.size else 0,
                "std": float(np.std(all_b_arr)) if all_b_arr.size else 0,
                "min": int(np.min(all_b_arr)) if all_b_arr.size else 0,
                "max": int(np.max(all_b_arr)) if all_b_arr.size else 0,
            },
            "vector_field": {
                "vx_mean": float(np.mean(all_vx)) if all_vx else 0,
                "vy_mean": float(np.mean(all_vy)) if all_vy else 0,
            },
            "frames_analyzed": frame_count,
        }

        return SpatiotemporalRegion(
            x1=x1, y1=y1, x2=x2, y2=y2, t0=t0, t1=t1,
            width=w, height=h, frame_count=frame_count,
            pixels=all_pixels,
            stats=stats,
            events_in_region=all_events,
            particles_in_region=sorted(all_pids),
            provenance={
                "video_sha256": self.video_sha256,
                "seed": self.video_state.config.seed,
                "region": [x1, y1, x2, y2, t0, t1],
                "schema": OBSERVATION_SCHEMA,
            }
        )

    def trajectory(self, pid: int) -> ParticleTrajectory:
        """Reconstruct particle trajectory from trace."""
        events = self._particle_events.get(pid, [])
        if not events:
            raise ValueError(f"No events found for particle {pid}")

        # Sort by tick
        events.sort(key=lambda e: e.get("tick", 0))

        # Find spawn
        spawn_ev = next((e for e in events if e["type"] == "PARTICLE_SPAWN"), None)
        spawn_t = spawn_ev["t"] if spawn_ev else 0
        spawn_x = spawn_ev["payload"].get("x", 0) if spawn_ev else 0
        spawn_y = spawn_ev["payload"].get("y", 0) if spawn_ev else 0

        # Build path from MOVE events
        path = []
        death_t = death_x = death_y = None
        for ev in events:
            if ev["type"] == "PARTICLE_MOVE":
                pl = ev["payload"]
                path.append({
                    "t": ev["t"],
                    "x": pl.get("x_to"),
                    "y": pl.get("y_to"),
                })
            elif ev["type"] == "PARTICLE_DEATH":
                death_t = ev["t"]
                death_x = ev["payload"].get("x")
                death_y = ev["payload"].get("y")

        return ParticleTrajectory(
            pid=pid,
            spawn_t=spawn_t,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            path=path,
            events=events,
            death_t=death_t,
            death_x=death_x,
            death_y=death_y,
            provenance={
                "video_sha256": self.video_sha256,
                "seed": self.video_state.config.seed,
                "schema": OBSERVATION_SCHEMA,
            }
        )

    def events(self, event_type: Optional[str] = None, region: Optional[Tuple[int, int, int, int, int, int]] = None) -> List[TemporalEvent]:
        """Query events by type and/or spatiotemporal region."""
        if event_type:
            result = [TemporalEvent.from_dict(e) for e in self._events_by_type.get(event_type, [])]
        else:
            result = list(self.trace.events) if self.trace else []

        if region:
            x1, y1, x2, y2, t0, t1 = region
            filtered = []
            for ev in result:
                x = ev.payload.get("x")
                y = ev.payload.get("y")
                t = ev.t
                if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and
                    x1 <= x <= x2 and y1 <= y <= y2 and t0 <= t <= t1):
                    filtered.append(ev)
            return filtered

        return result

    def transition(self, t_from: int, t_to: int) -> TemporalTransition:
        """Get temporal transition between frames."""
        return TemporalTransition.compute(self.video_state, t_from, t_to)

    def summary(self) -> Dict[str, Any]:
        """Compact summary for agents."""
        total_events = len(self.trace.events) if self.trace else 0
        event_types = {}
        if self.trace:
            for ev in self.trace.events:
                event_types[ev.type] = event_types.get(ev.type, 0) + 1

        return {
            "video": {
                "path": str(self.video_state.decoder.path),
                "sha256": self.video_sha256,
                "size": [self.metadata.width, self.metadata.height],
                "fps": self.metadata.fps,
                "frames": self.metadata.frame_count,
                "duration_sec": self.metadata.duration_sec,
            },
            "program": {
                "path": self.program_path,
                "sha256": self.program_sha256,
            } if self.program_path else None,
            "trace": {
                "path": getattr(self.trace, "_trace_path", None),
                "total_events": total_events,
                "event_types": event_types,
            } if self.trace else None,
            "provenance": {"schema": OBSERVATION_SCHEMA, "engine": ENGINE_VERSION},
        }

    def crop_observation(self, x1: int, y1: int, x2: int, y2: int, t: int) -> Dict[str, Any]:
        """Generate crop of frame + observation metadata."""
        x1, x2 = sorted([max(0, x1), min(self.metadata.width - 1, x2)])
        y1, y2 = sorted([max(0, y1), min(self.metadata.height - 1, y2)])

        frame = self.video_state.decoder.frame_at(t)
        crop_img = Image.fromarray(frame.rgb[y1:y2+1, x1:x2+1])
        crop_bytes = crop_img.tobytes()
        crop_hash = hashlib.sha256(crop_bytes).hexdigest()

        region_obs = self.region(x1, y1, x2, y2, t, include_pixels=False)

        return {
            "crop": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1 + 1, "height": y2 - y1 + 1,
                "sha256": crop_hash,
                "frame_t": t,
            },
            "region_observation": region_obs.to_dict(),
            "provenance": {
                "source_video": str(self.video_state.decoder.path),
                "source_sha256": self.video_sha256,
                "crop_sha256": crop_hash,
            }
        }


def observe_video(
    video_path: str | Path,
    trace: Optional[TemporalTrace] = None,
    program_path: Optional[str] = None,
    config: Optional[LangBusterConfig] = None,
) -> SpatiotemporalObserver:
    """Main entry point: create observer from video (+ optional trace)."""
    state = create_video_state(video_path, config)
    # If trace provided but state not run, we need to run VM to populate frame states
    # For now, assume frame states are loaded on demand via get_frame
    return SpatiotemporalObserver(state, trace, program_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m langbuster.observe <video> [--trace trace.json] [--program prog.lb]")
        sys.exit(1)

    video = sys.argv[1]
    trace_path = None
    prog_path = None

    if "--trace" in sys.argv:
        idx = sys.argv.index("--trace")
        trace_path = sys.argv[idx + 1]
    if "--program" in sys.argv:
        idx = sys.argv.index("--program")
        prog_path = sys.argv[idx + 1]

    trace = TemporalTrace.load(trace_path) if trace_path else None
    observer = observe_video(video, trace=trace, program_path=prog_path)

    # Default: print summary
    import json
    print(json.dumps(observer.summary(), indent=2, default=str))