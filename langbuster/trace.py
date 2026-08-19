#!/usr/bin/env python3
"""
LangBuster v0.1 — ExecutionTrace
Deterministic temporal trace with provenance, replay, and validation.
"""

from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .video import VideoMetadata
from .runtime import LangBusterVM, LangBusterConfig, run_program
from .grammar import parse


ENGINE_VERSION = "langbuster-0.1+trace-1"


@dataclass
class TemporalEvent:
    """Single event in temporal execution trace."""
    tick: int                    # logical tick (instruction count)
    t: int                       # frame index
    timestamp: float             # video timestamp
    type: str                    # event type
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "t": self.t,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemporalEvent":
        return cls(
            tick=d.get("tick", 0),
            t=d.get("t", 0),
            timestamp=d.get("timestamp", 0.0),
            type=d.get("type", ""),
            payload=d.get("payload", {}),
        )


@dataclass
class TemporalTrace:
    """Complete execution trace with metadata and events."""
    metadata: Dict[str, Any]
    events: List[TemporalEvent] = field(default_factory=list)
    frame_hashes: Dict[int, str] = field(default_factory=dict)  # t -> frame_b hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
            "frame_hashes": self.frame_hashes,
        }

    def save(self, path: str | Path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemporalTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        events = [TemporalEvent.from_dict(e) for e in data.get("events", [])]
        return cls(
            metadata=data.get("metadata", {}),
            events=events,
            frame_hashes=data.get("frame_hashes", {}),
        )


def program_hash(program_source: str) -> str:
    """SHA256 of program source."""
    return hashlib.sha256(program_source.encode()).hexdigest()


def video_hash(video_path: str | Path) -> str:
    """SHA256 of video file."""
    return hashlib.sha256(Path(video_path).read_bytes()).hexdigest()


def run_with_trace(
    video_path: str | Path,
    program_source: str,
    config: Optional[LangBusterConfig] = None,
    seed: Optional[int] = None,
) -> TemporalTrace:
    """Execute program and produce full TemporalTrace."""
    cfg = config or LangBusterConfig()
    if seed is not None:
        cfg.seed = seed

    prog = parse(program_source)
    vm = LangBusterVM(video_path, cfg, prog)
    result = vm.execute()

    # Build metadata
    meta = vm.video_state.metadata
    prog_hash = program_hash(program_source)
    vid_hash = video_hash(video_path)

    metadata = {
        "engine": "LangBuster",
        "engine_version": ENGINE_VERSION,
        "video_sha256": vid_hash,
        "video_path": str(video_path),
        "program_sha256": prog_hash,
        "seed": cfg.seed,
        "config": {
            "max_frames": cfg.max_frames,
            "trace_enabled": cfg.trace_enabled,
            "deterministic": cfg.deterministic,
        },
        "video": {
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "duration_sec": meta.duration_sec,
            "format": meta.format,
            "codec": meta.codec,
        },
        "execution": {
            "final_t": result["final_t"],
            "frames_processed": result.get("frames_processed", 0),
            "trace_events": len(result["trace"]),
        },
        "executed_at": datetime.utcnow().isoformat() + "Z",
    }

    # Convert trace events
    events = []
    for i, tr in enumerate(result["trace"]):
        events.append(TemporalEvent(
            tick=i,
            t=tr.get("t", 0),
            timestamp=tr.get("timestamp", 0.0),
            type=tr.get("type", "UNKNOWN"),
            payload=tr.get("payload", {}),
        ))

    return TemporalTrace(
        metadata=metadata,
        events=events,
        frame_hashes=result.get("frame_hashes", {}),
    )


def run_file_with_trace(
    video_path: str | Path,
    program_path: str | Path,
    config: Optional[LangBusterConfig] = None,
    seed: Optional[int] = None,
) -> TemporalTrace:
    """Execute program from file and produce trace."""
    source = Path(program_path).read_text(encoding="utf-8")
    return run_with_trace(video_path, source, config, seed)


# ============================================================
# Replay & Validation
# ============================================================

def replay_trace(
    video_path: str | Path,
    original_trace: TemporalTrace,
    config: Optional[LangBusterConfig] = None,
) -> TemporalTrace:
    """Re-execute with same parameters and produce new trace for comparison."""
    # Extract config from original trace
    orig_meta = original_trace.metadata
    orig_config = orig_meta.get("config", {})
    orig_seed = orig_meta.get("seed")

    cfg = config or LangBusterConfig(
        max_frames=orig_config.get("max_frames", 0),
        trace_enabled=orig_config.get("trace_enabled", True),
        deterministic=orig_config.get("deterministic", True),
        seed=orig_seed,
    )

    # Get program source - we need to re-run with same program
    # For replay, we assume the program is available or we re-run from trace
    # This is a simplified replay - in practice you'd store program too
    raise NotImplementedError("Replay requires program source. Use run_with_trace with same inputs.")


def validate_trace(trace: TemporalTrace) -> Dict[str, Any]:
    """Validate trace schema and internal consistency."""
    errors = []
    warnings = []

    # Required metadata
    required_meta = ["engine_version", "video_sha256", "program_sha256", "seed"]
    for key in required_meta:
        if key not in trace.metadata:
            errors.append(f"Missing required metadata: {key}")

    # Events ordered by tick non-decreasing
    last_tick = -1
    last_t = -1
    for i, ev in enumerate(trace.events):
        if ev.tick < last_tick:
            errors.append(f"Event {i}: tick {ev.tick} < previous {last_tick} (not monotonic)")
        last_tick = ev.tick

        if ev.t < last_t:
            warnings.append(f"Event {i}: frame t {ev.t} went backwards (may be valid for JUMP/PREV)")
        last_t = ev.t

    # Frame hashes consistency
    if trace.frame_hashes:
        for t, h in trace.frame_hashes.items():
            if not isinstance(h, str) or len(h) < 8:
                warnings.append(f"Frame {t} hash seems invalid: {h}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "event_count": len(trace.events),
        "frame_hash_count": len(trace.frame_hashes),
    }


# ============================================================
# Trace Diff (Semantic Comparison)
# ============================================================

def canonical_projection(trace: TemporalTrace | Dict) -> Dict[str, Any]:
    """Project trace to canonical form for comparison (excludes timestamps, session info)."""
    if isinstance(trace, TemporalTrace):
        d = trace.to_dict()
    else:
        d = trace

    # Keep only semantically relevant metadata
    meta = d.get("metadata", {})
    keep_meta = {
        k: meta.get(k) for k in
        ("engine_version", "video_sha256", "program_sha256", "seed",
         "config", "video", "execution")
    }

    # Events without wall-clock timestamp
    events = []
    for ev in d.get("events", []):
        events.append({
            "tick": ev.get("tick"),
            "t": ev.get("t"),
            "type": ev.get("type"),
            "payload": ev.get("payload"),
        })

    return {
        "metadata": keep_meta,
        "events": events,
        "frame_hashes": d.get("frame_hashes", {}),
    }


def trace_diff(a: TemporalTrace | Dict, b: TemporalTrace | Dict) -> Dict[str, Any]:
    """Semantic diff between two traces."""
    pa = canonical_projection(a)
    pb = canonical_projection(b)

    if pa == pb:
        return {"verdict": "identical"}

    # Compare metadata
    meta_diff = {}
    for k in set(pa["metadata"].keys()) | set(pb["metadata"].keys()):
        va, vb = pa["metadata"].get(k), pb["metadata"].get(k)
        if va != vb:
            meta_diff[k] = {"a": va, "b": vb}

    # Compare events
    ev_a, ev_b = pa["events"], pb["events"]
    common = min(len(ev_a), len(ev_b))
    first_diff = None
    for i in range(common):
        if ev_a[i] != ev_b[i]:
            first_diff = {"index": i, "a": ev_a[i], "b": ev_b[i]}
            break

    return {
        "verdict": "different",
        "identical": False,
        "metadata_diff": meta_diff,
        "events_count_a": len(ev_a),
        "events_count_b": len(ev_b),
        "first_event_diff": first_diff,
        "frame_hashes_identical": pa["frame_hashes"] == pb["frame_hashes"],
    }


# ============================================================
# Evidence Package (for reproducibility)
# ============================================================

@dataclass
class EvidencePackage:
    """Complete reproducible evidence package."""
    video_sha256: str
    video_path: str
    frames: int
    fps: float
    width: int
    height: int
    program_sha256: str
    program_path: str
    trace_sha256: str
    trace_path: str
    seed: Optional[int]
    config: Dict[str, Any]
    engine_version: str = ENGINE_VERSION
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    @classmethod
    def from_execution(
        cls,
        video_path: str | Path,
        program_path: str | Path,
        trace: TemporalTrace,
        trace_path: str,
        seed: Optional[int] = None,
        config: Optional[LangBusterConfig] = None,
    ) -> "EvidencePackage":
        vid_hash = video_hash(video_path)
        prog_hash = program_hash(Path(program_path).read_text(encoding="utf-8"))
        trace_hash = hashlib.sha256(json.dumps(trace.to_dict(), sort_keys=True).encode()).hexdigest()

        meta = trace.metadata
        vid_meta = meta.get("video", {})

        cfg_dict = {}
        if config:
            cfg_dict = {
                "max_frames": config.max_frames,
                "trace_enabled": config.trace_enabled,
                "deterministic": config.deterministic,
                "seed": config.seed,
            }

        return cls(
            video_sha256=vid_hash,
            video_path=str(video_path),
            frames=vid_meta.get("frame_count", 0),
            fps=vid_meta.get("fps", 0),
            width=vid_meta.get("width", 0),
            height=vid_meta.get("height", 0),
            program_sha256=prog_hash,
            program_path=str(program_path),
            trace_sha256=trace_hash,
            trace_path=str(trace_path),
            seed=seed or meta.get("seed"),
            config=cfg_dict,
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m langbuster.trace <video> <program.lb> [--seed N]")
        sys.exit(1)

    video = sys.argv[1]
    program = sys.argv[2]
    seed = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed = int(sys.argv[idx + 1])

    trace = run_file_with_trace(video, program, seed=seed)
    print(f"Trace: {len(trace.events)} events, {len(trace.frame_hashes)} frame hashes")

    # Save trace
    trace.save("trace.json")
    print("Saved trace.json")

    # Validate
    validation = validate_trace(trace)
    print(f"Validation: {validation}")

    # Evidence package
    evidence = EvidencePackage.from_execution(video, program, trace, "trace.json", seed=seed)
    evidence.save("evidence.json")
    print("Saved evidence.json")