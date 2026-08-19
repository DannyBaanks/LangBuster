#!/usr/bin/env python3
"""
LangBuster v0.1 — CLI
Deterministic commands for spatiotemporal video computation.

Subcommands:
  run        -> execute .lb program on video, produce trace.json + evidence.json
  compile    -> compile .lb source to .lbb bytecode
  run-lbb    -> execute .lbb bytecode on video
  disasm     -> disassemble .lbb bytecode
  verify     -> verify .lbb file integrity
  replay     -> re-execute with same inputs, verify trace equality
  trace-diff -> compare two trace.json semantically
  validate   -> validate trace.json schema & consistency
  observe    -> query spatiotemporal observations (pixel, region, trajectory, events)
  info       -> show video metadata
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .video import VideoDecoder, video_info
from .runtime import LangBusterVM, LangBusterConfig, run_file, parse_file
from .trace import (
    TemporalTrace, run_file_with_trace, validate_trace, trace_diff,
    EvidencePackage, replay_trace, TemporalEvent
)
from .observe import SpatiotemporalObserver, observe_video
from .lbb_format import LBBProgram, LBBHeader, compute_sha256
from .lbb_compiler import compile_file, compile_string
from .lbb_vm import LBBVM, run_lbb


ENGINE_VERSION = "langbuster-0.1+cli-1"


# ============================================================
# High-level .lb interpreter commands (original)
# ============================================================

def cmd_info(args: argparse.Namespace) -> int:
    """Show video metadata."""
    info = video_info(args.video)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"  {k}: {v}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute program on video, produce trace + evidence."""
    cfg = LangBusterConfig(
        max_frames=args.max_frames,
        trace_enabled=True,
        deterministic=True,
        seed=args.seed,
    )

    trace = run_file_with_trace(
        args.video, args.program, config=cfg, seed=args.seed
    )

    # Save trace
    trace.save(args.trace)
    print(f"Trace -> {args.trace}")

    # Save evidence package
    evidence = EvidencePackage.from_execution(
        args.video, args.program, trace, args.trace, seed=args.seed, config=cfg
    )
    evidence.save(args.evidence)
    print(f"Evidence -> {args.evidence}")

    # Print summary
    validation = validate_trace(trace)
    print(f"Validation: {'OK' if validation['valid'] else 'FAIL'}")
    print(f"  Events: {validation['event_count']}, Frame hashes: {validation['frame_hash_count']}")
    if validation['warnings']:
        for w in validation['warnings']:
            print(f"  WARN: {w}")

    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay: re-execute with same inputs and compare traces."""
    # Load original trace
    with open(args.trace, encoding="utf-8") as f:
        orig_data = json.load(f)

    orig_trace = TemporalTrace.load(args.trace)
    orig_meta = orig_trace.metadata

    # Extract original config
    orig_config = orig_meta.get("config", {})
    orig_seed = orig_meta.get("seed")
    orig_program_sha = orig_meta.get("program_sha256")

    # Verify video matches
    if orig_meta.get("video_sha256") != video_info(args.video)["sha256"]:
        print("ERROR: Video file differs from original trace (sha256 mismatch)")
        return 1

    # Re-run with same config
    cfg = LangBusterConfig(
        max_frames=orig_config.get("max_frames", 0),
        trace_enabled=orig_config.get("trace_enabled", True),
        deterministic=orig_config.get("deterministic", True),
        seed=orig_seed,
    )

    print(f"Replaying with seed={orig_seed}, config={orig_config}")
    new_trace = run_file_with_trace(args.video, args.program, config=cfg, seed=orig_seed)
    new_trace.save(args.output_trace)

    # Compare
    diff = trace_diff(orig_trace, new_trace)
    if diff["verdict"] == "identical":
        print("REPLAY OK — trace identical to original")
        return 0
    else:
        print("REPLAY DIFFERENT:")
        print(json.dumps(diff, indent=2, default=str))
        return 1


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two traces semantically."""
    a = TemporalTrace.load(args.trace_a)
    b = TemporalTrace.load(args.trace_b)
    diff = trace_diff(a, b)
    print(json.dumps(diff, indent=2, default=str))
    return 0 if diff.get("verdict") == "identical" else 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate trace schema and consistency."""
    trace = TemporalTrace.load(args.trace)
    result = validate_trace(trace)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["valid"] else 1


def cmd_observe(args: argparse.Namespace) -> int:
    """Query spatiotemporal observations."""
    # Load trace if provided
    trace = None
    if args.trace:
        trace = TemporalTrace.load(args.trace)

    # Create observer
    cfg = LangBusterConfig(max_frames=args.max_frames, seed=args.seed)
    observer = observe_video(args.video, trace=trace, program_path=args.program, config=cfg)

    result = None

    if args.pixel:
        x, y, t = map(int, args.pixel.split(","))
        result = observer.pixel(x, y, t).to_dict()

    elif args.pixel_traj:
        x, y, t0, t1 = map(int, args.pixel_traj.split(","))
        traj = observer.pixel_trajectory(x, y, t0, t1)
        result = [p.to_dict() for p in traj]

    elif args.region:
        coords = list(map(int, args.region.split(",")))
        if len(coords) == 5:
            x1, y1, x2, y2, t = coords
            result = observer.region(x1, y1, x2, y2, t, include_pixels=not args.no_pixels).to_dict()
        elif len(coords) == 6:
            x1, y1, x2, y2, t0, t1 = coords
            result = observer.region_trajectory(x1, y1, x2, y2, t0, t1, include_pixels=not args.no_pixels).to_dict()
        else:
            print("ERROR: region requires 5 (x1,y1,x2,y2,t) or 6 (x1,y1,x2,y2,t0,t1) ints")
            return 1

    elif args.trajectory:
        pid = int(args.trajectory)
        result = observer.trajectory(pid).to_dict()

    elif args.events:
        ev_type = args.event_type if args.event_type else None
        region = None
        if args.region_filter:
            region = tuple(map(int, args.region_filter.split(",")))
        events = observer.events(event_type=ev_type, region=region)
        result = [e.to_dict() for e in events]

    elif args.transition:
        t_from, t_to = map(int, args.transition.split(","))
        trans = observer.transition(t_from, t_to)
        result = {
            "t_from": t_from,
            "t_to": t_to,
            "delta_b_shape": trans.delta_b.shape,
            "delta_b_stats": {
                "min": float(trans.delta_b.min()),
                "max": float(trans.delta_b.max()),
                "mean": float(trans.delta_b.mean()),
            },
            "motion_shape": trans.motion_field.shape,
            "events": trans.events,
        }

    elif args.crop:
        x1, y1, x2, y2, t = map(int, args.crop.split(","))
        result = observer.crop_observation(x1, y1, x2, y2, t)

    elif args.summary:
        result = observer.summary()

    else:
        result = observer.summary()

    # Output
    if args.text:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    return 0


# ============================================================
# Binary .lbb commands (new)
# ============================================================

def cmd_compile(args: argparse.Namespace) -> int:
    """Compile .lb source to .lbb bytecode."""
    video_sha = b"\0" * 32
    if args.video:
        video_sha = compute_sha256(Path(args.video).read_bytes())

    prog = compile_file(args.source, video_sha)
    prog.save(args.output)
    print(f"Compiled {args.source} -> {args.output}")
    print(f"  Code: {len(prog.code)} bytes, Data: {len(prog.data)} bytes")
    print(f"  Entry: 0x{prog.header.entry_point:X}")
    if args.disasm:
        print("\nDisassembly:")
        for inst in prog.disassemble():
            print(f"  {inst.offset:04X}: {inst}")
    return 0


def cmd_run_lbb(args: argparse.Namespace) -> int:
    """Execute .lbb bytecode on video."""
    result = run_lbb(args.program, args.video, args.max_instructions)

    # Save trace if requested
    if args.trace:
        trace = TemporalTrace(
            metadata={
                "engine": "LangBuster",
                "engine_version": "langbuster-0.1+lbb-1",
                "video_sha256": video_info(args.video)["sha256"] if args.video else "none",
                "program_sha256": compute_sha256(Path(args.program).read_bytes()),
                "config": {"max_instructions": args.max_instructions},
            },
            events=[TemporalEvent(**e) for e in result["trace_events"]],
        )
        trace.save(args.trace)
        print(f"Trace -> {args.trace}")

    print(f"Halted: {result['halted']}")
    print(f"Instructions: {result['instructions']}")
    print(f"Final T: {result['final_t']}")
    print(f"Regs: {result['final_regs']}")
    return 0


def cmd_disasm(args: argparse.Namespace) -> int:
    """Disassemble .lbb bytecode."""
    prog = LBBProgram.load(args.program)
    print(f"LBB v{prog.header.version} | Flags: 0x{prog.header.flags:02X}")
    print(f"Entry: 0x{prog.header.entry_point:X} | Code: {prog.header.code_size} bytes | Data: {prog.header.data_size} bytes")
    print(f"Video SHA256: {prog.header.video_sha256.hex()}")
    print(f"Program SHA256: {prog.header.program_sha256.hex()}")
    print()
    for inst in prog.disassemble():
        print(f"  {inst.offset:04X}: {inst}")
    if prog.data:
        print(f"\nData section ({len(prog.data)} bytes):")
        for offset, s in prog.strings.items():
            print(f"  0x{offset:04X}: {s!r}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify .lbb file integrity."""
    prog = LBBProgram.load(args.program)

    # Check magic, version
    if prog.header.magic != b"LBB\0":
        print("FAIL: Invalid magic")
        return 1
    if prog.header.version != 1:
        print(f"WARN: Unknown version {prog.header.version}")

    # Verify sizes match
    if len(prog.code) != prog.header.code_size:
        print(f"FAIL: Code size mismatch (header={prog.header.code_size}, actual={len(prog.code)})")
        return 1
    if len(prog.data) != prog.header.data_size:
        print(f"FAIL: Data size mismatch (header={prog.header.data_size}, actual={len(prog.data)})")
        return 1

    # Verify program hash if source provided
    if args.source:
        source = Path(args.source).read_bytes()
        actual = compute_sha256(source)
        if actual != prog.header.program_sha256:
            print(f"FAIL: Program hash mismatch")
            return 1

    print("OK: LBB file is valid")
    return 0


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="langbuster",
        description="LangBuster v0.1 — Spatiotemporal Video Esolang"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # info
    p_info = sub.add_parser("info", help="Show video metadata")
    p_info.add_argument("video", help="Video file (mp4, webm, gif)")
    p_info.add_argument("--json", action="store_true", help="JSON output")
    p_info.set_defaults(func=cmd_info)

    # run (high-level .lb interpreter)
    p_run = sub.add_parser("run", help="Execute .lb program on video, produce trace + evidence")
    p_run.add_argument("video", help="Video file")
    p_run.add_argument("program", help="LangBuster program (.lb)")
    p_run.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    p_run.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    p_run.add_argument("--trace", default="trace.json", help="Output trace file")
    p_run.add_argument("--evidence", default="evidence.json", help="Output evidence file")
    p_run.set_defaults(func=cmd_run)

    # replay
    p_replay = sub.add_parser("replay", help="Re-execute and verify trace equality")
    p_replay.add_argument("video", help="Video file")
    p_replay.add_argument("program", help="LangBuster program (.lb)")
    p_replay.add_argument("--trace", default="trace.json", help="Original trace file")
    p_replay.add_argument("--output-trace", default="trace.replay.json", help="Replay trace output")
    p_replay.set_defaults(func=cmd_replay)

    # trace-diff
    p_diff = sub.add_parser("trace-diff", help="Compare two traces semantically")
    p_diff.add_argument("trace_a")
    p_diff.add_argument("trace_b")
    p_diff.set_defaults(func=cmd_diff)

    # validate
    p_val = sub.add_parser("validate", help="Validate trace schema")
    p_val.add_argument("trace")
    p_val.set_defaults(func=cmd_validate)

    # observe
    p_obs = sub.add_parser("observe", help="Query spatiotemporal observations")
    p_obs.add_argument("video", help="Video file")
    p_obs.add_argument("--trace", help="Trace.json (optional)")
    p_obs.add_argument("--program", help="Program.lb (optional, for provenance)")
    p_obs.add_argument("--max-frames", type=int, default=0)
    p_obs.add_argument("--seed", type=int, default=None)
    # Queries
    p_obs.add_argument("--pixel", help="Pixel at x,y,t")
    p_obs.add_argument("--pixel-traj", help="Pixel trajectory x,y,t0,t1")
    p_obs.add_argument("--region", help="Region x1,y1,x2,y2,t or x1,y1,x2,y2,t0,t1")
    p_obs.add_argument("--no-pixels", action="store_true", help="Exclude pixel data from region")
    p_obs.add_argument("--trajectory", type=int, help="Particle trajectory by PID")
    p_obs.add_argument("--events", action="store_true", help="List events")
    p_obs.add_argument("--event-type", help="Filter events by type")
    p_obs.add_argument("--region-filter", help="Filter events by region x1,y1,x2,y2,t0,t1")
    p_obs.add_argument("--transition", help="Transition t_from,t_to")
    p_obs.add_argument("--crop", help="Crop x1,y1,x2,y2,t")
    p_obs.add_argument("--summary", action="store_true", help="Show summary (default)")
    # Output
    p_obs.add_argument("--json", action="store_true", help="JSON output (default)")
    p_obs.add_argument("--text", action="store_true", help="Pretty text")
    p_obs.set_defaults(func=cmd_observe)

    # compile
    p_compile = sub.add_parser("compile", help="Compile .lb source to .lbb bytecode")
    p_compile.add_argument("source", help="Source file (.lb)")
    p_compile.add_argument("-o", "--output", default="out.lbb", help="Output .lbb file")
    p_compile.add_argument("--video", help="Video file (for hash in header)")
    p_compile.add_argument("--disasm", action="store_true", help="Show disassembly after compile")
    p_compile.set_defaults(func=cmd_compile)

    # run-lbb
    p_run_lbb = sub.add_parser("run-lbb", help="Execute .lbb bytecode on video")
    p_run_lbb.add_argument("program", help="Program file (.lbb)")
    p_run_lbb.add_argument("video", nargs="?", help="Video file (optional)")
    p_run_lbb.add_argument("--max-instructions", type=int, default=1000000)
    p_run_lbb.add_argument("--trace", help="Output trace.json")
    p_run_lbb.set_defaults(func=cmd_run_lbb)

    # disasm
    p_disasm = sub.add_parser("disasm", help="Disassemble .lbb bytecode")
    p_disasm.add_argument("program", help="Program file (.lbb)")
    p_disasm.set_defaults(func=cmd_disasm)

    # verify
    p_verify = sub.add_parser("verify", help="Verify .lbb file integrity")
    p_verify.add_argument("program", help="Program file (.lbb)")
    p_verify.add_argument("--source", help="Original .lb source (for hash verification)")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())