#!/usr/bin/env python3
"""
LangBuster v0.1 — Runtime / Interpreter
Executes LangBuster AST on VideoState (spatiotemporal computation).
"""

from __future__ import annotations
import hashlib
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

import numpy as np

from .video import VideoDecoder, Frame, VideoMetadata
from .temporal import (
    VideoState, FrameState, TemporalTransition, LangBusterConfig,
    TemporalOp, TEMPORAL_MNEMONIC, create_video_state
)
from .grammar import (
    Program, FrameStmt, TemporalStmt, SpatialStmt, ControlStmt,
    EmitStmt, QueryStmt, CoordExpr, RegionExpr, BinaryExpr, UnaryExpr,
    NumberLiteral, StringLiteral, IdentExpr, ParticleExpr, ASTNode,
    parse, parse_file
)


# ============================================================
# Evaluation Context
# ============================================================

@dataclass
class EvalContext:
    """Runtime evaluation context."""
    state: VideoState
    variables: Dict[str, Any] = field(default_factory=dict)
    event_handlers: Dict[str, List[ControlStmt]] = field(default_factory=dict)
    loop_stack: List[Dict[str, Any]] = field(default_factory=list)
    particle_counter: int = 0
    particles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    trace_events: List[Dict[str, Any]] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def eval_coord(self, coord: CoordExpr) -> tuple:
        x = int(self.eval_expr(coord.x))
        y = int(self.eval_expr(coord.y))
        t = int(self.eval_expr(coord.t)) if coord.t else self.state.current_t
        return x, y, t

    def eval_region(self, region: RegionExpr) -> tuple:
        x1 = int(self.eval_expr(region.x1))
        y1 = int(self.eval_expr(region.y1))
        x2 = int(self.eval_expr(region.x2))
        y2 = int(self.eval_expr(region.y2))
        t0 = int(self.eval_expr(region.t0)) if region.t0 else self.state.current_t
        t1 = int(self.eval_expr(region.t1)) if region.t1 else t0
        return x1, y1, x2, y2, t0, t1

    def eval_expr(self, expr: ASTNode) -> Any:
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, StringLiteral):
            return expr.value
        if isinstance(expr, IdentExpr):
            return self.variables.get(expr.name, 0)
        if isinstance(expr, ParticleExpr):
            return self.eval_expr(expr.pid)
        if isinstance(expr, BinaryExpr):
            left = self.eval_expr(expr.left)
            right = self.eval_expr(expr.right)
            return self._eval_binary(expr.op, left, right)
        if isinstance(expr, UnaryExpr):
            val = self.eval_expr(expr.expr)
            return self._eval_unary(expr.op, val)
        if isinstance(expr, CoordExpr):
            return self.eval_coord(expr)
        if isinstance(expr, RegionExpr):
            return self.eval_region(expr)
        return 0

    def _eval_binary(self, op: str, left: Any, right: Any) -> Any:
        if op == "PLUS": return left + right
        if op == "MINUS": return left - right
        if op == "STAR": return left * right
        if op == "SLASH": return left / right if right != 0 else 0
        if op == "MODULO": return left % right if right != 0 else 0
        if op == "EQ": return 1 if left == right else 0
        if op == "NEQ": return 1 if left != right else 0
        if op == "LT": return 1 if left < right else 0
        if op == "GT": return 1 if left > right else 0
        if op == "LE": return 1 if left <= right else 0
        if op == "GE": return 1 if left >= right else 0
        if op == "ASSIGN":
            if isinstance(expr.left, IdentExpr):
                self.variables[expr.left.name] = right
            return right
        return 0

    def _eval_unary(self, op: str, val: Any) -> Any:
        if op == "MINUS": return -val
        if op == "PLUS": return +val
        return val


# ============================================================
# Interpreter
# ============================================================

class LangBusterVM:
    """LangBuster Virtual Machine - executes spatiotemporal programs."""

    def __init__(self, video_path: str | Path, config: Optional[LangBusterConfig] = None, program: Optional[Program] = None):
        self.video_state = create_video_state(video_path, config)
        self.config = self.video_state.config
        self.program = program
        self.ctx = EvalContext(state=self.video_state)
        self.trace: List[Dict[str, Any]] = []
        self._frame_hashes: Dict[int, str] = {}

        # Seed RNG for determinism
        if self.config.seed is not None:
            self.ctx.rng.seed(self.config.seed)

    def _hash_frame(self, t: int) -> str:
        if t not in self._frame_hashes:
            fs = self.video_state.get_frame(t)
            self._frame_hashes[t] = hashlib.sha256(fs.field_b.tobytes()).hexdigest()[:16]
        return self._frame_hashes[t]

    def execute(self, program: Optional[Program] = None) -> Dict[str, Any]:
        """Execute program on video."""
        prog = program or self.program
        if not prog:
            raise ValueError("No program provided")

        self.trace = []
        self._emit_trace("PROGRAM_START", {"program_hash": self._hash_program(prog)})

        try:
            self._execute_block(prog.statements)
        except Exception as e:
            self._emit_trace("ERROR", {"error": str(e), "type": type(e).__name__})
            raise

        self._emit_trace("PROGRAM_END", {
            "final_t": self.video_state.current_t,
            "frames_processed": len(self._frame_hashes),
            "trace_events": len(self.trace),
        })

        return {
            "success": True,
            "final_t": self.video_state.current_t,
            "trace": self.trace,
            "video_hash": self.video_state.metadata.sha256,
            "program_hash": self._hash_program(prog),
        }

    def _hash_program(self, prog: Program) -> str:
        # Simple hash of AST structure
        import json
        return hashlib.sha256(json.dumps(self._ast_to_dict(prog), sort_keys=True).encode()).hexdigest()[:16]

    def _ast_to_dict(self, node: ASTNode) -> Any:
        if hasattr(node, '__dict__'):
            d = {"type": node.__class__.__name__}
            for k, v in node.__dict__.items():
                if isinstance(v, list):
                    d[k] = [self._ast_to_dict(item) for item in v]
                elif hasattr(v, '__dict__'):
                    d[k] = self._ast_to_dict(v)
                else:
                    d[k] = v
            return d
        return node

    def _execute_block(self, statements: List[ASTNode]):
        for stmt in statements:
            self._execute_statement(stmt)

    def _execute_statement(self, stmt: ASTNode):
        if isinstance(stmt, FrameStmt):
            self._exec_frame_stmt(stmt)
        elif isinstance(stmt, TemporalStmt):
            self._exec_temporal_stmt(stmt)
        elif isinstance(stmt, SpatialStmt):
            self._exec_spatial_stmt(stmt)
        elif isinstance(stmt, ControlStmt):
            self._exec_control_stmt(stmt)
        elif isinstance(stmt, EmitStmt):
            self._exec_emit_stmt(stmt)
        elif isinstance(stmt, QueryStmt):
            self._exec_query_stmt(stmt)
        elif isinstance(stmt, BinaryExpr) and stmt.op == "ASSIGN":
            # Variable assignment
            self.ctx.eval_expr(stmt)
        else:
            # Expression statement (evaluate for side effects)
            self.ctx.eval_expr(stmt)

    # --- Frame Navigation ---
    def _exec_frame_stmt(self, stmt: FrameStmt):
        if stmt.op == "FRAME":
            t = int(self.ctx.eval_expr(stmt.arg)) if stmt.arg else 0
            self.video_state.jump(t)
            self._emit_trace("FRAME", {"t": t, "frame_hash": self._hash_frame(t)})
        elif stmt.op == "NEXT":
            if self.video_state.advance():
                self._emit_trace("NEXT", {"t": self.video_state.current_t, "frame_hash": self._hash_frame(self.video_state.current_t)})
            else:
                self._emit_trace("NEXT", {"t": self.video_state.current_t, "at_end": True})
        elif stmt.op == "PREV":
            if self.video_state.retreat():
                self._emit_trace("PREV", {"t": self.video_state.current_t, "frame_hash": self._hash_frame(self.video_state.current_t)})
            else:
                self._emit_trace("PREV", {"t": self.video_state.current_t, "at_start": True})
        elif stmt.op == "JUMP":
            t = int(self.ctx.eval_expr(stmt.arg))
            if self.video_state.jump(t):
                self._emit_trace("JUMP", {"t": t, "frame_hash": self._hash_frame(t)})
            else:
                self._emit_trace("JUMP", {"t": t, "invalid": True})

    # --- Temporal Operations ---
    def _exec_temporal_stmt(self, stmt: TemporalStmt):
        fs_cur = self.video_state.current

        if stmt.op == "DELTA":
            x, y, t = self.ctx.eval_coord(stmt.args[0])
            # Ensure we're at the right frame
            if t != self.video_state.current_t:
                self.video_state.jump(t)
            delta = self.video_state.delta(x, y)
            self.ctx.variables["_delta"] = delta
            self._emit_trace("DELTA", {"x": x, "y": y, "t": t, "delta": delta})

        elif stmt.op == "MOTION":
            x1, y1, x2, y2, t0, t1 = self.ctx.eval_region(stmt.args[0])
            motion = self.video_state.motion_vectors(x1, y1, x2, y2)
            self.ctx.variables["_motion"] = motion
            self._emit_trace("MOTION", {"region": [x1, y1, x2, y2], "motion_shape": motion.shape})

        elif stmt.op == "TRACE":
            # Particle trajectory tracing - simplified
            pid = int(self.ctx.eval_expr(stmt.args[0]))
            self._emit_trace("TRACE", {"pid": pid, "note": "particle trace not fully implemented"})

        elif stmt.op == "PERSIST":
            x1, y1, x2, y2, t0, t1 = self.ctx.eval_region(stmt.args[0])
            # t0, t1 from args[1], args[2]
            t0_val = int(self.ctx.eval_expr(stmt.args[1]))
            t1_val = int(self.ctx.eval_expr(stmt.args[2]))
            persists = self.video_state.persist_check(x1, y1, x2, y2, t0_val, t1_val)
            self.ctx.variables["_persist"] = 1 if persists else 0
            self._emit_trace("PERSIST", {"region": [x1, y1, x2, y2], "t0": t0_val, "t1": t1_val, "persists": persists})

        elif stmt.op == "APPEAR":
            x1, y1, x2, y2, _, _ = self.ctx.eval_region(stmt.args[0])
            appears = self.video_state.appear_check(x1, y1, x2, y2)
            self.ctx.variables["_appear"] = 1 if appears else 0
            self._emit_trace("APPEAR", {"region": [x1, y1, x2, y2], "appears": appears})

        elif stmt.op == "DISAPPEAR":
            x1, y1, x2, y2, _, _ = self.ctx.eval_region(stmt.args[0])
            disappears = self.video_state.disappear_check(x1, y1, x2, y2)
            self.ctx.variables["_disappear"] = 1 if disappears else 0
            self._emit_trace("DISAPPEAR", {"region": [x1, y1, x2, y2], "disappears": disappears})

    # --- Spatial Operations ---
    def _exec_spatial_stmt(self, stmt: SpatialStmt):
        fs_cur = self.video_state.current

        if stmt.op == "PIXEL":
            x, y, t = self.ctx.eval_coord(stmt.args[0])
            if t != self.video_state.current_t:
                self.video_state.jump(t)
            val = fs_cur.sample_scalar(x + 0.5, y + 0.5)
            vx, vy = fs_cur.sample_vector(x + 0.5, y + 0.5)
            self.ctx.variables["_pixel_val"] = val
            self.ctx.variables["_pixel_vx"] = vx
            self.ctx.variables["_pixel_vy"] = vy
            self._emit_trace("PIXEL", {"x": x, "y": y, "t": t, "value": val, "vector": [vx, vy]})

        elif stmt.op == "VECTOR":
            x, y, t = self.ctx.eval_coord(stmt.args[0])
            if t != self.video_state.current_t:
                self.video_state.jump(t)
            vx, vy = fs_cur.sample_vector(x + 0.5, y + 0.5)
            self.ctx.variables["_vx"] = vx
            self.ctx.variables["_vy"] = vy
            self._emit_trace("VECTOR", {"x": x, "y": y, "t": t, "vx": vx, "vy": vy})

        elif stmt.op == "REGION":
            x1, y1, x2, y2, t0, t1 = self.ctx.eval_region(stmt.args[0])
            if t0 != self.video_state.current_t:
                self.video_state.jump(t0)
            region_b = fs_cur.field_b[y1:y2+1, x1:x2+1].copy()
            self.ctx.variables["_region_b"] = region_b
            self._emit_trace("REGION", {"region": [x1, y1, x2, y2], "t": t0, "shape": region_b.shape})

        elif stmt.op == "READ":
            x, y, t = self.ctx.eval_coord(stmt.args[0])
            if t != self.video_state.current_t:
                self.video_state.jump(t)
            val = fs_cur.sample_scalar(x + 0.5, y + 0.5)
            self.ctx.variables["_read"] = val
            self._emit_trace("READ", {"x": x, "y": y, "t": t, "value": val})

        elif stmt.op == "WRITE":
            x, y, t = self.ctx.eval_coord(stmt.args[0])
            val = int(self.ctx.eval_expr(stmt.args[1])) & 0xFF
            if t != self.video_state.current_t:
                self.video_state.jump(t)
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < fs_cur.frame.w and 0 <= iy < fs_cur.frame.h:
                fs_cur.field_b[iy, ix] = val
                self._emit_trace("WRITE", {"x": x, "y": y, "t": t, "value": val})

    # --- Control Flow ---
    def _exec_control_stmt(self, stmt: ControlStmt):
        if stmt.op == "LOOP":
            # LOOP t0 t1 { body }
            # We need to get t0, t1 from context - they were pushed by parser
            t0 = self.ctx.variables.get("_loop_t0", 0)
            t1 = self.ctx.variables.get("_loop_t1", self.video_state.metadata.frame_count - 1)
            for t in range(t0, min(t1 + 1, self.video_state.metadata.frame_count)):
                self.video_state.jump(t)
                self._emit_trace("LOOP_ITER", {"t": t})
                self._execute_block(stmt.body)

        elif stmt.op == "WHILE":
            # WHILE condition { body }
            max_iters = 10000
            for _ in range(max_iters):
                cond_val = self.ctx.eval_expr(stmt.cond)
                if not cond_val:
                    break
                self._execute_block(stmt.body)

        elif stmt.op == "ON":
            # ON event { body } - register event handler
            event_name = str(stmt.cond) if stmt.cond else "UNKNOWN"
            self.ctx.event_handlers.setdefault(event_name, []).append(stmt)
            self._emit_trace("ON_REGISTER", {"event": event_name})

        elif stmt.op == "IF":
            cond_val = self.ctx.eval_expr(stmt.cond)
            if cond_val:
                self._execute_block(stmt.body)
            elif stmt.else_body:
                self._execute_block(stmt.else_body)

    # --- Emission ---
    def _exec_emit_stmt(self, stmt: EmitStmt):
        if stmt.op == "EMIT":
            event_name = str(self.ctx.eval_expr(stmt.args[0])) if stmt.args else "EMIT"
            payload = {}
            for i, arg in enumerate(stmt.args[1:]):
                payload[f"arg{i}"] = self.ctx.eval_expr(arg)
            self.video_state.emit_event(event_name, payload)
            self._emit_trace("EMIT", {"event": event_name, "payload": payload})

        elif stmt.op == "SNAPSHOT":
            snap = self.video_state.snapshot()
            self.ctx.variables["_snapshot"] = snap
            self._emit_trace("SNAPSHOT", snap)

    # --- Queries ---
    def _exec_query_stmt(self, stmt: QueryStmt):
        if stmt.op == "INSPECT":
            x, y, t = self.ctx.eval_coord(stmt.args[0])
            if t != self.video_state.current_t:
                self.video_state.jump(t)
            fs = self.video_state.current
            val = fs.sample_scalar(x + 0.5, y + 0.5)
            vx, vy = fs.sample_vector(x + 0.5, y + 0.5)
            result = {"x": x, "y": y, "t": t, "value": val, "vector": [vx, vy]}
            self.ctx.variables["_inspect"] = result
            self._emit_trace("INSPECT", result)

        elif stmt.op == "OBSERVE":
            x1, y1, x2, y2, t0, t1 = self.ctx.eval_region(stmt.args[0])
            if t0 != self.video_state.current_t:
                self.video_state.jump(t0)
            fs = self.video_state.current
            region_b = fs.field_b[y1:y2+1, x1:x2+1].copy()
            result = {"region": [x1, y1, x2, y2], "t": t0, "data": region_b.tolist()}
            self.ctx.variables["_observe"] = result
            self._emit_trace("OBSERVE", {"region": [x1, y1, x2, y2], "t": t0, "shape": region_b.shape})

        elif stmt.op == "TRACE":
            # Particle trace query
            pid = int(self.ctx.eval_expr(stmt.args[0]))
            self._emit_trace("TRACE_QUERY", {"pid": pid})

    def _emit_trace(self, event_type: str, payload: Dict[str, Any]):
        self.trace.append({
            "tick": len(self.trace),
            "t": self.video_state.current_t,
            "timestamp": self.video_state.current.frame.timestamp if self.video_state.current else 0,
            "type": event_type,
            "payload": payload,
        })


def run_program(video_path: str | Path, program_source: str, config: Optional[LangBusterConfig] = None) -> Dict[str, Any]:
    """Convenience: parse and run."""
    prog = parse(program_source)
    vm = LangBusterVM(video_path, config, prog)
    return vm.execute()


def run_file(video_path: str | Path, program_path: str | Path, config: Optional[LangBusterConfig] = None) -> Dict[str, Any]:
    """Convenience: run from file."""
    prog = parse_file(program_path)
    vm = LangBusterVM(video_path, config, prog)
    return vm.execute()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m langbuster.runtime <video> <program.lb>")
        sys.exit(1)

    result = run_file(sys.argv[1], sys.argv[2])
    import json
    print(json.dumps(result, indent=2, default=str))