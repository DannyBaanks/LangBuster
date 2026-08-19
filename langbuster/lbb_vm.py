#!/usr/bin/env python3
"""
LangBuster v0.1 — Binary Virtual Machine
Executes .lbb bytecode on video spacetime substrate.
"""

from __future__ import annotations
import hashlib
import random
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from .lbb_format import (
    Op, LBBProgram, LBBHeader, decode_instruction, disassemble,
    FLAG_DETERMINISTIC, FLAG_TRACE_ENABLED, FLAG_SELF_MODIFYING, FLAG_VIDEO_MMAP,
    compute_sha256, MAGIC, VERSION
)
from .video import VideoDecoder, Frame, VideoMetadata


# ============================================================
# VM State
# ============================================================

@dataclass
class VMState:
    """Complete VM state for deterministic replay."""
    # Registers (16 x 32-bit)
    regs: List[int] = field(default_factory=lambda: [0] * 16)

    # Program Counter
    pc: int = 0

    # Stack (grows down)
    stack: List[int] = field(default_factory=list)
    sp: int = 0  # Stack pointer (index into stack list)

    # Tape (Turing)
    tape: List[int] = field(default_factory=lambda: [0] * 65536)
    tape_head: int = 0

    # Video state
    video_decoder: Optional[VideoDecoder] = None
    video_metadata: Optional[VideoMetadata] = None
    frame_cache: Dict[int, Frame] = field(default_factory=dict)
    current_frame: Optional[Frame] = None
    current_t: int = 0
    field_r: Optional[np.ndarray] = None
    field_g: Optional[np.ndarray] = None
    field_b: Optional[np.ndarray] = None
    field_trace: Optional[np.ndarray] = None

    # Code section (mutable for self-modifying)
    code: bytearray = field(default_factory=bytearray)

    # Data section
    data: bytes = b""

    # Execution trace
    trace_events: List[Dict[str, Any]] = field(default_factory=list)
    trace_enabled: bool = True

    # Config
    max_frames: int = 0
    deterministic: bool = True
    seed: Optional[int] = None
    rng: random.Random = field(default_factory=random.Random)

    # Stats
    instruction_count: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        self.sp = len(self.stack)
        if self.seed is not None:
            self.rng.seed(self.seed)


# ============================================================
# Video Helpers
# ============================================================

def load_frame(state: VMState, t: int) -> Frame:
    """Load frame t into VM state."""
    if t < 0:
        t = 0
    if state.video_metadata and t >= state.video_metadata.frame_count:
        t = state.video_metadata.frame_count - 1

    if t not in state.frame_cache:
        if state.video_decoder:
            state.frame_cache[t] = state.video_decoder.frame_at(t)
        else:
            # Create blank frame
            w, h = 64, 64
            if state.video_metadata:
                w, h = state.video_metadata.width, state.video_metadata.height
            blank = Frame(index=t, timestamp=t/30.0, rgb=np.zeros((h, w, 3), dtype=np.uint8))
            state.frame_cache[t] = blank

    frame = state.frame_cache[t]
    state.current_frame = frame
    state.current_t = t

    # Decompose to FLOW channels
    arr = frame.rgb.astype(np.float32)
    state.field_r = arr[:, :, 0]
    state.field_g = arr[:, :, 1]
    state.field_b = frame.rgb[:, :, 2].copy().astype(np.uint8)
    if state.field_trace is None:
        state.field_trace = np.zeros((frame.h, frame.w), dtype=np.uint8)

    return frame


def sample_bilinear(field: np.ndarray, x: float, y: float) -> float:
    h, w = field.shape
    x = max(0.0, min(float(w - 1), x))
    y = max(0.0, min(float(h - 1), y))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]
    return (v00 * (1 - dx) * (1 - dy) +
            v10 * dx * (1 - dy) +
            v01 * (1 - dx) * dy +
            v11 * dx * dy)


def sample_vector(state: VMState, x: float, y: float) -> Tuple[float, float]:
    vx = (sample_bilinear(state.field_r, x, y) / 127.5) - 1.0
    vy = (sample_bilinear(state.field_g, x, y) / 127.5) - 1.0
    return vx, vy


def sample_scalar(state: VMState, x: float, y: float) -> int:
    h, w = state.field_b.shape
    x = max(0.0, min(float(w - 1), x))
    y = max(0.0, min(float(h - 1), y))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    v00 = state.field_b[y0, x0]
    v10 = state.field_b[y0, x1]
    v01 = state.field_b[y1, x0]
    v11 = state.field_b[y1, x1]
    val = (v00 * (1 - dx) * (1 - dy) +
           v10 * dx * (1 - dy) +
           v01 * (1 - dx) * dy +
           v11 * dx * dy)
    return int(val + 0.5) & 0xFF


def update_flags(state: VMState, result: int, carry: bool = False, overflow: bool = False):
    state.regs[15] = 0
    if result == 0:
        state.regs[15] |= 1 << 0  # Z
    if result & 0x80000000:
        state.regs[15] |= 1 << 1  # N
    if carry:
        state.regs[15] |= 1 << 2  # C
    if overflow:
        state.regs[15] |= 1 << 3  # V


def check_flag(state: VMState, flag: int) -> bool:
    return (state.regs[15] & (1 << flag)) != 0


# ============================================================
# Temporal Operations
# ============================================================

def op_seek(state: VMState, t: int):
    load_frame(state, t)


def op_next(state: VMState):
    if state.video_metadata and state.current_t + 1 < state.video_metadata.frame_count:
        load_frame(state, state.current_t + 1)
    else:
        state.regs[15] |= 1 << 4  # T flag = temporal boundary


def op_prev(state: VMState):
    if state.current_t > 0:
        load_frame(state, state.current_t - 1)
    else:
        state.regs[15] |= 1 << 4


def op_sample(state: VMState, ch: int, x: int, y: int):
    if state.current_frame is None:
        load_frame(state, 0)

    if ch == 0:  # R
        val = sample_bilinear(state.field_r, x + 0.5, y + 0.5)
        state.regs[8] = int(val) & 0xFF
    elif ch == 1:  # G
        val = sample_bilinear(state.field_g, x + 0.5, y + 0.5)
        state.regs[8] = int(val) & 0xFF
    elif ch == 2:  # B
        state.regs[8] = sample_scalar(state, x + 0.5, y + 0.5)
    elif ch == 3:  # Trace
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < state.field_trace.shape[0] and 0 <= ix < state.field_trace.shape[1]:
            state.regs[8] = state.field_trace[iy, ix]
        else:
            state.regs[8] = 0


def op_sample_r(state: VMState, ch: int):
    x, y, t = state.regs[8], state.regs[9], state.regs[10]
    if t != state.current_t:
        load_frame(state, t)
    op_sample(state, ch, x, y)


def op_vector(state: VMState, x: int, y: int):
    if state.current_frame is None:
        load_frame(state, 0)
    vx, vy = sample_vector(state, x + 0.5, y + 0.5)
    state.regs[8] = int((vx + 1.0) * 127.5) & 0xFF
    state.regs[9] = int((vy + 1.0) * 127.5) & 0xFF


def op_vector_r(state: VMState):
    x, y, t = state.regs[8], state.regs[9], state.regs[10]
    if t != state.current_t:
        load_frame(state, t)
    vx, vy = sample_vector(state, x + 0.5, y + 0.5)
    state.regs[8] = int((vx + 1.0) * 127.5) & 0xFF
    state.regs[9] = int((vy + 1.0) * 127.5) & 0xFF


def op_delta(state: VMState, x: int, y: int):
    if state.current_t == 0:
        state.regs[8] = 0
        return
    cur = sample_scalar(state, x + 0.5, y + 0.5)
    # Load previous frame
    prev_frame = load_frame(state, state.current_t - 1)
    prev_arr = prev_frame.rgb.astype(np.float32)
    prev_b = prev_arr[:, :, 2].astype(np.uint8)
    # Sample previous
    h, w = prev_b.shape
    x = max(0.0, min(float(w - 1), x + 0.5))
    y = max(0.0, min(float(h - 1), y + 0.5))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    v00 = prev_b[y0, x0]
    v10 = prev_b[y0, x1]
    v01 = prev_b[y1, x0]
    v11 = prev_b[y1, y1]
    prev_val = (v00 * (1 - dx) * (1 - dy) +
                v10 * dx * (1 - dy) +
                v01 * (1 - dx) * dy +
                v11 * dx * dy)
    prev_val = int(prev_val + 0.5) & 0xFF
    state.regs[8] = (cur - prev_val) & 0xFF
    # Restore current frame
    load_frame(state, state.current_t)


def op_delta_r(state: VMState):
    x, y = state.regs[8], state.regs[9]
    op_delta(state, x, y)


def op_motion(state: VMState, x1: int, y1: int, x2: int, y2: int):
    if state.current_t == 0:
        return
    # Simple motion: difference in vector field
    cur_r = state.field_r[y1:y2+1, x1:x2+1]
    cur_g = state.field_g[y1:y2+1, x1:x2+1]

    prev_frame = load_frame(state, state.current_t - 1)
    prev_arr = prev_frame.rgb.astype(np.float32)
    prev_r = prev_arr[y1:y2+1, x1:x2+1, 0]
    prev_g = prev_arr[y1:y2+1, x1:x2+1, 1]

    motion_x = cur_r - prev_r
    motion_y = cur_g - prev_g

    # Store to tape
    tape_idx = state.tape_head
    for i, (mx, my) in enumerate(zip(motion_x.flatten(), motion_y.flatten())):
        if tape_idx + i < len(state.tape):
            state.tape[tape_idx + i] = int((mx + 128) & 0xFF) | (int((my + 128) & 0xFF) << 8)
    state.tape_head += motion_x.size * 2
    load_frame(state, state.current_t)


def op_persist(state: VMState, x1: int, y1: int, x2: int, y2: int, t0: int, t1: int):
    persists = 1
    for t in range(t0, min(t1 + 1, state.video_metadata.frame_count if state.video_metadata else t1 + 1)):
        frame = load_frame(state, t)
        region_b = frame.rgb[y1:y2+1, x1:x2+1, 2]
        if np.count_nonzero(region_b) == 0:
            persists = 0
            break
    state.regs[8] = persists
    load_frame(state, state.current_t)


def op_appear(state: VMState, x1: int, y1: int, x2: int, y2: int):
    if state.current_t == 0:
        state.regs[8] = 0
        return
    cur_frame = state.current_frame
    prev_frame = load_frame(state, state.current_t - 1)

    cur_region = cur_frame.rgb[y1:y2+1, x1:x2+1, 2]
    prev_region = prev_frame.rgb[y1:y2+1, x1:x2+1, 2]

    appears = 1 if np.all(prev_region == 0) and np.any(cur_region != 0) else 0
    state.regs[8] = appears
    load_frame(state, state.current_t)


def op_disappear(state: VMState, x1: int, y1: int, x2: int, y2: int):
    if state.current_t == 0:
        state.regs[8] = 0
        return
    cur_frame = state.current_frame
    prev_frame = load_frame(state, state.current_t - 1)

    cur_region = cur_frame.rgb[y1:y2+1, x1:x2+1, 2]
    prev_region = prev_frame.rgb[y1:y2+1, x1:x2+1, 2]

    disappears = 1 if np.any(prev_region != 0) and np.all(cur_region == 0) else 0
    state.regs[8] = disappears
    load_frame(state, state.current_t)


def op_region_sum(state: VMState, x1: int, y1: int, x2: int, y2: int):
    if state.current_frame is None:
        load_frame(state, 0)
    region = state.current_frame.rgb[y1:y2+1, x1:x2+1, 2]
    state.regs[8] = int(np.sum(region)) & 0xFFFFFFFF


def op_region_cnt(state: VMState, x1: int, y1: int, x2: int, y2: int):
    if state.current_frame is None:
        load_frame(state, 0)
    region = state.current_frame.rgb[y1:y2+1, x1:x2+1, 2]
    state.regs[8] = int(np.count_nonzero(region))


def op_trace_put(state: VMState, x: int, y: int, val: int):
    if state.field_trace is not None:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < state.field_trace.shape[0] and 0 <= ix < state.field_trace.shape[1]:
            state.field_trace[iy, ix] = val & 0xFF


def op_trace_get(state: VMState, x: int, y: int):
    if state.field_trace is not None:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < state.field_trace.shape[0] and 0 <= ix < state.field_trace.shape[1]:
            state.regs[8] = state.field_trace[iy, ix]
        else:
            state.regs[8] = 0
    else:
        state.regs[8] = 0


def op_frame_hash(state: VMState, t: int):
    frame = load_frame(state, t)
    state.regs[8] = int(hashlib.sha256(frame.rgb[:, :, 2].tobytes()).hexdigest()[:8], 16)


def op_emit(state: VMState, event_id: int):
    if state.trace_enabled:
        event_name = state.get_string(event_id) if hasattr(state, 'get_string') else f"event_{event_id}"
        state.trace_events.append({
            "tick": state.instruction_count,
            "t": state.current_t,
            "timestamp": state.current_frame.timestamp if state.current_frame else 0,
            "type": "EMIT",
            "payload": {"event": event_name},
        })


def op_snapshot(state: VMState):
    if state.trace_enabled:
        snap = {
            "tick": state.instruction_count,
            "t": state.current_t,
            "regs": state.regs.copy(),
            "pc": state.pc,
            "sp": state.sp,
            "tape_head": state.tape_head,
            "field_b_hash": hashlib.sha256(state.field_b.tobytes()).hexdigest()[:16] if state.field_b is not None else None,
        }
        state.trace_events.append({
            "tick": state.instruction_count,
            "t": state.current_t,
            "timestamp": state.current_frame.timestamp if state.current_frame else 0,
            "type": "SNAPSHOT",
            "payload": snap,
        })


# ============================================================
# VM Core
# ============================================================

class LBBVM:
    """LangBuster Binary Virtual Machine."""

    def __init__(self, program: LBBProgram, video_path: Optional[str | Path] = None):
        self.program = program
        self.header = program.header
        self.code = bytearray(program.code)  # mutable
        self.data = program.data

        # Initialize state
        self.state = VMState(
            code=self.code,
            data=self.data,
            trace_enabled=bool(self.header.flags & FLAG_TRACE_ENABLED),
            deterministic=bool(self.header.flags & FLAG_DETERMINISTIC),
            seed=self.header.timestamp if self.header.flags & FLAG_DETERMINISTIC else None,
        )

        # Load video if provided
        if video_path:
            self.state.video_decoder = VideoDecoder(video_path)
            self.state.video_metadata = self.state.video_decoder.metadata
            # Verify hash matches
            if self.header.video_sha256 != b"\0" * 32:
                actual = compute_sha256(Path(video_path).read_bytes())
                if actual != self.header.video_sha256:
                    print(f"WARNING: Video hash mismatch! Expected {self.header.video_sha256.hex()}, got {actual.hex()}")

        # Set PC to entry point
        self.state.pc = self.header.entry_point

        # Load initial frame
        load_frame(self.state, 0)

        # Execution trace
        self.execution_trace: List[Dict[str, Any]] = []

    def get_string(self, offset: int) -> str:
        """Get null-terminated string from data section."""
        if offset >= len(self.data):
            return ""
        end = self.data.find(b"\0", offset)
        if end == -1:
            end = len(self.data)
        return self.data[offset:end].decode("utf-8", errors="replace")

    def read_code(self, addr: int, size: int) -> bytes:
        if addr + size > len(self.code):
            return b"\0" * size
        return bytes(self.code[addr:addr+size])

    def write_code(self, addr: int, data: bytes):
        if addr + len(data) <= len(self.code):
            self.code[addr:addr+len(data)] = data

    def fetch(self) -> Tuple[int, bytes, int]:
        """Fetch next instruction. Returns (opcode, instruction_bytes, next_pc)."""
        if self.state.pc >= len(self.code):
            return Op.HALT, b"\x01", len(self.code)
        opcode = self.code[self.state.pc]
        # Decode to get size
        try:
            inst, next_pc = decode_instruction(self.code, self.state.pc)
            inst_bytes = self.code[self.state.pc:next_pc]
        except:
            inst_bytes = bytes([opcode])
            next_pc = self.state.pc + 1
        return opcode, inst_bytes, next_pc

    def step(self) -> bool:
        """Execute one instruction. Returns True if continuing."""
        if self.state.pc >= len(self.code):
            return False

        opcode, inst_bytes, next_pc = self.fetch()
        self.state.instruction_count += 1

        # Trace instruction
        if self.state.trace_enabled:
            self.execution_trace.append({
                "tick": self.state.instruction_count,
                "pc": self.state.pc,
                "opcode": f"0x{opcode:02X}",
                "instruction": inst_bytes.hex(),
                "regs_before": self.state.regs.copy(),
            })

        # Execute
        try:
            self.execute_instruction(opcode, inst_bytes, next_pc)
        except Exception as e:
            if self.state.trace_enabled:
                self.state.trace_events.append({
                    "tick": self.state.instruction_count,
                    "t": self.state.current_t,
                    "type": "ERROR",
                    "payload": {"error": str(e), "pc": self.state.pc},
                })
            raise

        # Trace after
        if self.state.trace_enabled and self.execution_trace:
            self.execution_trace[-1]["regs_after"] = self.state.regs.copy()

        return True

    def execute_instruction(self, opcode: int, inst_bytes: bytes, next_pc: int):
        """Execute a single instruction."""
        # Decode operands from inst_bytes
        inst, _ = decode_instruction(inst_bytes, 0)
        operands = inst.operands
        self.state.pc = next_pc  # default next PC

        # Helper to get operand value
        def get_op(idx: int, default: int = 0) -> int:
            if idx >= len(operands):
                return default
            op = operands[idx]
            if isinstance(op, tuple):
                return op[0] if op else default
            return op

        # --- Control Flow ---
        if opcode == Op.NOP:
            pass

        elif opcode == Op.HALT:
            self.state.pc = len(self.code)  # stop

        elif opcode == Op.JMP:
            self.state.pc = get_op(0)

        elif opcode == Op.JZ:
            if check_flag(self.state, 0):  # Z flag
                self.state.pc = get_op(0)

        elif opcode == Op.JNZ:
            if not check_flag(self.state, 0):
                self.state.pc = get_op(0)

        elif opcode == Op.JN:
            if check_flag(self.state, 1):  # N flag
                self.state.pc = get_op(0)

        elif opcode == Op.JC:
            if check_flag(self.state, 2):  # C flag
                self.state.pc = get_op(0)

        elif opcode == Op.CALL:
            self.state.stack.append(self.state.pc)
            self.state.sp = len(self.state.stack)
            self.state.pc = get_op(0)

        elif opcode == Op.RET:
            if self.state.stack:
                self.state.pc = self.state.stack.pop()
                self.state.sp = len(self.state.stack)
            else:
                self.state.pc = len(self.code)  # halt

        elif opcode == Op.JMP_R:
            self.state.pc = self.state.regs[get_op(0)]

        elif opcode == Op.CALL_R:
            self.state.stack.append(self.state.pc)
            self.state.sp = len(self.state.stack)
            self.state.pc = self.state.regs[get_op(0)]

        # --- Arithmetic/Logic ---
        elif opcode == Op.ADD:
            dst, src = operands[0]
            a = self.state.regs[dst]
            b = self.state.regs[src]
            result = (a + b) & 0xFFFFFFFF
            carry = (a + b) > 0xFFFFFFFF
            self.state.regs[dst] = result
            update_flags(self.state, result, carry)

        elif opcode == Op.SUB:
            dst, src = operands[0]
            a = self.state.regs[dst]
            b = self.state.regs[src]
            result = (a - b) & 0xFFFFFFFF
            carry = a < b
            self.state.regs[dst] = result
            update_flags(self.state, result, carry)

        elif opcode == Op.MUL:
            dst, src = operands[0]
            a = self.state.regs[dst]
            b = self.state.regs[src]
            result = (a * b) & 0xFFFFFFFF
            self.state.regs[dst] = result
            update_flags(self.state, result)

        elif opcode == Op.DIV:
            dst, src = operands[0]
            a = self.state.regs[dst]
            b = self.state.regs[src]
            if b != 0:
                result = (a // b) & 0xFFFFFFFF
                self.state.regs[dst] = result
                update_flags(self.state, result)
            else:
                self.state.regs[15] |= 1 << 3  # V flag

        elif opcode == Op.MOD:
            dst, src = operands[0]
            a = self.state.regs[dst]
            b = self.state.regs[src]
            if b != 0:
                result = (a % b) & 0xFFFFFFFF
                self.state.regs[dst] = result
                update_flags(self.state, result)

        elif opcode == Op.AND:
            dst, src = operands[0]
            result = self.state.regs[dst] & self.state.regs[src]
            self.state.regs[dst] = result
            update_flags(self.state, result)

        elif opcode == Op.OR:
            dst, src = operands[0]
            result = self.state.regs[dst] | self.state.regs[src]
            self.state.regs[dst] = result
            update_flags(self.state, result)

        elif opcode == Op.XOR:
            dst, src = operands[0]
            result = self.state.regs[dst] ^ self.state.regs[src]
            self.state.regs[dst] = result
            update_flags(self.state, result)

        elif opcode == Op.NOT:
            dst = operands[0]
            result = (~self.state.regs[dst]) & 0xFFFFFFFF
            self.state.regs[dst] = result
            update_flags(self.state, result)

        elif opcode == Op.SHL:
            dst, src = operands[0]
            shift = self.state.regs[src] & 0x1F
            a = self.state.regs[dst]
            carry = (a >> (32 - shift)) & 1 if shift > 0 else 0
            result = (a << shift) & 0xFFFFFFFF
            self.state.regs[dst] = result
            update_flags(self.state, result, carry=bool(carry))

        elif opcode == Op.SHR:
            dst, src = operands[0]
            shift = self.state.regs[src] & 0x1F
            a = self.state.regs[dst]
            carry = (a >> (shift - 1)) & 1 if shift > 0 else 0
            result = a >> shift
            self.state.regs[dst] = result
            update_flags(self.state, result, carry=bool(carry))

        elif opcode == Op.INC:
            dst = operands[0]
            a = self.state.regs[dst]
            result = (a + 1) & 0xFFFFFFFF
            carry = a == 0xFFFFFFFF
            self.state.regs[dst] = result
            update_flags(self.state, result, carry)

        elif opcode == Op.DEC:
            dst = operands[0]
            a = self.state.regs[dst]
            result = (a - 1) & 0xFFFFFFFF
            carry = a == 0
            self.state.regs[dst] = result
            update_flags(self.state, result, carry)

        elif opcode == Op.NEG:
            dst = operands[0]
            a = self.state.regs[dst]
            result = (-a) & 0xFFFFFFFF
            self.state.regs[dst] = result
            update_flags(self.state, result)

        elif opcode == Op.CMP:
            dst, src = operands[0]
            a = self.state.regs[dst]
            b = self.state.regs[src]
            result = (a - b) & 0xFFFFFFFF
            update_flags(self.state, result, carry=(a < b))

        elif opcode == Op.TEST:
            dst, src = operands[0]
            result = self.state.regs[dst] & self.state.regs[src]
            update_flags(self.state, result)

        # --- Memory/Register ---
        elif opcode == Op.MOV:
            dst, src = operands[0]
            self.state.regs[dst] = self.state.regs[src]

        elif opcode == Op.LOAD:
            dst, addr = operands[0]
            if 0 <= addr < len(self.data):
                self.state.regs[dst] = struct.unpack("<I", self.data[addr:addr+4])[0]
            else:
                self.state.regs[dst] = 0

        elif opcode == Op.STORE:
            addr, src = operands[0]
            if 0 <= addr + 4 <= len(self.data):
                # Data section is immutable in this implementation
                pass

        elif opcode == Op.LOAD_I:
            ops = operands[0] if operands else (0, 0)
            if isinstance(ops, tuple):
                dst, imm = ops[0], ops[1] if len(ops) > 1 else 0
            else:
                dst, imm = ops, 0
            self.state.regs[dst] = imm & 0xFFFFFFFF

        elif opcode == Op.PUSH:
            src = operands[0]
            self.state.stack.append(self.state.regs[src])
            self.state.sp = len(self.state.stack)

        elif opcode == Op.POP:
            dst = operands[0]
            if self.state.stack:
                self.state.regs[dst] = self.state.stack.pop()
                self.state.sp = len(self.state.stack)
            else:
                self.state.regs[dst] = 0

        elif opcode == Op.PUSH_I:
            imm = operands[0]
            self.state.stack.append(imm & 0xFFFFFFFF)
            self.state.sp = len(self.state.stack)

        elif opcode == Op.PEEK:
            dst, offset = operands[0]
            idx = self.state.sp - 1 - offset
            if 0 <= idx < len(self.state.stack):
                self.state.regs[dst] = self.state.stack[idx]
            else:
                self.state.regs[dst] = 0

        elif opcode == Op.POKE:
            offset, src = operands[0]
            idx = self.state.sp - 1 - offset
            if 0 <= idx < len(self.state.stack):
                self.state.stack[idx] = self.state.regs[src]

        elif opcode == Op.XCHG:
            a, b = operands[0]
            self.state.regs[a], self.state.regs[b] = self.state.regs[b], self.state.regs[a]

        # --- Tape ---
        elif opcode == Op.TAPE_L:
            self.state.tape_head = max(0, self.state.tape_head - 1)

        elif opcode == Op.TAPE_R:
            self.state.tape_head = min(len(self.state.tape) - 1, self.state.tape_head + 1)

        elif opcode == Op.TAPE_READ:
            dst = operands[0]
            self.state.regs[dst] = self.state.tape[self.state.tape_head]

        elif opcode == Op.TAPE_WRITE:
            src = operands[0]
            self.state.tape[self.state.tape_head] = self.state.regs[src]

        elif opcode == Op.TAPE_JMP:
            self.state.tape_head = operands[0] % len(self.state.tape)

        elif opcode == Op.TAPE_COPY:
            length, addr = operands[0]
            for i in range(min(length, len(self.state.tape) - self.state.tape_head)):
                if addr + i < len(self.data):
                    self.state.tape[self.state.tape_head + i] = self.data[addr + i]

        elif opcode == Op.TAPE_FIND:
            val, max_len = operands[0]
            for i in range(min(max_len, len(self.state.tape) - self.state.tape_head)):
                if self.state.tape[self.state.tape_head + i] == val:
                    self.state.regs[8] = i
                    return
            self.state.regs[8] = -1

        # --- Spatiotemporal ---
        elif opcode == Op.SEEK:
            t = self.state.regs[10]
            op_seek(self.state, t)

        elif opcode == Op.NEXT:
            op_next(self.state)

        elif opcode == Op.PREV:
            op_prev(self.state)

        elif opcode == Op.SAMPLE:
            ch, x, y = operands[0]
            op_sample(self.state, ch, x, y)

        elif opcode == Op.SAMPLE_R:
            ch = operands[0]
            op_sample_r(self.state, ch)

        elif opcode == Op.VECTOR:
            x, y = operands[0]
            op_vector(self.state, x, y)

        elif opcode == Op.VECTOR_R:
            op_vector_r(self.state)

        elif opcode == Op.DELTA:
            x, y = operands[0]
            op_delta(self.state, x, y)

        elif opcode == Op.DELTA_R:
            op_delta_r(self.state)

        elif opcode == Op.MOTION:
            x1, y1, x2, y2 = operands[0]
            op_motion(self.state, x1, y1, x2, y2)

        elif opcode == Op.PERSIST:
            # Operands on stack: t1, t0 (pushed in reverse)
            t1 = self.state.stack.pop() if self.state.stack else 0
            t0 = self.state.stack.pop() if self.state.stack else 0
            x1, y1, x2, y2 = self.state.regs[8], self.state.regs[9], self.state.regs[10], self.state.regs[11]
            op_persist(self.state, x1, y1, x2, y2, t0, t1)

        elif opcode == Op.APPEAR:
            x1, y1, x2, y2 = self.state.regs[8], self.state.regs[9], self.state.regs[10], self.state.regs[11]
            op_appear(self.state, x1, y1, x2, y2)

        elif opcode == Op.DISAPPEAR:
            x1, y1, x2, y2 = self.state.regs[8], self.state.regs[9], self.state.regs[10], self.state.regs[11]
            op_disappear(self.state, x1, y1, x2, y2)

        elif opcode == Op.REGION_SUM:
            x1, y1, x2, y2 = self.state.regs[8], self.state.regs[9], self.state.regs[10], self.state.regs[11]
            op_region_sum(self.state, x1, y1, x2, y2)

        elif opcode == Op.REGION_CNT:
            x1, y1, x2, y2 = self.state.regs[8], self.state.regs[9], self.state.regs[10], self.state.regs[11]
            op_region_cnt(self.state, x1, y1, x2, y2)

        elif opcode == Op.TRACE_PUT:
            x, y, val = operands[0]
            op_trace_put(self.state, x, y, val)

        elif opcode == Op.TRACE_GET:
            x, y = operands[0]
            op_trace_get(self.state, x, y)

        elif opcode == Op.FRAME_HASH:
            t = operands[0]
            op_frame_hash(self.state, t)

        elif opcode == Op.EMIT:
            event_id = self.state.regs[0]
            op_emit(self.state, event_id)

        elif opcode == Op.SNAPSHOT:
            op_snapshot(self.state)

        # --- Self-Modifying Code ---
        elif opcode == Op.CODE_READ:
            dst, addr = operands[0]
            if 0 <= addr < len(self.code):
                self.state.regs[dst] = self.code[addr]

        elif opcode == Op.CODE_WRITE:
            addr, src = operands[0]
            if 0 <= addr < len(self.code):
                self.code[addr] = self.state.regs[src] & 0xFF

        elif opcode == Op.CODE_JMP:
            self.state.pc = operands[0]

        elif opcode == Op.PATCH:
            addr, opcode_val, op1, op2, op3 = operands[0]
            if 0 <= addr < len(self.code):
                self.code[addr] = opcode_val & 0xFF
                # Could patch more bytes for operands

        # --- Video I/O ---
        elif opcode == Op.VID_OPEN:
            pass  # Already opened

        elif opcode == Op.VID_CLOSE:
            pass

        elif opcode == Op.VID_INFO:
            if self.state.video_metadata:
                meta = self.state.video_metadata
                self.state.stack.extend([
                    meta.width, meta.height, int(meta.fps),
                    meta.frame_count, int(meta.duration_sec * 1000)
                ])
                self.state.sp = len(self.state.stack)

        elif opcode == Op.VID_FRAMES:
            self.state.regs[8] = self.state.video_metadata.frame_count if self.state.video_metadata else 0

        # --- Crypto ---
        elif opcode == Op.SHA256:
            addr, length, dst = operands[0]
            data = self.data[addr:addr+length] if addr + length <= len(self.data) else b""
            h = hashlib.sha256(data).digest()
            self.state.regs[dst] = int.from_bytes(h[:4], "little")

        elif opcode == Op.RAND:
            dst = operands[0]
            self.state.regs[dst] = self.state.rng.randint(0, 0xFFFFFFFF)

        else:
            # Unknown opcode - NOP
            pass

    def run(self, max_instructions: int = 1000000) -> Dict[str, Any]:
        """Run VM until HALT or max instructions."""
        self.state.start_time = time.time()
        self.state.instruction_count = 0

        while self.step() and self.state.instruction_count < max_instructions:
            if self.state.pc >= len(self.code):
                break

        elapsed = time.time() - self.state.start_time

        return {
            "halted": self.state.pc >= len(self.code),
            "instructions": self.state.instruction_count,
            "elapsed_sec": elapsed,
            "final_pc": self.state.pc,
            "final_regs": self.state.regs.copy(),
            "final_t": self.state.current_t,
            "trace_events": self.state.trace_events,
            "execution_trace": self.execution_trace,
        }


def run_lbb(
    lbb_path: str | Path,
    video_path: Optional[str | Path] = None,
    max_instructions: int = 1000000,
) -> Dict[str, Any]:
    """Convenience: load and run .lbb program."""
    program = LBBProgram.load(lbb_path)
    vm = LBBVM(program, video_path)
    return vm.run(max_instructions)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m langbuster.lbb_vm <program.lbb> [video.mp4]")
        sys.exit(1)

    lbb_path = sys.argv[1]
    video_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_lbb(lbb_path, video_path)
    print(f"Halted: {result['halted']}")
    print(f"Instructions: {result['instructions']}")
    print(f"Time: {result['elapsed_sec']:.3f}s")
    print(f"Final PC: {result['final_pc']}")
    print(f"Final T: {result['final_t']}")
    print(f"Regs: {result['final_regs']}")
    print(f"Trace events: {len(result['trace_events'])}")