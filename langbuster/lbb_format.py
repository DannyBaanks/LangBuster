#!/usr/bin/env python3
"""
LangBuster v0.1 — Binary Format (.lbb)
Bytecode serialization, header, instruction encoding.
"""

from __future__ import annotations
import hashlib
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from io import BytesIO

# ============================================================
# Constants
# ============================================================

MAGIC = b"LBB\0"
VERSION = 0x01

# Flags
FLAG_DETERMINISTIC = 0x01
FLAG_TRACE_ENABLED = 0x02
FLAG_SELF_MODIFYING = 0x04
FLAG_VIDEO_MMAP = 0x08

# Opcodes
class Op:
    # Control Flow
    NOP = 0x00
    HALT = 0x01
    JMP = 0x02
    JZ = 0x03
    JNZ = 0x04
    JN = 0x05
    JC = 0x06
    CALL = 0x07
    RET = 0x08
    JMP_R = 0x09
    CALL_R = 0x0A

    # Arithmetic/Logic
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14
    AND = 0x15
    OR = 0x16
    XOR = 0x17
    NOT = 0x18
    SHL = 0x19
    SHR = 0x1A
    INC = 0x1B
    DEC = 0x1C
    NEG = 0x1D
    CMP = 0x1E
    TEST = 0x1F

    # Memory/Register
    MOV = 0x20
    LOAD = 0x21
    STORE = 0x22
    LOAD_I = 0x23
    PUSH = 0x24
    POP = 0x25
    PUSH_I = 0x26
    PEEK = 0x27
    POKE = 0x28
    XCHG = 0x29

    # Tape (Turing)
    TAPE_L = 0x30
    TAPE_R = 0x31
    TAPE_READ = 0x32
    TAPE_WRITE = 0x33
    TAPE_JMP = 0x34
    TAPE_COPY = 0x35
    TAPE_FIND = 0x36

    # Spatiotemporal Native
    SEEK = 0x40
    NEXT = 0x41
    PREV = 0x42
    SAMPLE = 0x43
    SAMPLE_R = 0x44
    VECTOR = 0x45
    VECTOR_R = 0x46
    DELTA = 0x47
    DELTA_R = 0x48
    MOTION = 0x49
    PERSIST = 0x4A
    APPEAR = 0x4B
    DISAPPEAR = 0x4C
    REGION_SUM = 0x4D
    REGION_CNT = 0x4E
    TRACE_PUT = 0x4F
    TRACE_GET = 0x50
    FRAME_HASH = 0x51
    EMIT = 0x52
    SNAPSHOT = 0x53

    # Self-Modifying Code
    CODE_READ = 0x60
    CODE_WRITE = 0x61
    CODE_JMP = 0x62
    PATCH = 0x63

    # Video I/O
    VID_OPEN = 0x70
    VID_CLOSE = 0x71
    VID_INFO = 0x72
    VID_FRAMES = 0x73

    # Crypto
    SHA256 = 0x80
    RAND = 0x81


# Operand encoding helpers
def encode_reg(reg: int) -> int:
    """Encode register (0-15) as 4-bit nibble."""
    return reg & 0xF

def decode_reg(byte: int, high: bool = True) -> int:
    """Decode register from nibble."""
    return (byte >> 4) & 0xF if high else byte & 0xF

def encode_two_regs(dst: int, src: int) -> int:
    """Encode two registers in one byte: high nibble=dst, low=src."""
    return ((dst & 0xF) << 4) | (src & 0xF)

def decode_two_regs(byte: int) -> Tuple[int, int]:
    return (byte >> 4) & 0xF, byte & 0xF


# ============================================================
# Instruction Representation
# ============================================================

@dataclass
class Instruction:
    """Decoded instruction."""
    opcode: int
    operands: List[Any] = field(default_factory=list)
    offset: int = 0          # byte offset in code section
    size: int = 1            # total bytes
    mnemonic: str = ""

    def __bytes__(self) -> bytes:
        return encode_instruction(self)

    def __repr__(self):
        ops = " ".join(str(o) for o in self.operands)
        return f"{self.mnemonic} {ops}".strip()


# Opcode -> (mnemonic, operand_pattern)
# operand_pattern: list of types: 'r'=reg, 'a16'=addr16, 'i16'=imm16, 'i32'=imm32, 'rr'=two_regs
OPCODE_INFO = {
    Op.NOP: ("NOP", []),
    Op.HALT: ("HALT", []),
    Op.JMP: ("JMP", ["a16"]),
    Op.JZ: ("JZ", ["a16"]),
    Op.JNZ: ("JNZ", ["a16"]),
    Op.JN: ("JN", ["a16"]),
    Op.JC: ("JC", ["a16"]),
    Op.CALL: ("CALL", ["a16"]),
    Op.RET: ("RET", []),
    Op.JMP_R: ("JMP_R", ["r"]),
    Op.CALL_R: ("CALL_R", ["r"]),

    Op.ADD: ("ADD", ["rr"]),
    Op.SUB: ("SUB", ["rr"]),
    Op.MUL: ("MUL", ["rr"]),
    Op.DIV: ("DIV", ["rr"]),
    Op.MOD: ("MOD", ["rr"]),
    Op.AND: ("AND", ["rr"]),
    Op.OR: ("OR", ["rr"]),
    Op.XOR: ("XOR", ["rr"]),
    Op.NOT: ("NOT", ["r"]),
    Op.SHL: ("SHL", ["rr"]),
    Op.SHR: ("SHR", ["rr"]),
    Op.INC: ("INC", ["r"]),
    Op.DEC: ("DEC", ["r"]),
    Op.NEG: ("NEG", ["r"]),
    Op.CMP: ("CMP", ["rr"]),
    Op.TEST: ("TEST", ["rr"]),

    Op.MOV: ("MOV", ["rr"]),
    Op.LOAD: ("LOAD", ["r", "a16"]),
    Op.STORE: ("STORE", ["a16", "r"]),
    Op.LOAD_I: ("LOAD_I", ["r", "i16"]),
    Op.PUSH: ("PUSH", ["r"]),
    Op.POP: ("POP", ["r"]),
    Op.PUSH_I: ("PUSH_I", ["i16"]),
    Op.PEEK: ("PEEK", ["r", "i16"]),
    Op.POKE: ("POKE", ["i16", "r"]),
    Op.XCHG: ("XCHG", ["rr"]),

    Op.TAPE_L: ("TAPE_L", []),
    Op.TAPE_R: ("TAPE_R", []),
    Op.TAPE_READ: ("TAPE_READ", ["r"]),
    Op.TAPE_WRITE: ("TAPE_WRITE", ["r"]),
    Op.TAPE_JMP: ("TAPE_JMP", ["a16"]),
    Op.TAPE_COPY: ("TAPE_COPY", ["i16", "a16"]),
    Op.TAPE_FIND: ("TAPE_FIND", ["i16", "i16"]),

    Op.SEEK: ("SEEK", ["i16"]),
    Op.NEXT: ("NEXT", []),
    Op.PREV: ("PREV", []),
    Op.SAMPLE: ("SAMPLE", ["i16", "i16", "i16"]),  # ch, x, y
    Op.SAMPLE_R: ("SAMPLE_R", ["i16"]),  # ch
    Op.VECTOR: ("VECTOR", ["i16", "i16"]),
    Op.VECTOR_R: ("VECTOR_R", []),
    Op.DELTA: ("DELTA", ["i16", "i16"]),
    Op.DELTA_R: ("DELTA_R", []),
    Op.MOTION: ("MOTION", ["i16", "i16", "i16", "i16"]),
    Op.PERSIST: ("PERSIST", ["i16", "i16", "i16", "i16", "i16", "i16"]),
    Op.APPEAR: ("APPEAR", ["i16", "i16", "i16", "i16"]),
    Op.DISAPPEAR: ("DISAPPEAR", ["i16", "i16", "i16", "i16"]),
    Op.REGION_SUM: ("REGION_SUM", ["i16", "i16", "i16", "i16"]),
    Op.REGION_CNT: ("REGION_CNT", ["i16", "i16", "i16", "i16"]),
    Op.TRACE_PUT: ("TRACE_PUT", ["i16", "i16", "i16"]),
    Op.TRACE_GET: ("TRACE_GET", ["i16", "i16"]),
    Op.FRAME_HASH: ("FRAME_HASH", ["i16"]),
    Op.EMIT: ("EMIT", ["i16"]),
    Op.SNAPSHOT: ("SNAPSHOT", []),

    Op.CODE_READ: ("CODE_READ", ["r", "a16"]),
    Op.CODE_WRITE: ("CODE_WRITE", ["a16", "r"]),
    Op.CODE_JMP: ("CODE_JMP", ["a16"]),
    Op.PATCH: ("PATCH", ["a16", "i16", "i16", "i16", "i16"]),  # addr, opcode, op1, op2, op3

    Op.VID_OPEN: ("VID_OPEN", ["a16"]),  # pointer to string in data
    Op.VID_CLOSE: ("VID_CLOSE", []),
    Op.VID_INFO: ("VID_INFO", []),
    Op.VID_FRAMES: ("VID_FRAMES", []),

    Op.SHA256: ("SHA256", ["a16", "i16", "r"]),
    Op.RAND: ("RAND", ["r"]),
}


def encode_instruction(inst: Instruction) -> bytes:
    """Encode instruction to bytes."""
    buf = BytesIO()
    buf.write(bytes([inst.opcode]))

    info = OPCODE_INFO.get(inst.opcode)
    if not info:
        return buf.getvalue()

    _, pattern = info
    operands = inst.operands

    for i, ptype in enumerate(pattern):
        val = operands[i] if i < len(operands) else 0

        if ptype == "r":
            buf.write(bytes([encode_reg(val)]))
        elif ptype == "rr":
            buf.write(bytes([encode_two_regs(val[0], val[1])]))
        elif ptype == "a16":
            buf.write(struct.pack("<H", val & 0xFFFF))
        elif ptype == "i16":
            buf.write(struct.pack("<h", val & 0xFFFF))
        elif ptype == "i32":
            buf.write(struct.pack("<i", val & 0xFFFFFFFF))

    return buf.getvalue()


def decode_instruction(data: bytes, offset: int = 0) -> Tuple[Instruction, int]:
    """Decode instruction from bytes. Returns (instruction, next_offset)."""
    if offset >= len(data):
        raise ValueError("Unexpected end of code")

    opcode = data[offset]
    info = OPCODE_INFO.get(opcode, (f"UNKNOWN_{opcode:02X}", []))
    mnemonic, pattern = info

    inst = Instruction(opcode=opcode, offset=offset, mnemonic=mnemonic)
    pos = offset + 1

    for ptype in pattern:
        if ptype == "r":
            if pos >= len(data): break
            inst.operands.append(decode_reg(data[pos], high=True))
            pos += 1
        elif ptype == "rr":
            if pos >= len(data): break
            dst, src = decode_two_regs(data[pos])
            inst.operands.append((dst, src))
            pos += 1
        elif ptype == "a16":
            if pos + 2 > len(data): break
            inst.operands.append(struct.unpack("<H", data[pos:pos+2])[0])
            pos += 2
        elif ptype == "i16":
            if pos + 2 > len(data): break
            inst.operands.append(struct.unpack("<h", data[pos:pos+2])[0])
            pos += 2
        elif ptype == "i32":
            if pos + 4 > len(data): break
            inst.operands.append(struct.unpack("<i", data[pos:pos+4])[0])
            pos += 4

    inst.size = pos - offset
    return inst, pos


def disassemble(code: bytes, base_offset: int = 0) -> List[Instruction]:
    """Disassemble entire code section."""
    instructions = []
    pos = 0
    while pos < len(code):
        inst, pos = decode_instruction(code, pos)
        inst.offset += base_offset
        instructions.append(inst)
    return instructions


# ============================================================
# LBB File Format
# ============================================================

@dataclass
class LBBHeader:
    magic: bytes = MAGIC
    version: int = VERSION
    flags: int = 0
    entry_point: int = 0
    code_size: int = 0
    data_size: int = 0
    video_sha256: bytes = field(default_factory=lambda: b"\0" * 32)
    program_sha256: bytes = field(default_factory=lambda: b"\0" * 32)
    timestamp: int = field(default_factory=lambda: int(time.time()))
    reserved: bytes = field(default_factory=lambda: b"\0" * 8)

    HEADER_SIZE = 98

    def pack(self) -> bytes:
        return struct.pack(
            "<4s B B I I I 32s 32s Q 8s",
            self.magic,
            self.version,
            self.flags,
            self.entry_point,
            self.code_size,
            self.data_size,
            self.video_sha256,
            self.program_sha256,
            self.timestamp,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "LBBHeader":
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Header too small")
        magic, version, flags, entry_point, code_size, data_size, \
        video_sha, prog_sha, timestamp, reserved = struct.unpack(
            "<4s B B I I I 32s 32s Q 8s", data[:cls.HEADER_SIZE]
        )
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")
        return cls(
            magic=magic, version=version, flags=flags,
            entry_point=entry_point, code_size=code_size, data_size=data_size,
            video_sha256=video_sha, program_sha256=prog_sha,
            timestamp=timestamp, reserved=reserved,
        )


@dataclass
class LBBProgram:
    """Complete LBB program."""
    header: LBBHeader
    code: bytes
    data: bytes
    labels: Dict[str, int] = field(default_factory=dict)  # label -> code offset
    strings: Dict[int, str] = field(default_factory=dict)  # data offset -> string

    def save(self, path: str | Path):
        """Write .lbb file."""
        with open(path, "wb") as f:
            f.write(self.header.pack())
            f.write(self.code)
            f.write(self.data)

    @classmethod
    def load(cls, path: str | Path) -> "LBBProgram":
        """Load .lbb file."""
        with open(path, "rb") as f:
            data = f.read()

        header = LBBHeader.unpack(data[:LBBHeader.HEADER_SIZE])
        code_start = LBBHeader.HEADER_SIZE
        code_end = code_start + header.code_size
        data_start = code_end
        data_end = data_start + header.data_size

        code = data[code_start:code_end]
        data_section = data[data_start:data_end]

        return cls(header=header, code=code, data=data_section)

    def disassemble(self) -> List[Instruction]:
        return disassemble(self.code)

    def get_string(self, offset: int) -> str:
        """Extract null-terminated string from data section."""
        if offset >= len(self.data):
            return ""
        end = self.data.find(b"\0", offset)
        if end == -1:
            end = len(self.data)
        return self.data[offset:end].decode("utf-8", errors="replace")


def compute_sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def create_lbb(
    code: bytes,
    data: bytes = b"",
    entry_point: int = 0,
    flags: int = FLAG_DETERMINISTIC | FLAG_TRACE_ENABLED,
    video_sha256: bytes = b"\0" * 32,
    program_source: bytes = b"",
) -> LBBProgram:
    """Factory to create LBB program."""
    header = LBBHeader(
        flags=flags,
        entry_point=entry_point,
        code_size=len(code),
        data_size=len(data),
        video_sha256=video_sha256,
        program_sha256=compute_sha256(program_source),
    )
    return LBBProgram(header=header, code=code, data=data)


if __name__ == "__main__":
    # Quick test
    code = bytes([
        Op.LOAD_I, 0x00, 0x00,  # LOAD_I R0, 0
        Op.LOAD_I, 0x10, 0x27,  # LOAD_I R1, 10000
        Op.ADD, 0x01,           # ADD R0, R1
        Op.HALT,                # HALT
    ])
    prog = create_lbb(code, entry_point=0)
    prog.save("test.lbb")

    loaded = LBBProgram.load("test.lbb")
    for inst in loaded.disassemble():
        print(f"{inst.offset:04X}: {inst}")