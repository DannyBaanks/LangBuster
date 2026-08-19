#!/usr/bin/env python3
"""
LangBuster v0.1 — Compiler
AST (.lb) → Bytecode (.lbb)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .grammar import (
    Program, FrameStmt, TemporalStmt, SpatialStmt, ControlStmt,
    EmitStmt, QueryStmt, CoordExpr, RegionExpr, BinaryExpr, UnaryExpr,
    NumberLiteral, StringLiteral, IdentExpr, ParticleExpr, ASTNode,
    parse, parse_file
)
from .lbb_format import (
    Op, encode_instruction, create_lbb, LBBProgram, compute_sha256,
    FLAG_DETERMINISTIC, FLAG_TRACE_ENABLED, FLAG_SELF_MODIFYING
)


# ============================================================
# Compiler State
# ============================================================

@dataclass
class CompilerState:
    code: List[int] = field(default_factory=list)      # bytecode
    data: List[int] = field(default_factory=list)      # data section
    labels: Dict[str, int] = field(default_factory=dict)  # label -> code offset
    strings: Dict[str, int] = field(default_factory=dict) # string -> data offset
    label_refs: List[Tuple[str, int]] = field(default_factory=list)  # (label, code_offset)
    string_refs: List[Tuple[str, int]] = field(default_factory=list) # (string, code_offset where addr needed)
    variables: Dict[str, int] = field(default_factory=dict)  # var_name -> register
    next_reg: int = 0
    temp_regs: List[int] = field(default_factory=list)  # available temp registers
    loop_stack: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def emit(self, *bytes_vals: int):
        """Emit raw bytes to code."""
        for b in bytes_vals:
            self.code.append(b & 0xFF)

    def emit_inst(self, opcode: int, *operands):
        """Emit instruction with operands."""
        from .lbb_format import Instruction
        inst = Instruction(opcode=opcode, operands=list(operands))
        self.code.extend(encode_instruction(inst))

    def reserve_label(self, name: str) -> int:
        """Create a forward reference to a label."""
        offset = len(self.code)
        self.label_refs.append((name, offset))
        return offset

    def define_label(self, name: str):
        """Define label at current code position."""
        self.labels[name] = len(self.code)

    def get_string_offset(self, s: str) -> int:
        """Get or create string in data section, return offset."""
        if s in self.strings:
            return self.strings[s]
        offset = len(self.data)
        self.data.extend(s.encode("utf-8"))
        self.data.append(0)  # null terminator
        self.strings[s] = offset
        return offset

    def alloc_reg(self) -> int:
        """Allocate a register (R0-R7 for general, R8-R11 for temporal)."""
        if self.temp_regs:
            return self.temp_regs.pop()
        if self.next_reg < 8:
            reg = self.next_reg
            self.next_reg += 1
            return reg
        # Spill to stack if needed
        self.errors.append("Register pressure: out of registers")
        return 0

    def free_reg(self, reg: int):
        if 0 <= reg < 8:
            self.temp_regs.append(reg)


# ============================================================
# Expression Compilation
# ============================================================

def compile_expr(state: CompilerState, expr: ASTNode) -> int:
    """Compile expression, return register holding result."""
    if isinstance(expr, NumberLiteral):
        reg = state.alloc_reg()
        val = int(expr.value)
        state.emit_inst(Op.LOAD_I, reg, val)
        return reg

    if isinstance(expr, StringLiteral):
        reg = state.alloc_reg()
        offset = state.get_string_offset(expr.value)
        state.emit_inst(Op.LOAD_I, reg, offset)
        return reg

    if isinstance(expr, IdentExpr):
        # Variable lookup
        if expr.name in state.variables:
            return state.variables[expr.name]
        # Could be a register name like "R0"
        if expr.name.startswith("R") and expr.name[1:].isdigit():
            return int(expr.name[1:])
        # Unknown identifier -> allocate reg and load 0
        reg = state.alloc_reg()
        state.emit_inst(Op.LOAD_I, reg, 0)
        return reg

    if isinstance(expr, BinaryExpr):
        return compile_binary(state, expr)

    if isinstance(expr, UnaryExpr):
        return compile_unary(state, expr)

    if isinstance(expr, CoordExpr):
        # For coordinates, we typically need x, y in registers
        # Compile x and y, leave in temporal registers R8, R9
        x_reg = compile_expr(state, expr.x)
        y_reg = compile_expr(state, expr.y)
        state.emit_inst(Op.MOV, (8, x_reg))  # R8 = x
        state.emit_inst(Op.MOV, (9, y_reg))  # R9 = y
        if expr.t:
            t_reg = compile_expr(state, expr.t)
            state.emit_inst(Op.MOV, (10, t_reg))  # R10 = t
        else:
            state.emit_inst(Op.LOAD_I, 10, 0)  # default t=0
        state.free_reg(x_reg)
        state.free_reg(y_reg)
        return 8  # Return R8 as base

    if isinstance(expr, RegionExpr):
        # Compile region coords to R8-R11
        x1_reg = compile_expr(state, expr.x1)
        y1_reg = compile_expr(state, expr.y1)
        x2_reg = compile_expr(state, expr.x2)
        y2_reg = compile_expr(state, expr.y2)
        state.emit_inst(Op.MOV, (8, x1_reg))
        state.emit_inst(Op.MOV, (9, y1_reg))
        state.emit_inst(Op.MOV, (10, x2_reg))
        state.emit_inst(Op.MOV, (11, y2_reg))
        if expr.t0:
            t0_reg = compile_expr(state, expr.t0)
            # Store t0/t1 in memory or use more regs
            state.emit_inst(Op.PUSH, t0_reg)
        if expr.t1:
            t1_reg = compile_expr(state, expr.t1)
            state.emit_inst(Op.PUSH, t1_reg)
        state.free_reg(x1_reg); state.free_reg(y1_reg)
        state.free_reg(x2_reg); state.free_reg(y2_reg)
        return 8

    if isinstance(expr, ParticleExpr):
        pid_reg = compile_expr(state, expr.pid)
        return pid_reg

    # Default
    reg = state.alloc_reg()
    state.emit_inst(Op.LOAD_I, reg, 0)
    return reg


def compile_binary(state: CompilerState, expr: BinaryExpr) -> int:
    left_reg = compile_expr(state, expr.left)
    right_reg = compile_expr(state, expr.right)
    dest_reg = left_reg  # reuse left as destination

    op_map = {
        "PLUS": Op.ADD, "MINUS": Op.SUB, "STAR": Op.MUL, "SLASH": Op.DIV,
        "MODULO": Op.MOD, "EQ": Op.CMP, "NEQ": Op.CMP, "LT": Op.CMP,
        "GT": Op.CMP, "LE": Op.CMP, "GE": Op.CMP,
        "AND": Op.AND, "OR": Op.OR, "XOR": Op.XOR,
    }

    if expr.op == "ASSIGN":
        # Assignment: left must be IdentExpr
        if isinstance(expr.left, IdentExpr):
            state.variables[expr.left.name] = right_reg
            state.free_reg(left_reg)
            return right_reg
        state.errors.append(f"Invalid assignment target: {expr.left}")
        return right_reg

    opcode = op_map.get(expr.op)
    if opcode:
        state.emit_inst(opcode, (left_reg, right_reg))
        state.free_reg(right_reg)
        return left_reg

    state.errors.append(f"Unknown binary operator: {expr.op}")
    state.free_reg(right_reg)
    return left_reg


def compile_unary(state: CompilerState, expr: UnaryExpr) -> int:
    reg = compile_expr(state, expr.expr)
    if expr.op == "MINUS":
        state.emit_inst(Op.NEG, reg)
    elif expr.op == "PLUS":
        pass  # no-op
    return reg


# ============================================================
# Statement Compilation
# ============================================================

def compile_stmt(state: CompilerState, stmt: ASTNode):
    if isinstance(stmt, FrameStmt):
        compile_frame_stmt(state, stmt)
    elif isinstance(stmt, TemporalStmt):
        compile_temporal_stmt(state, stmt)
    elif isinstance(stmt, SpatialStmt):
        compile_spatial_stmt(state, stmt)
    elif isinstance(stmt, ControlStmt):
        compile_control_stmt(state, stmt)
    elif isinstance(stmt, EmitStmt):
        compile_emit_stmt(state, stmt)
    elif isinstance(stmt, QueryStmt):
        compile_query_stmt(state, stmt)
    elif isinstance(stmt, BinaryExpr) and stmt.op == "ASSIGN":
        compile_expr(state, stmt)  # Assignment handled in expr
    else:
        # Expression statement
        reg = compile_expr(state, stmt)
        state.free_reg(reg)


def compile_frame_stmt(state: CompilerState, stmt: FrameStmt):
    if stmt.op == "FRAME":
        t_reg = compile_expr(state, stmt.arg) if stmt.arg else None
        if t_reg is not None:
            state.emit_inst(Op.MOV, (10, t_reg))  # R10 = t
            state.emit_inst(Op.SEEK)
            state.free_reg(t_reg)
        else:
            state.emit_inst(Op.LOAD_I, 10, 0)
            state.emit_inst(Op.SEEK)
    elif stmt.op == "NEXT":
        state.emit_inst(Op.NEXT)
    elif stmt.op == "PREV":
        state.emit_inst(Op.PREV)
    elif stmt.op == "JUMP":
        t_reg = compile_expr(state, stmt.arg)
        state.emit_inst(Op.MOV, (10, t_reg))
        state.emit_inst(Op.SEEK)
        state.free_reg(t_reg)


def compile_temporal_stmt(state: CompilerState, stmt: TemporalStmt):
    args = stmt.args

    if stmt.op == "DELTA":
        if len(args) >= 1:
            # Expects coord in R8, R9, R10
            compile_expr(state, args[0])
            state.emit_inst(Op.DELTA)
        else:
            state.emit_inst(Op.DELTA_R)

    elif stmt.op == "MOTION":
        if len(args) >= 1:
            compile_expr(state, args[0])  # sets up R8-R11
            state.emit_inst(Op.MOTION)

    elif stmt.op == "TRACE":
        if len(args) >= 1:
            pid_reg = compile_expr(state, args[0])
            # TRACE_PUT or custom - simplified
            state.free_reg(pid_reg)

    elif stmt.op == "PERSIST":
        if len(args) >= 3:
            compile_expr(state, args[0])  # region -> R8-R11
            t0_reg = compile_expr(state, args[1])
            t1_reg = compile_expr(state, args[2])
            # Need to push t0, t1 for PERSIST opcode
            state.emit_inst(Op.PUSH, t0_reg)
            state.emit_inst(Op.PUSH, t1_reg)
            state.emit_inst(Op.PERSIST)
            state.free_reg(t0_reg); state.free_reg(t1_reg)

    elif stmt.op == "APPEAR":
        if len(args) >= 1:
            compile_expr(state, args[0])
            state.emit_inst(Op.APPEAR)

    elif stmt.op == "DISAPPEAR":
        if len(args) >= 1:
            compile_expr(state, args[0])
            state.emit_inst(Op.DISAPPEAR)

    # Binary VM / Tape instructions
    elif stmt.op == "TAPE_L":
        state.emit_inst(Op.TAPE_L)

    elif stmt.op == "TAPE_R":
        state.emit_inst(Op.TAPE_R)

    elif stmt.op == "TAPE_READ":
        if len(args) >= 1:
            reg = compile_expr(state, args[0])
            # TAPE_READ expects register in operand
            state.emit_inst(Op.TAPE_READ, reg)
            state.free_reg(reg)

    elif stmt.op == "TAPE_WRITE":
        if len(args) >= 1:
            reg = compile_expr(state, args[0])
            state.emit_inst(Op.TAPE_WRITE, reg)
            state.free_reg(reg)

    elif stmt.op == "TAPE_JMP":
        if len(args) >= 1:
            reg = compile_expr(state, args[0])
            state.emit_inst(Op.TAPE_JMP, reg)
            state.free_reg(reg)

    elif stmt.op == "LOAD_I":
        if len(args) >= 2:
            reg = compile_expr(state, args[0])
            imm = compile_expr(state, args[1])
            state.emit_inst(Op.LOAD_I, reg, imm)
            state.free_reg(reg)
            state.free_reg(imm)

    elif stmt.op == "MOV":
        if len(args) >= 2:
            dst = compile_expr(state, args[0])
            src = compile_expr(state, args[1])
            state.emit_inst(Op.MOV, (dst, src))
            state.free_reg(dst)
            state.free_reg(src)

    elif stmt.op == "ADD":
        if len(args) >= 2:
            dst = compile_expr(state, args[0])
            src = compile_expr(state, args[1])
            state.emit_inst(Op.ADD, (dst, src))
            state.free_reg(dst)
            state.free_reg(src)

    elif stmt.op == "SUB":
        if len(args) >= 2:
            dst = compile_expr(state, args[0])
            src = compile_expr(state, args[1])
            state.emit_inst(Op.SUB, (dst, src))
            state.free_reg(dst)
            state.free_reg(src)

    elif stmt.op == "JZ":
        if len(args) >= 1:
            addr = compile_expr(state, args[0])
            state.emit_inst(Op.JZ, addr)
            state.free_reg(addr)

    elif stmt.op == "JMP":
        if len(args) >= 1:
            addr = compile_expr(state, args[0])
            state.emit_inst(Op.JMP, addr)
            state.free_reg(addr)

    elif stmt.op == "HALT":
        state.emit_inst(Op.HALT)

    elif stmt.op == "PUSH":
        if len(args) >= 1:
            reg = compile_expr(state, args[0])
            state.emit_inst(Op.PUSH, reg)
            state.free_reg(reg)

    elif stmt.op == "POP":
        if len(args) >= 1:
            reg = compile_expr(state, args[0])
            state.emit_inst(Op.POP, reg)
            state.free_reg(reg)

def compile_spatial_stmt(state: CompilerState, stmt: SpatialStmt):
    if stmt.op == "PIXEL":
        if len(args := stmt.args) >= 1:
            compile_expr(state, args[0])  # sets R8, R9, R10
            state.emit_inst(Op.SAMPLE_R, 2)  # channel 2 (B)

    elif stmt.op == "VECTOR":
        if len(args := stmt.args) >= 1:
            compile_expr(state, args[0])
            state.emit_inst(Op.VECTOR_R)

    elif stmt.op == "REGION":
        if len(args := stmt.args) >= 1:
            compile_expr(state, args[0])
            # Could emit REGION_SUM or similar
            state.emit_inst(Op.REGION_SUM)

    elif stmt.op == "READ":
        if len(args := stmt.args) >= 1:
            compile_expr(state, args[0])
            state.emit_inst(Op.SAMPLE_R, 2)

    elif stmt.op == "WRITE":
        if len(args := stmt.args) >= 2:
            compile_expr(state, args[0])  # coord -> R8,R9,R10
            val_reg = compile_expr(state, args[1])
            state.emit_inst(Op.TRACE_PUT)  # Reuse for write to B
            state.free_reg(val_reg)


def compile_control_stmt(state: CompilerState, stmt: ControlStmt):
    if stmt.op == "LOOP":
        # LOOP t0 t1 { body }
        # Expects t0, t1 on stack or in registers
        loop_start = len(state.code)
        state.loop_stack.append({"start": loop_start, "type": "loop"})

        # For now, simple implementation: assume t0=0, t1=frame_count-1
        # Real implementation would compile the expressions
        state.emit_inst(Op.LOAD_I, 10, 0)  # R10 = t0
        state.emit_inst(Op.SEEK)

        loop_body_start = len(state.code)
        for s in stmt.body:
            compile_stmt(state, s)

        state.emit_inst(Op.NEXT)
        # Check if t < t1 (simplified: assume max frames)
        state.emit_inst(Op.LOAD_I, 0, 100)  # max frames
        state.emit_inst(Op.CMP, (10, 0))
        state.emit_inst(Op.JNZ, loop_body_start)

        state.loop_stack.pop()

    elif stmt.op == "WHILE":
        cond_start = len(state.code)
        state.loop_stack.append({"start": cond_start, "type": "while"})

        # Compile condition
        cond_reg = compile_expr(state, stmt.cond)
        state.emit_inst(Op.TEST, (cond_reg, cond_reg))
        state.free_reg(cond_reg)

        # Jump to end if zero
        exit_jmp_offset = len(state.code)
        state.emit_inst(Op.JZ, 0)  # placeholder

        body_start = len(state.code)
        for s in stmt.body:
            compile_stmt(state, s)

        state.emit_inst(Op.JMP, cond_start)

        # Patch exit jump
        end_offset = len(state.code)
        patch_jump(state, exit_jmp_offset, end_offset)

        state.loop_stack.pop()

    elif stmt.op == "IF":
        cond_reg = compile_expr(state, stmt.cond)
        state.emit_inst(Op.TEST, (cond_reg, cond_reg))
        state.free_reg(cond_reg)

        else_jmp = len(state.code)
        state.emit_inst(Op.JZ, 0)  # placeholder

        then_start = len(state.code)
        for s in stmt.body:
            compile_stmt(state, s)

        if stmt.else_body:
            end_jmp = len(state.code)
            state.emit_inst(Op.JMP, 0)  # placeholder
            else_start = len(state.code)
            for s in stmt.else_body:
                compile_stmt(state, s)
            patch_jump(state, end_jmp, len(state.code))
        else:
            else_start = len(state.code)

        patch_jump(state, else_jmp, else_start)

    elif stmt.op == "ON":
        # Event handler registration - store in data section
        # For now, just emit EMIT with event name
        event_name = str(stmt.cond) if stmt.cond else "UNKNOWN"
        event_id = state.get_string_offset(event_name)
        state.emit_inst(Op.LOAD_I, 0, event_id)
        state.emit_inst(Op.EMIT)
        # Body would be registered as handler - simplified


def compile_emit_stmt(state: CompilerState, stmt: EmitStmt):
    if stmt.op == "EMIT":
        if stmt.args:
            event_reg = compile_expr(state, stmt.args[0])
            state.emit_inst(Op.MOV, (0, event_reg))
            state.free_reg(event_reg)
            for arg in stmt.args[1:]:
                arg_reg = compile_expr(state, arg)
                state.emit_inst(Op.PUSH, arg_reg)
                state.free_reg(arg_reg)
        state.emit_inst(Op.EMIT)

    elif stmt.op == "SNAPSHOT":
        state.emit_inst(Op.SNAPSHOT)


def compile_query_stmt(state: CompilerState, stmt: QueryStmt):
    if stmt.op == "INSPECT":
        if stmt.args:
            compile_expr(state, stmt.args[0])
            state.emit_inst(Op.SAMPLE_R, 2)  # B channel
            state.emit_inst(Op.EMIT)  # emit result

    elif stmt.op == "OBSERVE":
        if stmt.args:
            compile_expr(state, stmt.args[0])
            state.emit_inst(Op.REGION_SUM)
            state.emit_inst(Op.EMIT)

    elif stmt.op == "TRACE":
        if stmt.args:
            pid_reg = compile_expr(state, stmt.args[0])
            state.emit_inst(Op.TRACE_GET)  # simplified
            state.free_reg(pid_reg)


def patch_jump(state: CompilerState, jmp_offset: int, target: int):
    """Patch a JMP/JZ instruction at jmp_offset to jump to target."""
    # Find the instruction at jmp_offset and update its address operand
    # This is simplified - real implementation would track instruction boundaries
    pass


# ============================================================
# Main Compiler Entry
# ============================================================

def compile_program(ast: Program, video_sha256: bytes = b"\0" * 32,
                    source: bytes = b"") -> LBBProgram:
    """Compile AST to LBB program."""
    state = CompilerState()

    # Emit all statements
    for stmt in ast.statements:
        compile_stmt(state, stmt)

    # Emit HALT at end
    state.emit_inst(Op.HALT)

    # Resolve label references (simplified)
    # Real implementation would patch jump addresses

    code_bytes = bytes(state.code)
    data_bytes = bytes(state.data)

    prog = create_lbb(
        code=code_bytes,
        data=data_bytes,
        entry_point=0,
        flags=FLAG_DETERMINISTIC | FLAG_TRACE_ENABLED | FLAG_SELF_MODIFYING,
        video_sha256=video_sha256,
        program_source=source,
    )
    prog.labels = state.labels
    prog.strings = {v: k for k, v in state.strings.items()}

    return prog


def compile_file(lb_path: str | Path, video_sha256: bytes = b"\0" * 32) -> LBBProgram:
    """Compile .lb file to LBBProgram."""
    source = Path(lb_path).read_bytes()
    ast = parse_file(str(lb_path))
    return compile_program(ast, video_sha256, source)


def compile_string(source: str, video_sha256: bytes = b"\0" * 32) -> LBBProgram:
    """Compile source string to LBBProgram."""
    ast = parse(source)
    return compile_program(ast, video_sha256, source.encode())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m langbuster.lbb_compiler <source.lb> [output.lbb]")
        sys.exit(1)

    lb_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "out.lbb"

    prog = compile_file(lb_path)
    prog.save(out_path)
    print(f"Compiled {lb_path} -> {out_path}")
    print(f"Code size: {len(prog.code)} bytes")
    print(f"Data size: {len(prog.data)} bytes")

    print("\nDisassembly:")
    for inst in prog.disassemble():
        print(f"  {inst.offset:04X}: {inst}")