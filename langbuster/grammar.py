#!/usr/bin/env python3
"""
LangBuster v0.1 — Grammar & Parser
Spatiotemporal esolang grammar for video computation.

Grammar (EBNF-inspired):

program        ::= statement*
statement      ::= frame_stmt
                 | temporal_stmt
                 | spatial_stmt
                 | control_stmt
                 | emit_stmt
                 | query_stmt

# Frame navigation
frame_stmt     ::= "FRAME" expr                    # Load frame t
               |  "NEXT"                           # t+1
               |  "PREV"                           # t-1
               |  "JUMP" expr                      # Absolute frame

# Temporal operations
temporal_stmt  ::= "DELTA"  coord                  # Pixel delta at (x,y)
               |  "MOTION" region                 # Motion vectors in region
               |  "TRACE"  particle_expr          # Particle trajectory
               |  "PERSIST" region expr expr      # Region persists t0..t1
               |  "APPEAR"  region                # Region appears
               |  "DISAPPEAR" region              # Region disappears

# Spatial operations (FLOW-like)
spatial_stmt   ::= "PIXEL"  coord                 # Pixel at (x,y,t)
               |  "REGION" region                 # Region at current t
               |  "VECTOR" coord                  # Vector field at (x,y)
               |  "WRITE"  coord expr             # Write to B field
               |  "READ"   coord                  # Read from B field

# Control flow
control_stmt   ::= "LOOP"   expr expr block       # Loop t0..t1
               |  "WHILE"  temporal_cond block   # While condition
               |  "ON"     temporal_event block  # Event handler
               |  "IF"     expr block ("ELSE" block)?

# Events
temporal_event ::= "CHANGE"  coord
               |  "MOTION"  region
               |  "APPEAR"  region
               |  "DISAPPEAR" region
               |  "ENTER"   region
               |  "EXIT"    region

# Emission
emit_stmt      ::= "EMIT"   string_expr (expr)*   # Emit event to trace
               |  "SNAPSHOT"                     # Snapshot state

# Queries / output
query_stmt     ::= "INSPECT" coord               # Inspect pixel
               |  "OBSERVE" region              # Observe region
               |  "TRACE"   particle_expr       # Trace particle

# Expressions
expr           ::= number | identifier | string | expr op expr | "(" expr ")"
coord          ::= expr "," expr ("," expr)?     # x,y or x,y,t
region         ::= expr "," expr "," expr "," expr ("," expr "," expr)?  # x1,y1,x2,y2 or x1,y1,x2,y2,t0,t1
particle_expr  ::= "PARTICLE" expr               # Particle by ID
block          ::= "{" statement* "}"

# Identifiers
identifier     ::= letter (letter | digit | "_")*
number         ::= digit+ ("." digit+)?
string         ::= '"' [^"]* '"'
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from enum import Enum, auto
import re


class TokenType(Enum):
    # Keywords
    FRAME = auto()
    NEXT = auto()
    PREV = auto()
    JUMP = auto()
    DELTA = auto()
    MOTION = auto()
    TRACE = auto()
    PERSIST = auto()
    APPEAR = auto()
    DISAPPEAR = auto()
    ENTER = auto()
    EXIT = auto()
    PIXEL = auto()
    REGION = auto()
    VECTOR = auto()
    WRITE = auto()
    READ = auto()
    LOOP = auto()
    WHILE = auto()
    ON = auto()
    IF = auto()
    ELSE = auto()
    EMIT = auto()
    SNAPSHOT = auto()
    INSPECT = auto()
    OBSERVE = auto()
    PARTICLE = auto()

    # Binary VM / Tape instructions
    TAPE_L = auto()
    TAPE_R = auto()
    TAPE_READ = auto()
    TAPE_WRITE = auto()
    TAPE_JMP = auto()
    LOAD_I = auto()
    PUSH = auto()
    POP = auto()
    MOV = auto()
    ADD = auto()
    SUB = auto()
    JZ = auto()
    JMP = auto()
    HALT = auto()

    # Symbols
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    COLON = auto()
    ARROW = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    MODULO = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    ASSIGN = auto()

    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()


KEYWORDS = {
    "FRAME": TokenType.FRAME,
    "NEXT": TokenType.NEXT,
    "PREV": TokenType.PREV,
    "JUMP": TokenType.JUMP,
    "DELTA": TokenType.DELTA,
    "MOTION": TokenType.MOTION,
    "TRACE": TokenType.TRACE,
    "PERSIST": TokenType.PERSIST,
    "APPEAR": TokenType.APPEAR,
    "DISAPPEAR": TokenType.DISAPPEAR,
    "ENTER": TokenType.ENTER,
    "EXIT": TokenType.EXIT,
    "PIXEL": TokenType.PIXEL,
    "REGION": TokenType.REGION,
    "VECTOR": TokenType.VECTOR,
    "WRITE": TokenType.WRITE,
    "READ": TokenType.READ,
    "LOOP": TokenType.LOOP,
    "WHILE": TokenType.WHILE,
    "ON": TokenType.ON,
    "IF": TokenType.IF,
    "ELSE": TokenType.ELSE,
    "EMIT": TokenType.EMIT,
    "SNAPSHOT": TokenType.SNAPSHOT,
    "INSPECT": TokenType.INSPECT,
    "OBSERVE": TokenType.OBSERVE,
    "PARTICLE": TokenType.PARTICLE,
    # Tape/Binary VM instructions
    "TAPE_L": TokenType.TAPE_L,
    "TAPE_R": TokenType.TAPE_R,
    "TAPE_READ": TokenType.TAPE_READ,
    "TAPE_WRITE": TokenType.TAPE_WRITE,
    "TAPE_JMP": TokenType.TAPE_JMP,
    "LOAD_I": TokenType.LOAD_I,
    "PUSH": TokenType.PUSH,
    "POP": TokenType.POP,
    "MOV": TokenType.MOV,
    "ADD": TokenType.ADD,
    "SUB": TokenType.SUB,
    "JZ": TokenType.JZ,
    "JMP": TokenType.JMP,
    "HALT": TokenType.HALT,
    "CHANGE": TokenType.IDENT,  # handled in parser
    "AND": TokenType.IDENT,
    "OR": TokenType.IDENT,
    "NOT": TokenType.IDENT,
}


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    def peek(self, n: int = 0) -> str:
        if self.pos + n >= len(self.source):
            return '\0'
        return self.source[self.pos + n]

    def advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            ch = self.peek()
            if ch in ' \t\r':
                self.advance()
            elif ch == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, None, self.line, self.col))
                self.advance()
            elif ch == '#':
                while self.peek() not in '\n\0':
                    self.advance()
            elif ch == '{':
                self.tokens.append(Token(TokenType.LBRACE, None, self.line, self.col))
                self.advance()
            elif ch == '}':
                self.tokens.append(Token(TokenType.RBRACE, None, self.line, self.col))
                self.advance()
            elif ch == '(':
                self.tokens.append(Token(TokenType.LPAREN, None, self.line, self.col))
                self.advance()
            elif ch == ')':
                self.tokens.append(Token(TokenType.RPAREN, None, self.line, self.col))
                self.advance()
            elif ch == ',':
                self.tokens.append(Token(TokenType.COMMA, None, self.line, self.col))
                self.advance()
            elif ch == ':':
                self.tokens.append(Token(TokenType.COLON, None, self.line, self.col))
                self.advance()
            elif ch == '=' and self.peek(1) == '=':
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.EQ, None, self.line, self.col))
            elif ch == '!' and self.peek(1) == '=':
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.NEQ, None, self.line, self.col))
            elif ch == '<' and self.peek(1) == '=':
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.LE, None, self.line, self.col))
            elif ch == '>' and self.peek(1) == '=':
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.GE, None, self.line, self.col))
            elif ch == '<':
                self.tokens.append(Token(TokenType.LT, None, self.line, self.col))
                self.advance()
            elif ch == '>':
                self.tokens.append(Token(TokenType.GT, None, self.line, self.col))
                self.advance()
            elif ch == '=':
                self.tokens.append(Token(TokenType.ASSIGN, None, self.line, self.col))
                self.advance()
            elif ch == '+':
                self.tokens.append(Token(TokenType.PLUS, None, self.line, self.col))
                self.advance()
            elif ch == '-':
                if self.peek(1) == '>':
                    self.advance(); self.advance()
                    self.tokens.append(Token(TokenType.ARROW, None, self.line, self.col))
                else:
                    self.tokens.append(Token(TokenType.MINUS, None, self.line, self.col))
                    self.advance()
            elif ch == '*':
                self.tokens.append(Token(TokenType.STAR, None, self.line, self.col))
                self.advance()
            elif ch == '/':
                self.tokens.append(Token(TokenType.SLASH, None, self.line, self.col))
                self.advance()
            elif ch == '%':
                self.tokens.append(Token(TokenType.MODULO, None, self.line, self.col))
                self.advance()
            elif ch == '"':
                self.tokens.append(self.read_string())
            elif ch.isdigit() or (ch == '.' and self.peek(1).isdigit()):
                self.tokens.append(self.read_number())
            elif ch.isalpha() or ch == '_':
                self.tokens.append(self.read_ident())
            else:
                raise SyntaxError(f"Unknown character {ch!r} at {self.line}:{self.col}")

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return self.tokens

    def read_string(self) -> Token:
        start_line, start_col = self.line, self.col
        self.advance()  # skip opening "
        chars = []
        while True:
            ch = self.peek()
            if ch == '\0' or ch == '\n':
                raise SyntaxError(f"Unterminated string at {start_line}:{start_col}")
            if ch == '"':
                self.advance()
                break
            if ch == '\\':
                self.advance()
                esc = self.peek()
                if esc == 'n': chars.append('\n')
                elif esc == 't': chars.append('\t')
                elif esc == '\\': chars.append('\\')
                elif esc == '"': chars.append('"')
                else: chars.append(esc)
                self.advance()
            else:
                chars.append(ch)
                self.advance()
        return Token(TokenType.STRING, ''.join(chars), start_line, start_col)

    def read_number(self) -> Token:
        start_line, start_col = self.line, self.col
        chars = []
        has_dot = False
        while True:
            ch = self.peek()
            if ch.isdigit():
                chars.append(ch)
                self.advance()
            elif ch == '.' and not has_dot:
                has_dot = True
                chars.append(ch)
                self.advance()
            else:
                break
        val = float(''.join(chars)) if has_dot else int(''.join(chars))
        return Token(TokenType.NUMBER, val, start_line, start_col)

    def read_ident(self) -> Token:
        start_line, start_col = self.line, self.col
        chars = []
        while True:
            ch = self.peek()
            if ch.isalnum() or ch == '_':
                chars.append(ch)
                self.advance()
            else:
                break
        text = ''.join(chars)
        tok_type = KEYWORDS.get(text.upper(), TokenType.IDENT)
        return Token(tok_type, text, start_line, start_col)


# ============================================================
# AST Nodes
# ============================================================

@dataclass
class ASTNode:
    pass


@dataclass
class Program(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class FrameStmt(ASTNode):
    op: str  # FRAME, NEXT, PREV, JUMP
    arg: Optional[ASTNode] = None


@dataclass
class TemporalStmt(ASTNode):
    op: str  # DELTA, MOTION, TRACE, PERSIST, APPEAR, DISAPPEAR
    args: List[ASTNode] = field(default_factory=list)


@dataclass
class SpatialStmt(ASTNode):
    op: str  # PIXEL, REGION, VECTOR, WRITE, READ
    args: List[ASTNode] = field(default_factory=list)


@dataclass
class ControlStmt(ASTNode):
    op: str  # LOOP, WHILE, ON, IF
    cond: Optional[ASTNode] = None
    body: List[ASTNode] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None


@dataclass
class EmitStmt(ASTNode):
    op: str  # EMIT, SNAPSHOT
    args: List[ASTNode] = field(default_factory=list)


@dataclass
class QueryStmt(ASTNode):
    op: str  # INSPECT, OBSERVE, TRACE
    args: List[ASTNode] = field(default_factory=list)


@dataclass
class CoordExpr(ASTNode):
    x: ASTNode
    y: ASTNode
    t: Optional[ASTNode] = None


@dataclass
class RegionExpr(ASTNode):
    x1: ASTNode
    y1: ASTNode
    x2: ASTNode
    y2: ASTNode
    t0: Optional[ASTNode] = None
    t1: Optional[ASTNode] = None


@dataclass
class BinaryExpr(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass
class UnaryExpr(ASTNode):
    op: str
    expr: ASTNode


@dataclass
class NumberLiteral(ASTNode):
    value: Union[int, float]


@dataclass
class StringLiteral(ASTNode):
    value: str


@dataclass
class IdentExpr(ASTNode):
    name: str


@dataclass
class ParticleExpr(ASTNode):
    pid: ASTNode


# ============================================================
# Parser
# ============================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]

    def expect(self, ttype: TokenType) -> Token:
        if self.current and self.current.type == ttype:
            tok = self.current
            self.advance()
            return tok
        raise SyntaxError(f"Expected {ttype.name}, got {self.current.type.name if self.current else 'EOF'} at {self.current.line if self.current else '?'}:{self.current.col if self.current else '?'}")

    def match(self, *ttypes: TokenType) -> bool:
        return self.current and self.current.type in ttypes

    def consume_newlines(self):
        while self.match(TokenType.NEWLINE):
            self.advance()

    def parse(self) -> Program:
        prog = Program()
        self.consume_newlines()
        while not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                prog.statements.append(stmt)
            self.consume_newlines()
        return prog

    def parse_statement(self) -> Optional[ASTNode]:
        if not self.current or self.match(TokenType.EOF, TokenType.RBRACE):
            return None

        # Frame navigation
        if self.match(TokenType.FRAME, TokenType.NEXT, TokenType.PREV, TokenType.JUMP):
            return self.parse_frame_stmt()

        # Temporal operations
        if self.match(TokenType.DELTA, TokenType.MOTION, TokenType.TRACE,
                      TokenType.PERSIST, TokenType.APPEAR, TokenType.DISAPPEAR):
            return self.parse_temporal_stmt()

        # Spatial operations
        if self.match(TokenType.PIXEL, TokenType.REGION, TokenType.VECTOR,
                      TokenType.WRITE, TokenType.READ):
            return self.parse_spatial_stmt()

        # Control flow
        if self.match(TokenType.LOOP, TokenType.WHILE, TokenType.ON, TokenType.IF):
            return self.parse_control_stmt()

        # Emission
        if self.match(TokenType.EMIT, TokenType.SNAPSHOT):
            return self.parse_emit_stmt()

        # Queries
        if self.match(TokenType.INSPECT, TokenType.OBSERVE, TokenType.TRACE):
            return self.parse_query_stmt()

        # Binary VM / Tape instructions
        if self.match(TokenType.TAPE_L, TokenType.TAPE_R, TokenType.TAPE_READ,
                      TokenType.TAPE_WRITE, TokenType.TAPE_JMP,
                      TokenType.LOAD_I, TokenType.PUSH, TokenType.POP,
                      TokenType.MOV, TokenType.ADD, TokenType.SUB,
                      TokenType.JZ, TokenType.JMP, TokenType.HALT):
            return self.parse_binary_stmt()

        # Expression statement (assignment, etc.)
        return self.parse_expression()

    def parse_frame_stmt(self) -> FrameStmt:
        tok = self.current
        self.advance()
        arg = None
        if tok.type in (TokenType.FRAME, TokenType.JUMP):
            arg = self.parse_expression()
        return FrameStmt(op=tok.type.name, arg=arg)

    def parse_temporal_stmt(self) -> TemporalStmt:
        tok = self.current
        self.advance()
        args = []

        if tok.type == TokenType.DELTA:
            args.append(self.parse_coord())
        elif tok.type == TokenType.MOTION:
            args.append(self.parse_region())
        elif tok.type == TokenType.TRACE:
            args.append(self.parse_particle_expr())
        elif tok.type == TokenType.PERSIST:
            args.append(self.parse_region())
            args.append(self.parse_expression())  # t0
            args.append(self.parse_expression())  # t1
        elif tok.type in (TokenType.APPEAR, TokenType.DISAPPEAR):
            args.append(self.parse_region())

        return TemporalStmt(op=tok.type.name, args=args)

    def parse_spatial_stmt(self) -> SpatialStmt:
        tok = self.current
        self.advance()
        args = []

        if tok.type in (TokenType.PIXEL, TokenType.VECTOR, TokenType.READ):
            args.append(self.parse_coord())
        elif tok.type == TokenType.REGION:
            args.append(self.parse_region())
        elif tok.type == TokenType.WRITE:
            args.append(self.parse_coord())
            args.append(self.parse_expression())

        return SpatialStmt(op=tok.type.name, args=args)

    def parse_control_stmt(self) -> ControlStmt:
        tok = self.current
        self.advance()

        if tok.type == TokenType.LOOP:
            t0 = self.parse_expression()
            t1 = self.parse_expression()
            body = self.parse_block()
            return ControlStmt(op="LOOP", cond=None, body=body)
            # Note: t0, t1 stored in working memory via special handling

        elif tok.type == TokenType.WHILE:
            cond = self.parse_temporal_condition()
            body = self.parse_block()
            return ControlStmt(op="WHILE", cond=cond, body=body)

        elif tok.type == TokenType.ON:
            event = self.parse_temporal_event()
            body = self.parse_block()
            return ControlStmt(op="ON", cond=event, body=body)

        elif tok.type == TokenType.IF:
            cond = self.parse_expression()
            body = self.parse_block()
            else_body = None
            if self.match(TokenType.ELSE):
                self.advance()
                else_body = self.parse_block()
            return ControlStmt(op="IF", cond=cond, body=body, else_body=else_body)

        raise SyntaxError(f"Unknown control statement at {tok.line}:{tok.col}")

    def parse_temporal_condition(self) -> ASTNode:
        # Simplified: parse as expression for now
        return self.parse_expression()

    def parse_temporal_event(self) -> ASTNode:
        # ON CHANGE x,y | ON MOTION region | ON APPEAR region | ...
        if self.match(TokenType.IDENT):
            event_type = self.current.value.upper()
            self.advance()
            if event_type == "CHANGE":
                coord = self.parse_coord()
                return IdentExpr(name=f"ON_CHANGE_{coord.x}_{coord.y}")
            elif event_type in ("MOTION", "APPEAR", "DISAPPEAR", "ENTER", "EXIT"):
                region = self.parse_region()
                return IdentExpr(name=f"ON_{event_type}")
        raise SyntaxError(f"Expected temporal event at {self.current.line}:{self.current.col}")

    def parse_emit_stmt(self) -> EmitStmt:
        tok = self.current
        self.advance()
        args = []
        if tok.type == TokenType.EMIT:
            args.append(self.parse_expression())  # event name
            while self.match(TokenType.NUMBER, TokenType.STRING, TokenType.IDENT, TokenType.LPAREN):
                args.append(self.parse_expression())
        return EmitStmt(op=tok.type.name, args=args)

    def parse_query_stmt(self) -> QueryStmt:
        tok = self.current
        self.advance()
        args = []
        if tok.type == TokenType.INSPECT:
            args.append(self.parse_coord())
        elif tok.type == TokenType.OBSERVE:
            args.append(self.parse_region())
        elif tok.type == TokenType.TRACE:
            args.append(self.parse_particle_expr())
        return QueryStmt(op=tok.type.name, args=args)

    def parse_binary_stmt(self) -> ASTNode:
        """Parse binary VM / tape instructions."""
        tok = self.current
        self.advance()

        # Simple instructions with no operands
        if tok.type in (TokenType.TAPE_L, TokenType.TAPE_R, TokenType.HALT):
            return TemporalStmt(op=tok.type.name, args=[])

        # Instructions with register operand
        if tok.type in (TokenType.TAPE_READ, TokenType.TAPE_WRITE, TokenType.PUSH, TokenType.POP):
            reg = self.parse_expression()
            return TemporalStmt(op=tok.type.name, args=[reg])

        # LOAD_I reg, imm
        if tok.type == TokenType.LOAD_I:
            reg = self.parse_expression()
            self.expect(TokenType.COMMA)
            imm = self.parse_expression()
            return TemporalStmt(op=tok.type.name, args=[reg, imm])

        # MOV dst, src
        if tok.type == TokenType.MOV:
            dst = self.parse_expression()
            self.expect(TokenType.COMMA)
            src = self.parse_expression()
            return TemporalStmt(op=tok.type.name, args=[dst, src])

        # ADD dst, src / SUB dst, src
        if tok.type in (TokenType.ADD, TokenType.SUB):
            dst = self.parse_expression()
            self.expect(TokenType.COMMA)
            src = self.parse_expression()
            return TemporalStmt(op=tok.type.name, args=[dst, src])

        # JZ addr / JMP addr
        if tok.type in (TokenType.JZ, TokenType.JMP):
            addr = self.parse_expression()
            return TemporalStmt(op=tok.type.name, args=[addr])

        # TAPE_JMP addr
        if tok.type == TokenType.TAPE_JMP:
            addr = self.parse_expression()
            return TemporalStmt(op=tok.type.name, args=[addr])

        # Fallback: treat as expression
        return self.parse_expr_stmt()
        # Assignment or bare expression
        expr = self.parse_expression()
        if self.match(TokenType.ASSIGN):
            self.advance()
            rhs = self.parse_expression()
            if isinstance(expr, IdentExpr):
                return BinaryExpr(op="=", left=expr, right=rhs)
        return expr

    def parse_block(self) -> List[ASTNode]:
        self.expect(TokenType.LBRACE)
        self.consume_newlines()
        stmts = []
        while not self.match(TokenType.RBRACE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                stmts.append(stmt)
            self.consume_newlines()
        self.expect(TokenType.RBRACE)
        return stmts

    def parse_expression(self) -> ASTNode:
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        left = self.parse_additive()
        while self.match(TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op_tok = self.current
            self.advance()
            right = self.parse_additive()
            left = BinaryExpr(op=op_tok.type.name, left=left, right=right)
        return left

    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_tok = self.current
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryExpr(op=op_tok.type.name, left=left, right=right)
        return left

    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_unary()
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.MODULO):
            op_tok = self.current
            self.advance()
            right = self.parse_unary()
            left = BinaryExpr(op=op_tok.type.name, left=left, right=right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.MINUS, TokenType.PLUS):
            op_tok = self.current
            self.advance()
            expr = self.parse_unary()
            return UnaryExpr(op=op_tok.type.name, expr=expr)
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        if self.match(TokenType.NUMBER):
            val = self.current.value
            self.advance()
            return NumberLiteral(value=val)
        if self.match(TokenType.STRING):
            val = self.current.value
            self.advance()
            return StringLiteral(value=val)
        if self.match(TokenType.IDENT):
            name = self.current.value
            self.advance()
            return IdentExpr(name=name)
        if self.match(TokenType.PARTICLE):
            self.advance()
            pid = self.parse_expression()
            return ParticleExpr(pid=pid)
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        raise SyntaxError(f"Expected expression at {self.current.line}:{self.current.col}")

    def parse_coord(self) -> CoordExpr:
        x = self.parse_expression()
        self.expect(TokenType.COMMA)
        y = self.parse_expression()
        t = None
        if self.match(TokenType.COMMA):
            self.advance()
            t = self.parse_expression()
        return CoordExpr(x=x, y=y, t=t)

    def parse_region(self) -> RegionExpr:
        x1 = self.parse_expression()
        self.expect(TokenType.COMMA)
        y1 = self.parse_expression()
        self.expect(TokenType.COMMA)
        x2 = self.parse_expression()
        self.expect(TokenType.COMMA)
        y2 = self.parse_expression()
        t0 = t1 = None
        if self.match(TokenType.COMMA):
            self.advance()
            t0 = self.parse_expression()
            self.expect(TokenType.COMMA)
            t1 = self.parse_expression()
        return RegionExpr(x1=x1, y1=y1, x2=x2, y2=y2, t0=t0, t1=t1)

    def parse_particle_expr(self) -> ParticleExpr:
        self.expect(TokenType.PARTICLE)
        pid = self.parse_expression()
        return ParticleExpr(pid=pid)


def parse(source: str) -> Program:
    """Parse LangBuster source code into AST."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


def parse_file(path: str) -> Program:
    with open(path, 'r', encoding='utf-8') as f:
        return parse(f.read())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m langbuster.grammar <source.lb>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        source = f.read()

    ast = parse(source)
    import json
    print(json.dumps(ast.__dict__, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))