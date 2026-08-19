# LangBuster v0.1 — Binary Specification (.lbb)

**Máquina Virtual Turing-Completa con Sustrato Espaciotemporal Nativo**

---

## 1. Filosofía

LangBuster no es "Python para video". Es una **máquina abstracta** cuyo espacio de datos incluye:

- **Cinta/Tape** tradicional (acceso aleatorio, infinita conceptualmente)
- **Registros** de propósito general (R0–R15)
- **Pila** de llamadas (call stack)
- **Cursor Temporal** (frame actual `t`)
- **Campo Video** (x, y, t) — memoria direccionable nativa del sustrato

El bytecode `.lbb` es **código real**, no JSON disfrazado. La VM lo ejecuta nativamente.

---

## 2. Formato .lbb (Binary)

```
┌─────────────────────────────────────┐
│           HEADER (64 bytes)         │
├─────────────────────────────────────┤
│  Magic: "LBB\0"         (4 bytes)   │
│  Version: 0x01          (1 byte)    │
│  Flags                  (1 byte)    │
│  Entry Point (PC)       (4 bytes)   │
│  Code Size              (4 bytes)   │
│  Data Size              (4 bytes)   │
│  Video SHA256           (32 bytes)  │
│  Program SHA256         (32 bytes)  │
│  Timestamp              (8 bytes)   │
│  Reserved               (8 bytes)   │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│           CODE SECTION              │
│  (variable, Code Size bytes)        │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│           DATA SECTION              │
│  (variable, Data Size bytes)        │
└─────────────────────────────────────┘
```

### Flags (1 byte)
```
Bit 0: DETERMINISTIC    (seed fixed)
Bit 1: TRACE_ENABLED    (emit temporal events)
Bit 2: SELF_MODIFYING   (code section writable)
Bit 3: VIDEO_MMAP       (video memory mapped)
Bit 4-7: Reserved
```

---

## 3. Arquitectura de la VM

### Registros (16 × 32-bit)
```
R0–R7   : General purpose
R8–R11  : Temporal ops (x, y, t, result)
R12     : SP (Stack Pointer)
R13     : BP (Base Pointer)
R14     : PC (Program Counter) — implicit
R15     : FLAGS (Z, N, C, V, T)
```

### FLAGS Register (R15)
```
Bit 0: Z  (Zero)
Bit 1: N  (Negative)
Bit 2: C  (Carry)
Bit 3: V  (Overflow)
Bit 4: T  (Temporal — set by temporal ops)
Bit 5-31: Reserved
```

### Stack
- Crecimiento hacia abajo
- 32-bit values
- Call frames: [return_addr][old_bp][locals...]

### Video Memory Map (Virtual)
```
Dirección virtual → (x, y, t, channel)
Channel 0: R (vx)     Channel 1: G (vy)
Channel 2: B (scalar) Channel 3: Trace
```
Acceso via instrucciones `SAMPLE` / `SEEK` / `STORE`.

---

## 4. Conjunto de Instrucciones (Opcode Map)

### 4.1 Control Flow (0x00–0x0F)
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x00 | `NOP` | — | No operation |
| 0x01 | `HALT` | — | Stop execution |
| 0x02 | `JMP` | addr16 | PC = addr |
| 0x03 | `JZ` | addr16 | if Z: PC = addr |
| 0x04 | `JNZ` | addr16 | if !Z: PC = addr |
| 0x05 | `JN` | addr16 | if N: PC = addr |
| 0x06 | `JC` | addr16 | if C: PC = addr |
| 0x07 | `CALL` | addr16 | push(PC); PC = addr |
| 0x08 | `RET` | — | PC = pop() |
| 0x09 | `JMP_R` | reg | PC = R[reg] |
| 0x0A | `CALL_R` | reg | push(PC); PC = R[reg] |

### 4.2 Arithmetic/Logic (0x10–0x1F)
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x10 | `ADD` | dst, src | R[dst] += R[src]; flags |
| 0x11 | `SUB` | dst, src | R[dst] -= R[src]; flags |
| 0x12 | `MUL` | dst, src | R[dst] *= R[src]; flags |
| 0x13 | `DIV` | dst, src | R[dst] /= R[src]; flags |
| 0x14 | `MOD` | dst, src | R[dst] %= R[src]; flags |
| 0x15 | `AND` | dst, src | R[dst] &= R[src]; flags |
| 0x16 | `OR`  | dst, src | R[dst] \|= R[src]; flags |
| 0x17 | `XOR` | dst, src | R[dst] ^= R[src]; flags |
| 0x18 | `NOT` | dst | R[dst] = ~R[dst]; flags |
| 0x19 | `SHL` | dst, src | R[dst] <<= R[src]; flags |
| 0x1A | `SHR` | dst, src | R[dst] >>= R[src]; flags |
| 0x1B | `INC` | dst | R[dst]++; flags |
| 0x1C | `DEC` | dst | R[dst]--; flags |
| 0x1D | `NEG` | dst | R[dst] = -R[dst]; flags |
| 0x1E | `CMP` | a, b | flags = R[a] - R[b] |
| 0x1F | `TEST` | a, b | flags = R[a] & R[b] |

### 4.3 Memory/Register (0x20–0x2F)
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x20 | `MOV` | dst, src | R[dst] = R[src] |
| 0x21 | `LOAD` | dst, addr | R[dst] = MEM[addr] |
| 0x22 | `STORE` | addr, src | MEM[addr] = R[src] |
| 0x23 | `LOAD_I` | dst, imm16 | R[dst] = imm16 |
| 0x24 | `PUSH` | src | push(R[src]) |
| 0x25 | `POP` | dst | R[dst] = pop() |
| 0x26 | `PUSH_I` | imm16 | push(imm16) |
| 0x27 | `PEEK` | dst, offset | R[dst] = stack[SP+offset] |
| 0x28 | `POKE` | offset, src | stack[SP+offset] = R[src] |
| 0x29 | `XCHG` | a, b | swap(R[a], R[b]) |

### 4.4 Tape Operations (0x30–0x3F) — Turing Tape
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x30 | `TAPE_L` | — | tape_head-- |
| 0x31 | `TAPE_R` | — | tape_head++ |
| 0x32 | `TAPE_READ` | dst | R[dst] = tape[tape_head] |
| 0x33 | `TAPE_WRITE` | src | tape[tape_head] = R[src] |
| 0x34 | `TAPE_JMP` | addr16 | tape_head = addr |
| 0x35 | `TAPE_COPY` | len, dst | copy len bytes tape→mem |
| 0x36 | `TAPE_FIND` | val, max | search val in tape[head:head+max] |

### 4.5 Spatiotemporal Native (0x40–0x5F) — THE CORE
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x40 | `SEEK` | t | temporal_cursor = t; load frame |
| 0x41 | `NEXT` | — | temporal_cursor++; load frame |
| 0x42 | `PREV` | — | temporal_cursor--; load frame |
| 0x43 | `SAMPLE` | ch, x, y | R[R8]=sample(ch, x, y, t) |
| 0x44 | `SAMPLE_R` | ch | R[R8]=sample(ch, R[R8], R[R9], R[R10]) |
| 0x45 | `VECTOR` | x, y | R[R8]=vx; R[R9]=vy at (x,y,t) |
| 0x46 | `VECTOR_R` | — | sample vector at (R8,R9,R10) |
| 0x47 | `DELTA` | x, y | R[R8]=B(t)-B(t-1) at (x,y) |
| 0x48 | `DELTA_R` | — | delta at (R8,R9,R10) |
| 0x49 | `MOTION` | x1,y1,x2,y2 | motion vectors in region → tape |
| 0x4A | `PERSIST` | x1,y1,x2,y2,t0,t1 | R[R8]=1 if region≠0 ∀t∈[t0,t1] |
| 0x4B | `APPEAR` | x1,y1,x2,y2 | R[R8]=1 if B(t-1)=0 ∧ B(t)≠0 |
| 0x4C | `DISAPPEAR` | x1,y1,x2,y2 | R[R8]=1 if B(t-1)≠0 ∧ B(t)=0 |
| 0x4D | `REGION_SUM` | x1,y1,x2,y2 | sum of B in region → R[R8] |
| 0x4E | `REGION_CNT` | x1,y1,x2,y2 | count non-zero in region → R[R8] |
| 0x4F | `TRACE_PUT` | x, y, val | trace_layer[x,y] = val |
| 0x50 | `TRACE_GET` | x, y | R[R8] = trace_layer[x,y] |
| 0x51 | `FRAME_HASH` | t | R[R8] = hash(frame_b[t]) |
| 0x52 | `EMIT` | event_id | emit temporal event to trace |
| 0x53 | `SNAPSHOT` | — | snapshot VM state to trace |

### 4.6 Self-Modifying Code (0x60–0x6F)
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x60 | `CODE_READ` | dst, addr | R[dst] = CODE[addr] |
| 0x61 | `CODE_WRITE` | addr, src | CODE[addr] = R[src] |
| 0x62 | `CODE_JMP` | addr | PC = addr (indirect) |
| 0x63 | `PATCH` | addr, opcode, ops... | patch instruction at addr |

### 4.7 Video I/O (0x70–0x7F)
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x70 | `VID_OPEN` | path_ptr | open video file |
| 0x71 | `VID_CLOSE` | — | close video |
| 0x72 | `VID_INFO` | — | push video metadata to stack |
| 0x73 | `VID_FRAMES` | — | R[R8] = frame_count |

### 4.8 Crypto/Hash (0x80–0x8F)
| Opcode | Mnemónico | Operandos | Semántica |
|--------|-----------|-----------|-----------|
| 0x80 | `SHA256` | addr, len, dst | hash memory region |
| 0x81 | `RAND` | dst | R[dst] = random() |

---

## 5. Encoding de Instrucciones

### Formato General (variable length)
```
[OPCODE:1] [OPERANDS...]

Operand types:
- reg:      4 bits (0-15)
- addr16:   2 bytes little-endian
- imm16:    2 bytes little-endian
- imm32:    4 bytes little-endian
- addr8:    1 byte
```

### Ejemplos
```
NOP                    → 0x00
HALT                   → 0x01
JMP 0x1234             → 0x02 0x34 0x12
JZ 0x0050              → 0x03 0x50 0x00
ADD R3, R5             → 0x10 0x35       (dst=3<<4 | src=5)
LOAD_I R7, 0xABCD      → 0x23 0x07 0xCD 0xAB
SEEK 42                → 0x40 0x2A 0x00
SAMPLE 2, 100, 100     → 0x43 0x02 0x64 0x00 0x64 0x00
PERSIST 10,10,20,20,0,99 → 0x4A 0x0A 0x00 0x0A 0x00 0x14 0x00 0x14 0x00 0x00 0x00 0x63 0x00
```

---

## 6. Pruebas de Turing-Completitud

### Método 1: Simulación de Máquina de Turing (Tape + State)
La cinta (`TAPE_*`) + estado finito (registros + PC) + transiciones condicionales (`JZ`, `JNZ`) = MT universal.

### Método 2: Tag System (Post Tag System)
- Cinta = cola de símbolos
- `TAPE_READ`/`TAPE_WRITE`/`TAPE_L`/`TAPE_R` implementan producción/borrado
- Reglas de producción codificadas en código auto-modificable (`CODE_WRITE`)

### Método 3: Rule 110 Cellular Automaton
- Cinta = fila de celdas
- `TAPE_READ` vecinos (offset -1, 0, +1)
- Regla 110 codificada en lookup table en memoria
- Iteración via `JMP` loop

### Método 4: Brainfuck Equivalence
```
>  → TAPE_R
<  → TAPE_L
+  → TAPE_READ R0; INC R0; TAPE_WRITE R0
-  → TAPE_READ R0; DEC R0; TAPE_WRITE R0
.  → TAPE_READ R0; EMIT_OUTPUT R0
,  → INPUT → TAPE_WRITE
[  → TAPE_READ R0; JZ after_loop
]  → JMP loop_start
```
Todos mapeables 1:1 a instrucciones LangBuster.

---

## 7. Self-Modifying Code Pattern

```assembly
; Programa que se reescribe basado en video
START:
    SEEK 0
    SAMPLE 2, 320, 240     ; lee B en centro frame 0
    CMP R8, 128
    JZ BRIGHT_PATH
    
DARK_PATH:
    ; Parchea la siguiente instrucción a HALT
    LOAD_I R0, PATCH_TARGET
    LOAD_I R1, 0x01        ; HALT opcode
    CODE_WRITE R0, R1
    JMP CONTINUE

BRIGHT_PATH:
    ; Parchea a NOP (continúa normal)
    LOAD_I R0, PATCH_TARGET
    LOAD_I R1, 0x00        ; NOP
    CODE_WRITE R0, R1

PATCH_TARGET:
    NOP                    ; ← será reescrito en runtime!

CONTINUE:
    NEXT
    JMP START
```

---

## 8. ExecutionTrace Binary Format

Trace binario compacto (no JSON):
```
[EVENT_COUNT:4] [EVENTS...]

EVENT:
  [TICK:4] [T:4] [TIMESTAMP:8] [TYPE:1] [PAYLOAD_LEN:2] [PAYLOAD...]
```

Types: 0=FRAME, 1=SEEK, 2=SAMPLE, 3=DELTA, 4=MOTION, 5=PERSIST, 6=APPEAR, 7=DISAPPEAR, 8=EMIT, 9=SNAPSHOT, 10=CODE_PATCH, 255=HALT

---

## 9. Compiler Pipeline

```
source.lb
    │
    ▼
Lexer → Tokens
    │
    ▼
Parser → AST
    │
    ▼
Semantic Analysis (resolve labels, check types)
    │
    ▼
Code Gen → Bytecode (instructions + data)
    │
    ▼
Linker (resolve addresses, compute sizes)
    │
    ▼
Emitter → program.lbb
```

---

## 10. Deterministic Replay

Mismo `.lbb` + mismo video + mismo seed = **bit-identical trace**.

La VM no tiene fuentes de entropía excepto `RAND` (seedeada). Todo lo demás es puro.

---

*LangBuster merece su formato. Su bytecode. Su VM. Su locura.* 🗿