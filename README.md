# LangBuster v0.1 — Spatiotemporal Video Esolang

**Premisa**: El programa ES un video. El video codifica un campo vectorial-espacial-temporal. Partículas fluyen por el espacio-tiempo, leyendo/escribiendo el campo escalar en su posición continua (x,y,t).

---

## 1. Representación del Programa

Archivo: **video** (mp4, webm, gif — decodificado via OpenCV/PIL).

Cada frame es un estado espacial estilo FLOW:
- **R (Red)** = componente X del campo vectorial ∈ [0, 255] → mapeado a [-1, 1]
- **G (Green)** = componente Y del campo vectorial ∈ [0, 255] → mapeado a [-1, 1]
- **B (Blue)** = campo escalar (datos + instrucciones) ∈ [0, 255], **mutable por frame**

El tiempo `t` = índice de frame (0-based). Timestamp = `t / fps`.

Coordenadas: (0,0) = esquina superior izquierda. X crece a la derecha, Y crece hacia abajo.
Interpolación bilineal para posiciones sub-pixel y sub-frame.

---

## 2. Estado Espaciotemporal

La unidad computacional fundamental: **(x, y, t)** — un voxel en el cubo espacio-tiempo.

```
VIDEO (mp4/webm/gif)
    ↓ decode
FRAME t (estado espacial FLOW-like)
    ↓
field_r(x,y,t), field_g(x,y,t), field_b(x,y,t), field_trace(x,y,t)
    ↓ temporal transition
FRAME t+1
    ↓
...
```

**VideoState** mantiene:
- `current_t`: frame actual
- `previous_t`: frame anterior (para DELTA, MOTION, APPEAR, DISAPPEAR)
- `frame_states`: caché LRU de FrameState decodificados
- `working_memory`: variables del programa
- `trace_log`: eventos temporales emitidos

---

## 3. Semántica de Ejecución

### Paso de Simulación (por instrucción temporal)

```
1. FETCH:    Leer instrucción desde campo B en (x,y,t) actual
2. EXECUTE:  Ejecutar operación temporal/espacial/control
3. TRANSITION:  Avanzar/retroceder/saltar en t (FRAME, NEXT, PREV, JUMP)
4. EMIT:     Registrar eventos en trace temporal
5. TERMINATE: Si no hay más frames o max_frames alcanzado
```

El tiempo **forma parte del estado computacional**:
```
state(t) → transition(t → t+1) → state(t+1)
```

No son frames independientes; existe continuidad explícita via `previous_t` y `current_t`.

---

## 4. Conjunto de Instrucciones Temporales

Extiende instrucciones FLOW (0-255 en canal B) con operaciones nativas de tiempo (128-149):

### Navegación de Frames
| Op | Mnemónico | Semántica |
|---|---|---|
| 128 | `FRAME t` | Cargar frame `t` como estado actual |
| 129 | `NEXT` | Avanzar a `t+1` |
| 130 | `PREV` | Retroceder a `t-1` |
| 131 | `JUMP t` | Saltar a frame absoluto `t` |

### Observación Temporal
| Op | Mnemónico | Semántica |
|---|---|---|
| 132 | `DELTA x,y` | Diferencia de píxel: B(t) - B(t-1) en (x,y) |
| 133 | `MOTION region` | Vectores de movimiento entre frames en región |
| 134 | `TRACE pid` | Trayectoria de partícula a través de frames |
| 135 | `PERSIST region t0 t1` | ¿Región no-cero en todos frames [t0,t1]? |
| 136 | `APPEAR region` | ¿Región era 0 en t-1 y no-0 en t? |
| 137 | `DISAPPEAR region` | ¿Región era no-0 en t-1 y 0 en t? |
| 138 | `ENTER region` | Partícula entra en región |
| 139 | `EXIT region` | Partícula sale de región |

### Control de Flujo Temporal
| Op | Mnemónico | Semántica |
|---|---|---|
| 140 | `LOOP t0 t1 {body}` | Iterar frames t0..t1 ejecutando body |
| 141 | `WHILE cond {body}` | Mientras condición temporal se cumpla |
| 142 | `ON event {body}` | Manejador de evento temporal |
| 143 | `IF cond {body} ELSE {body}` | Condicional |

### Eventos Temporales (para ON)
- `ON CHANGE x,y` — cambio en píxel
- `ON MOTION region` — movimiento detectado
- `ON APPEAR region` — objeto aparece
- `ON DISAPPEAR region` — objeto desaparece
- `ON ENTER region` — partícula entra
- `ON EXIT region` — partícula sale

### Consultas Espaciales (estilo FLOW)
| Op | Mnemónico | Semántica |
|---|---|---|
| — | `PIXEL x,y,t` | Valor B + vector en (x,y,t) |
| — | `VECTOR x,y,t` | Campo vectorial (vx,vy) en (x,y,t) |
| — | `REGION x1,y1,x2,y2,t` | Región en frame t |
| — | `READ x,y,t` | Leer B en (x,y,t) |
| — | `WRITE x,y,t val` | Escribir B en (x,y,t) |

### Emisión / Evidencia
| Op | Mnemónico | Semántica |
|---|---|---|
| 148 | `EMIT "name" args...` | Emitir evento a trace |
| 149 | `SNAPSHOT` | Capturar estado completo |

---

## 5. Gramática (EBNF)

```
program        ::= statement*
statement      ::= frame_stmt | temporal_stmt | spatial_stmt
                 | control_stmt | emit_stmt | query_stmt

frame_stmt     ::= "FRAME" expr | "NEXT" | "PREV" | "JUMP" expr
temporal_stmt  ::= "DELTA" coord | "MOTION" region | "TRACE" particle
                 | "PERSIST" region expr expr | "APPEAR" region | "DISAPPEAR" region
spatial_stmt   ::= "PIXEL" coord | "REGION" region | "VECTOR" coord
                 | "WRITE" coord expr | "READ" coord
control_stmt   ::= "LOOP" expr expr block | "WHILE" expr block
                 | "ON" event block | "IF" expr block ("ELSE" block)?
emit_stmt      ::= "EMIT" string expr* | "SNAPSHOT"
query_stmt     ::= "INSPECT" coord | "OBSERVE" region | "TRACE" particle

coord          ::= expr "," expr ("," expr)?        # x,y or x,y,t
region         ::= expr "," expr "," expr "," expr ("," expr "," expr)?
particle       ::= "PARTICLE" expr
block          ::= "{" statement* "}"
expr           ::= number | ident | string | expr op expr | "(" expr ")"
```

---

## 6. Ejemplos

### 01_inspect_frame.lb — Inspeccionar frame individual
```langbuster
FRAME 42
INSPECT 320, 240, 42
EMIT "frame_42_inspected" _inspect
SNAPSHOT
```

### 02_detect_movement.lb — Detectar movimiento
```langbuster
FRAME 0
LOOP 1 99 {
    DELTA 320, 240
    IF _delta > 10 {
        EMIT "motion_detected" t _delta
    }
    NEXT
}
```

### 03_detect_persistent.lb — Región persistente
```langbuster
FRAME 0
PERSIST 270, 190, 370, 290, 0, 99
IF _persist == 1 {
    EMIT "region_persists" "center" 0 99
}
```

### 04_detect_appear.lb — Objeto aparece
```langbuster
FRAME 0
LOOP 1 99 {
    APPEAR 100, 100, 200, 200
    IF _appear == 1 {
        EMIT "object_appeared" t 100 100 200 200
        SNAPSHOT
    }
    NEXT
}
```

### 05_detect_disappear.lb — Objeto desaparece
```langbuster
FRAME 0
LOOP 1 99 {
    DISAPPEAR 150, 150, 250, 250
    IF _disappear == 1 {
        EMIT "object_disappeared" t 150 150 250 250
        NEXT
    }
}
```

### 06_detect_oscillation.lb — Patrón oscilante
```langbuster
FRAME 0
prev = 0; oscillations = 0; last_dir = 0
LOOP 1 99 {
    PIXEL 320, 240
    dir = (_pixel_val > prev) - (_pixel_val < prev)
    IF last_dir != 0 AND dir != 0 AND dir != last_dir {
        oscillations = oscillations + 1
    }
    prev = _pixel_val; last_dir = dir
    NEXT
}
EMIT "oscillations" oscillations
```

### 07_trace_trajectory.lb — Trayectoria de partícula
```langbuster
FRAME 0
TRACE PARTICLE 0
LOOP 0 99 {
    MOTION 300, 200, 340, 240
    EMIT "motion" t _motion
    NEXT
}
```

### 08_temporal_loop.lb — Bucle temporal con acumulación
```langbuster
FRAME 0
acc = 0
LOOP 0 99 {
    PIXEL 160, 120
    acc = acc + _pixel_val
    IF t % 10 == 0 { EMIT "checkpoint" t acc }
    NEXT
}
EMIT "sum" acc
```

### 09_frame_transform.lb — Transformación de frame
```langbuster
FRAME 0
LOOP 0 99 {
    REGION 100, 100, 200, 200
    EMIT "frame_data" t _region_b
    NEXT
}
```

### 10_absurd_esoteric.lb — "El Reloj de Sombras"
```langbuster
FRAME 0
shadow_clock = 0; max_shadow = 0; shadow_start = -1
LOOP 0 999 {
    PIXEL 100, 100
    IF _pixel_val < 10 {
        IF shadow_start == -1 { shadow_start = t }
        shadow_clock = shadow_clock + 1
    } ELSE {
        IF shadow_start != -1 {
            duration = t - shadow_start
            IF duration > max_shadow { max_shadow = duration }
            EMIT "shadow_end" shadow_start t duration
            shadow_start = -1
        }
        shadow_clock = 0
    }
    IF t % 100 == 0 { EMIT "report" t shadow_clock max_shadow }
    NEXT
}
EMIT "final" max_shadow
SNAPSHOT
```

---

## 7. ExecutionTrace — Evidencia Reproducible

Cada ejecución produce `trace.json` con:

```json
{
  "metadata": {
    "engine": "LangBuster",
    "engine_version": "langbuster-0.1+trace-1",
    "video_sha256": "...",
    "program_sha256": "...",
    "seed": 42,
    "config": { "max_frames": 0, "trace_enabled": true, "deterministic": true },
    "video": { "width": 640, "height": 480, "fps": 30, "frame_count": 900, ... },
    "execution": { "final_t": 899, "frames_processed": 900, "trace_events": 1247 }
  },
  "events": [
    { "tick": 0, "t": 0, "timestamp": 0.0, "type": "FRAME", "payload": { "t": 0 } },
    { "tick": 1, "t": 0, "timestamp": 0.0, "type": "INSPECT", "payload": { "x": 10, "y": 10, "value": 42 } },
    { "tick": 2, "t": 1, "timestamp": 0.033, "type": "DELTA", "payload": { "x": 10, "y": 10, "delta": 5 } },
    { "tick": 3, "t": 1, "timestamp": 0.033, "type": "EMIT", "payload": { "event": "motion", "delta": 5 } }
  ],
  "frame_hashes": { "0": "a1b2c3d4", "1": "e5f6g7h8", ... }
}
```

### Evidence Package (`evidence.json`)
Paquete completo para reproducibilidad:
```json
{
  "video_sha256": "...",
  "video_path": "input.mp4",
  "frames": 900,
  "fps": 30.0,
  "width": 640,
  "height": 480,
  "program_sha256": "...",
  "program_path": "program.lb",
  "trace_sha256": "...",
  "trace_path": "trace.json",
  "seed": 42,
  "config": { "max_frames": 0, "trace_enabled": true, "deterministic": true }
}
```

---

## 8. CLI — Comandos Deterministas

```bash
# Info del video
langbuster info video.mp4

# Ejecutar programa → trace.json + evidence.json
langbuster run video.mp4 program.lb --seed 42 --trace trace.json --evidence evidence.json

# Replay: re-ejecutar y verificar igualdad
langbuster replay video.mp4 program.lb --trace trace.json --output-trace trace.replay.json

# Comparar dos traces semánticamente
langbuster trace-diff trace_a.json trace_b.json

# Validar schema de trace
langbuster validate trace.json

# Observar / consultar
langbuster observe video.mp4 --trace trace.json --pixel 320,240,42
langbuster observe video.mp4 --trace trace.json --region 100,100,200,200,42
langbuster observe video.mp4 --trace trace.json --pixel-traj 100,100,0,99
langbuster observe video.mp4 --trace trace.json --trajectory 5
langbuster observe video.mp4 --trace trace.json --events --event-type MOTION
langbuster observe video.mp4 --trace trace.json --transition 0,1
langbuster observe video.mp4 --trace trace.json --crop 100,100,200,200,42
langbuster observe video.mp4 --trace trace.json --summary
```

---

## 9. Arquitectura

```
LangBuster/
├── video.py        # Decoder streaming (OpenCV + PIL)
├── temporal.py     # VideoState, FrameState, TemporalTransition, ops temporales
├── grammar.py      # Lexer + Parser (EBNF → AST)
├── runtime.py      # LangBusterVM (interpreter)
├── trace.py        # TemporalTrace, replay, validate, trace-diff, EvidencePackage
├── observe.py      # SpatiotemporalObserver (pixel, region, trajectory, events)
├── cli.py          # CLI commands (run, replay, trace-diff, validate, observe, info)
├── examples/       # 10 programas .lb de ejemplo
└── tests/          # Tests unitarios + integración
```

### Reutilización de FLOW
- **Bilineal sampling** → `sample_bilinear`, `sample_vec`
- **ExecutionTrace** → `TemporalTrace` (extendido con frame index `t`, timestamp)
- **Replay/Validation** → `trace_diff`, `validate_trace` (mismo patrón)
- **VisualObserver** → `SpatiotemporalObserver` (extendido a 3D: x,y,t)
- **Determinismo** → `seed` en config, RNG aislado
- **Provenance** → SHA256 de video, programa, trace, evidence

---

## 10. Criterio de Éxito

- [x] Runtime funcional (`LangBusterVM`)
- [x] Gramática + Parser (`grammar.py`)
- [x] CLI (`langbuster` command)
- [x] Video real ejecutable (mp4, webm, gif)
- [x] Frame individual inspeccionable (`INSPECT`, `OBSERVE`)
- [x] Consulta (x,y,t) (`PIXEL`, `DELTA`, `VECTOR`)
- [x] Transición temporal (`NEXT`, `PREV`, `JUMP`, `LOOP`)
- [x] Trace temporal (`TemporalTrace` con eventos + frame_hashes)
- [x] Provenance (SHA256 video, programa, trace, evidence)
- [x] Replay determinista (`replay` command)
- [x] Determinismo verificado (`seed`, `trace-diff`)
- [x] 10 ejemplos funcionales
- [x] Tests pasando
- [x] README/spec (este archivo)
- [x] Output machine-readable (JSON)
- [x] Arquitectura documentada

---

## 11. Diferencias Clave vs FLOW

| Aspecto | FLOW | LangBuster |
|---|---|---|
| Programa | Imagen (PNG) | Video (mp4/webm/gif) |
| Espacio | 2D (x,y) | 3D (x,y,t) |
| Tiempo | Ticks discretos internos | Frames de video + ticks |
| Estado | Partículas en campo | VideoState + frame cache |
| Transición | `step()` partícula | `NEXT`/`PREV`/`JUMP` frame |
| Observación | Píxel/región/partícula | Voxel/región-trayectoria/evento-ST |
| Trace | CSV + JSON eventos | JSON eventos + frame_hashes + evidence |

---

## 12. Instalación y Uso

```bash
# Dependencias
pip install opencv-python pillow numpy

# Desde la raiz del repositorio
cd IsycoLangs/langbuster

# Ver video
python -m langbuster.video video.mp4

# Ejecutar ejemplo
python -m langbuster.cli run video.mp4 examples/01_inspect_frame.lb --seed 42

# Observar
python -m langbuster.cli observe video.mp4 --trace trace.json --pixel 320,240,42

# Tests
python -m pytest tests/test_langbuster.py -v
```

---

## 13. Licencia / Espíritu

LangBuster es un **esolang experimental**. No es un procesador de video convencional.
El espacio computacional es **X × Y × TIME**.
La pregunta fundamental: *"¿Qué puede computarse cuando un frame no es solamente una imagen, sino un estado dentro de una trayectoria temporal?"*

Hecho para ser **extraño, reproducible, observable y experimental**.