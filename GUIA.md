# LangBuster — Video Ejecutable (GUIA)

Guía para la persona que opera esto a mano. En español a proposito; el resto
del repo va en ingles.

## 1. El comando que viniste a buscar

```powershell
py -m langbuster.cli pack video.mp4 programa.lbb -o video-ejecutable.mp4 --manifest artifact.json
py -m langbuster.cli inspect-video video-ejecutable.mp4
py -m langbuster.cli run-video video-ejecutable.mp4 --trace trace.json
```

## 2. La regla de oro

Un `.lbb` es un binario real que se extrae byte-a-byte y se verifica con dos
hashes antes de ejecutarse. Si `inspect-video` dice `CAPSULE: PASS`, el
programa recuperado es exactamente el que se empaqueto; no hay "version
aproximada". Nunca empaquetes con `python` desnudo (el shim muere con exit
2718): usa `py`.

## 3. Que hace cada comando (con salida real)

### Compilar un programa `.lb` a bytecode `.lbb`

```powershell
py -m langbuster.cli compile examples/doorman_webmcp_video.lb -o build/doorman.lbb --disasm
```

Salida real (verificada):

```
Compiled examples/doorman_webmcp_video.lb -> build/doorman.lbb
  Code: 47 bytes, Data: 30 bytes
  Entry: 0x0

Disassembly:
  0000: LOAD_I 0 0
  0004: MOV (10, 0)
  0006: SEEK 0
  0009: LOAD_I 0 960
  000D: LOAD_I 1 540
  0011: MOV (8, 0)
  0013: MOV (9, 1)
  0015: LOAD_I 2 0
  0019: MOV (10, 2)
  001B: SAMPLE_R 2
  001E: LOAD_I 1 0
  0022: MOV (0, 1)
  0024: LOAD_I 1 0
  0028: PUSH 1
  002A: EMIT 0
  002D: SNAPSHOT
  002E: HALT
```

### Empacar el video ejecutable

```powershell
py -m langbuster.cli pack build\doorman-final-v3.mp4 build\doorman.lbb -o build\doorman-langbuster-executable.mp4 --manifest build\doorman-langbuster-artifact.json
```

Salida real:

```
PACK: PASS
Output: C:\...\doorman-langbuster-executable.mp4
SHA256: 6af0bbf62bcb65fc20574009805060533cffd990b61e55cf02a3e35ebbcba4c9
Semantic program SHA256: 52ba7559167b6a8b3292a7baeae88e3144a979915610c947f964fadf21c33391
Packets: 9
Manifest: C:\...\doorman-langbuster-artifact.json
```

Se anade una secuencia de frames QR (4-5 s) al final del video. El manifest
JSON guarda el hash exacto del MP4, el del `.lbb` y el hash semantico.

### Inspeccionar (sin escribir el `.lbb`)

```powershell
py -m langbuster.cli inspect-video build\doorman-langbuster-executable.mp4
```

Salida real:

```
CAPSULE: PASS
LBB SHA256: ca66ac2fd19d82ccd0f18325032fd4f1d90c797133e6470a0e1c573bf970e4af
Semantic program SHA256: 52ba7559167b6a8b3292a7baeae88e3144a979915610c947f964fadf21c33391
Packets recovered: 9
Source video SHA256: 4012bcafa55842d6c3d3d2edd67568e7d14dc51e6771510f4e149b9720404af3
Downloaded video SHA256: 6af0bbf62bcb65fc20574009805060533cffd990b61e55cf02a3e35ebbcba4c9
```

### Recuperar y ejecutar

```powershell
py -m langbuster.cli run-video build\doorman-langbuster-executable.mp4 --trace trace.json
```

Salida real:

```
Trace: C:\...\trace.json
CAPSULE: PASS
Semantic program SHA256: 52ba7559167b6a8b3292a7baeae88e3144a979915610c947f964fadf21c33391
Halted: True
Instructions: 17
Final T: 0
```

## 4. Como leer la salida

| Codigo | Significado | Que hacer |
|---|---|---|
| `PACK: PASS` | Capsula escrita y verificable | sigue |
| `CAPSULE: PASS` | `inspect`/`run` recuperaron todos los fragmentos y los hashes cuadran | listo para ejecutar |
| `Halted: True` | la VM llego a `HALT` de forma controlada | exito |
| `Instructions: N` | numero de instrucciones ejecutadas | saneza / comparar entre runs |
| `No complete LangBuster visual capsule found` | no se pudo recuperar el conjunto completo de fragmentos | el video se perdio/transcodifico a baja resolucion; usa el original |
| `Cannot open video` | ruta inexistente o archivo no decodificable | revisa la ruta |
| `Video hash mismatch` (WARNING) | el video no es el mismo que se empaqueto | esperable tras transcodificar; el hash semantico sigue siendo la verdad |

## 5. Los trampas

- **Resolucion nativa es requisito.** La capsula de Doorman sobrevive a 1080p
  (PASS) y NO a 480p (control negativo real, 2026-08-30). Manten el video a
  la resolucion de origen para que el QR sea legible. No "optimices" subiendo
  el video ejecutable a 480p.
- **Nunca edites el manifest ni reuses una ruta de salida** para otro run: el
  hash del MP4 debe cuadrar con lo que el manifest declara.
- **`python` desnudo muere (exit 2718)** en esta maquina. Usa `py`.
- **`--chunk-size` bajo = mas paquetes pero QR mas legibles.** El default es
  64; no lo subas a 128+ para videos con contenido denso, el decodificador
  OpenCV empieza a fallar.
- **`capture_screenshot` de OBS exige `imageHeight >= 8`**; pasa 360 como alto
  si usas la API de screenshot en diagnostico.
- **El codigo `pack` usa el `semantic_program_hash`** (hash de code+data
  canonico), no el hash de los bytes del archivo `.lbb`, asi el mismo programa
  tiene identidad estable aunque el video cambie.

## 6. Como se hace el video Doorman (referencia completa)

1. `setup_obs.py` configura una fuente OBS `window_capture` aislada
   ("Doorman WebMCP Demo Capture"), desactiva el monitor completo y usa
   Chromium con `--disable-gpu`.
2. `shitvid ... --record` graba la demo guiada (nueve segmentos de narracion
   Piper) y sella `recording_sha256` en el manifest.
3. `shitvid ... --render --from-run <manifest-record>` mezcla audio + captions
   y produce el MP4 narrado de ~52 s.
4. `langbuster pack` anade la capsula visual final (~5 s) y genera el manifest
   de artefacto con los tres hashes.

Todo lo anterior esta ejecutado y verificado en la maquina de origen
(2026-08-30). El flujo YouTube real (subir, descargar, ejecutar) queda como
`NOT_DEMONSTRATED` hasta que se pruebe en la plataforma.