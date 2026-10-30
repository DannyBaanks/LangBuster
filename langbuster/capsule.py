"""Loss-tolerant visual transport for embedding LBB bytecode in video."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .lbb_format import LBBHeader


CAPSULE_MAGIC = b"LBVC\x01"
CAPSULE_SCHEMA = "langbuster-video-capsule/1"
ARTIFACT_SCHEMA = "langbuster-video-artifact/1"
PACKET_PREFIX = "LBV1"
DEFAULT_CHUNK_SIZE = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_program_hash(lbb_bytes: bytes) -> str:
    header = LBBHeader.unpack(lbb_bytes[:LBBHeader.HEADER_SIZE])
    expected = LBBHeader.HEADER_SIZE + header.code_size + header.data_size
    if len(lbb_bytes) != expected:
        raise ValueError(
            f"LBB size mismatch: header declares {expected} bytes, payload has {len(lbb_bytes)}"
        )
    code_start = LBBHeader.HEADER_SIZE
    code_end = code_start + header.code_size
    canonical = (
        b"LangBuster semantic program v1\0"
        + struct.pack(">II", header.code_size, header.data_size)
        + lbb_bytes[code_start:code_end]
        + lbb_bytes[code_end:expected]
    )
    return sha256_bytes(canonical)


def build_capsule(source_video: str | Path, lbb_path: str | Path) -> tuple[bytes, dict[str, Any]]:
    source_video = Path(source_video)
    lbb_path = Path(lbb_path)
    lbb_bytes = lbb_path.read_bytes()
    header = {
        "schema": CAPSULE_SCHEMA,
        "lbb_name": lbb_path.name,
        "lbb_sha256": sha256_bytes(lbb_bytes),
        "semantic_program_sha256": semantic_program_hash(lbb_bytes),
        "source_video_name": source_video.name,
        "source_video_sha256": sha256_file(source_video),
    }
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    container = CAPSULE_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + lbb_bytes
    return container, header


def parse_capsule(container: bytes) -> tuple[dict[str, Any], bytes]:
    if not container.startswith(CAPSULE_MAGIC) or len(container) < len(CAPSULE_MAGIC) + 4:
        raise ValueError("Invalid LangBuster video capsule magic")
    offset = len(CAPSULE_MAGIC)
    header_size = struct.unpack(">I", container[offset:offset + 4])[0]
    header_start = offset + 4
    header_end = header_start + header_size
    if header_end > len(container):
        raise ValueError("Truncated LangBuster video capsule header")
    header = json.loads(container[header_start:header_end].decode("ascii"))
    if header.get("schema") != CAPSULE_SCHEMA:
        raise ValueError(f"Unsupported capsule schema: {header.get('schema')}")
    lbb_bytes = container[header_end:]
    if sha256_bytes(lbb_bytes) != header.get("lbb_sha256"):
        raise ValueError("Recovered LBB SHA-256 mismatch")
    if semantic_program_hash(lbb_bytes) != header.get("semantic_program_sha256"):
        raise ValueError("Recovered semantic program SHA-256 mismatch")
    return header, lbb_bytes


def packetize(
    container: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    frame_size: tuple[int, int] | None = None,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    container_hash = sha256_bytes(container)
    capsule_id = container_hash[:16]
    chunks = [container[i:i + chunk_size] for i in range(0, len(container), chunk_size)]
    packets = []
    for index, chunk in enumerate(chunks):
        encoded = base64.b32encode(chunk).decode("ascii").rstrip("=")
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        base = (
            f"{PACKET_PREFIX}:{capsule_id.upper()}:{index}:{len(chunks)}:"
            f"{container_hash.upper()}:{crc:08X}"
        )
        for salt in range(256):
            candidate = f"{base}:{salt:02X}:{encoded}"
            if frame_size:
                candidate_image = render_packet_frame(candidate, *frame_size)
            else:
                qr = _qr_matrix(candidate)
                candidate_image = cv2.resize(
                    qr, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST
                )
            decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(candidate_image)
            if decoded == candidate:
                packets.append(candidate)
                break
        else:
            raise ValueError(f"Could not create a decodable QR packet at index {index}")
    return packets


def _parse_packet(text: str) -> dict[str, Any]:
    parts = text.split(":", 7)
    if len(parts) != 8 or parts[0] != PACKET_PREFIX:
        raise ValueError("Not a LangBuster visual packet")
    _, capsule_id, index_text, total_text, container_hash, crc_text, salt, encoded = parts
    capsule_id = capsule_id.lower()
    container_hash = container_hash.lower()
    index, total = int(index_text), int(total_text)
    if total <= 0 or not 0 <= index < total:
        raise ValueError("Invalid visual packet index")
    if len(salt) != 2:
        raise ValueError("Invalid visual packet salt")
    padding = "=" * (-len(encoded) % 8)
    chunk = base64.b32decode(encoded + padding)
    if zlib.crc32(chunk) & 0xFFFFFFFF != int(crc_text, 16):
        raise ValueError("Visual packet CRC mismatch")
    if capsule_id != container_hash[:16] or len(container_hash) != 64:
        raise ValueError("Visual packet capsule identity mismatch")
    return {
        "capsule_id": capsule_id,
        "index": index,
        "total": total,
        "container_sha256": container_hash,
        "chunk": chunk,
    }


def _qr_matrix(text: str) -> np.ndarray:
    params = cv2.QRCodeEncoder_Params()
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
    params.mode = cv2.QRCodeEncoder_MODE_ALPHANUMERIC
    matrix = cv2.QRCodeEncoder_create(params).encode(text)
    return cv2.copyMakeBorder(matrix, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)


def render_packet_frame(text: str, width: int, height: int) -> np.ndarray:
    if width < 320 or height < 240:
        raise ValueError("Visual capsules require at least 320x240 video")
    packet = _parse_packet(text)
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    marker = max(24, min(width, height) // 16)
    frame[:marker, :marker] = (255, 0, 255)

    qr = _qr_matrix(text)
    target = min(width - 40, height - 100)
    scale = max(1, target // qr.shape[0])
    qr = cv2.resize(qr, (qr.shape[1] * scale, qr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    y0 = (height - qr.shape[0]) // 2 + 15
    x0 = (width - qr.shape[1]) // 2
    frame[y0:y0 + qr.shape[0], x0:x0 + qr.shape[1]] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)

    title = "LANGBUSTER EXECUTABLE VIDEO"
    detail = f"CAPSULE {packet['index'] + 1}/{packet['total']}  ID {packet['capsule_id']}"
    cv2.putText(frame, title, (marker + 18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (marker + 18, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 2, cv2.LINE_AA)
    return frame


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}")


def pack_video(
    source_video: str | Path,
    lbb_path: str | Path,
    output: str | Path,
    *,
    manifest_path: str | Path | None = None,
    capsule_seconds: float = 3.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    source_video, lbb_path, output = Path(source_video), Path(lbb_path), Path(output)
    if not source_video.is_file() or not lbb_path.is_file():
        raise FileNotFoundError("Source video and LBB program must exist")
    if source_video.resolve() == output.resolve():
        raise ValueError("Output must not overwrite the source video")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to pack an executable video")
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot read source video dimensions: {source_video}")

    container, capsule_header = build_capsule(source_video, lbb_path)
    packets = packetize(container, chunk_size, frame_size=(width, height))
    seconds_per_packet = max(0.5, capsule_seconds / len(packets))

    with tempfile.TemporaryDirectory(prefix="langbuster-capsule-") as tmp_name:
        tmp = Path(tmp_name)
        concat_lines = []
        for index, packet in enumerate(packets):
            image_path = tmp / f"packet-{index:04d}.png"
            if not cv2.imwrite(str(image_path), render_packet_frame(packet, width, height)):
                raise RuntimeError(f"Could not write capsule frame: {image_path}")
            escaped = image_path.as_posix().replace("'", "'\\''")
            concat_lines.extend([f"file '{escaped}'", f"duration {seconds_per_packet:.6f}"])
        concat_lines.append(f"file '{(tmp / f'packet-{len(packets) - 1:04d}.png').as_posix()}'")
        concat_file = tmp / "capsule-frames.txt"
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        capsule_video = tmp / "capsule.mp4"

        _run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-vf", f"fps={fps:.6f},format=yuv420p", "-c:v", "libx264", "-preset", "medium",
            "-crf", "8", "-an", str(capsule_video),
        ])
        filter_graph = (
            f"[0:v]fps={fps:.6f},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,settb=AVTB[v0];"
            f"[1:v]fps={fps:.6f},scale={width}:{height},setsar=1,settb=AVTB[v1];"
            "[v0][v1]concat=n=2:v=1:a=0[v]"
        )
        _run([
            "ffmpeg", "-y", "-i", str(source_video), "-i", str(capsule_video),
            "-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ])

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "artifact": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "source_video": str(source_video.resolve()),
        "source_video_sha256": capsule_header["source_video_sha256"],
        "lbb": str(lbb_path.resolve()),
        "lbb_sha256": capsule_header["lbb_sha256"],
        "semantic_program_sha256": capsule_header["semantic_program_sha256"],
        "capsule_sha256": sha256_bytes(container),
        "transport": {
            "format": "QR",
            "error_correction": "Reed-Solomon H",
            "packet_crc": "CRC32",
            "packets": len(packets),
            "chunk_bytes": chunk_size,
            "seconds_per_packet": seconds_per_packet,
        },
    }
    manifest = Path(manifest_path) if manifest_path else output.with_suffix(output.suffix + ".langbuster.json")
    manifest.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifact["manifest"] = str(manifest.resolve())
    return artifact


def extract_video(
    video_path: str | Path,
    output: str | Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    scan_step = max(1, int(round(fps / 8.0)))
    detector = cv2.QRCodeDetector()
    groups: dict[str, dict[str, Any]] = {}
    frame_index = 0
    decoded_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % scan_step:
            frame_index += 1
            continue
        marker = frame[:max(8, frame.shape[0] // 20), :max(8, frame.shape[1] // 20)]
        mean_b, mean_g, mean_r = marker.mean(axis=(0, 1))
        if mean_b > 130 and mean_r > 130 and mean_g < 120:
            text, _, _ = detector.detectAndDecode(frame)
            if text.startswith(PACKET_PREFIX + ":"):
                packet = _parse_packet(text)
                group = groups.setdefault(packet["capsule_id"], {
                    "total": packet["total"],
                    "container_sha256": packet["container_sha256"],
                    "chunks": {},
                })
                if group["total"] != packet["total"] or group["container_sha256"] != packet["container_sha256"]:
                    raise ValueError("Conflicting visual packets for one capsule")
                group["chunks"][packet["index"]] = packet["chunk"]
                decoded_frames += 1
                if len(group["chunks"]) == group["total"]:
                    break
        frame_index += 1
    capture.release()

    complete = [group for group in groups.values() if len(group["chunks"]) == group["total"]]
    if not complete:
        raise ValueError("No complete LangBuster visual capsule found")
    if len(complete) > 1:
        raise ValueError("Multiple complete LangBuster visual capsules found")
    group = complete[0]
    container = b"".join(group["chunks"][index] for index in range(group["total"]))
    if sha256_bytes(container) != group["container_sha256"]:
        raise ValueError("Recovered capsule SHA-256 mismatch")
    header, lbb_bytes = parse_capsule(container)
    if output is not None:
        Path(output).write_bytes(lbb_bytes)
    report = {
        **header,
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "capsule_sha256": group["container_sha256"],
        "packets_recovered": group["total"],
        "decoded_packet_frames": decoded_frames,
        "frames_scanned": frame_index + 1,
        "transport_status": "PASS",
    }
    return lbb_bytes, report


__all__ = [
    "ARTIFACT_SCHEMA", "CAPSULE_SCHEMA", "build_capsule", "extract_video",
    "pack_video", "packetize", "parse_capsule", "render_packet_frame",
    "semantic_program_hash", "sha256_file",
]
