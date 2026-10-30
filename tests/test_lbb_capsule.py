import json
from pathlib import Path

import cv2
import pytest

from langbuster.capsule import (
    build_capsule,
    extract_video,
    pack_video,
    packetize,
    render_packet_frame,
    semantic_program_hash,
)
from langbuster.lbb_compiler import compile_string
from langbuster.lbb_format import (
    Instruction,
    LBBProgram,
    Op,
    decode_instruction,
    encode_instruction,
)
from langbuster.lbb_vm import run_lbb
from test_langbuster import make_test_video


@pytest.mark.parametrize("reg", range(16))
def test_standalone_register_round_trip(reg):
    encoded = encode_instruction(Instruction(Op.INC, [reg]))
    decoded, end = decode_instruction(encoded)
    assert end == len(encoded)
    assert decoded.operands == [reg]


@pytest.mark.parametrize("value", [-32768, -1, 0, 32767, 65535])
def test_i16_round_trip_uses_twos_complement(value):
    encoded = encode_instruction(Instruction(Op.PUSH_I, [value]))
    decoded, _ = decode_instruction(encoded)
    expected = value if value <= 32767 else value - 65536
    assert decoded.operands == [expected]


def test_lbb_vm_executes_compiled_emit_and_snapshot(tmp_path):
    program = compile_string('EMIT "capsule_verified"\nSNAPSHOT\n')
    lbb_path = tmp_path / "program.lbb"
    program.save(lbb_path)

    result = run_lbb(lbb_path)

    assert result["halted"] is True
    assert [event["type"] for event in result["trace_events"]] == ["EMIT", "SNAPSHOT"]
    assert result["trace_events"][0]["payload"]["event"] == "capsule_verified"


def test_lbb_load_rejects_trailing_bytes(tmp_path):
    path = tmp_path / "bad.lbb"
    program = compile_string("SNAPSHOT\n")
    program.save(path)
    path.write_bytes(path.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="size mismatch"):
        LBBProgram.load(path)


def test_visual_packet_decodes_from_rendered_frame(tmp_path):
    source = tmp_path / "source.mp4"
    make_test_video(str(source), frames=2, w=1280, h=720)
    lbb = tmp_path / "program.lbb"
    compile_string("SNAPSHOT\n").save(lbb)
    container, _ = build_capsule(source, lbb)
    packet = packetize(container)[0]
    frame = render_packet_frame(packet, 1280, 720)

    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(frame)

    assert decoded == packet


def test_pack_extract_and_manifest_round_trip(tmp_path):
    source = tmp_path / "source.mp4"
    make_test_video(str(source), frames=5, w=1280, h=720)
    lbb = tmp_path / "program.lbb"
    compile_string('EMIT "video_capsule_ok"\nSNAPSHOT\n').save(lbb)
    output = tmp_path / "executable.mp4"

    artifact = pack_video(source, lbb, output, capsule_seconds=1.0)
    recovered_path = tmp_path / "recovered.lbb"
    recovered, report = extract_video(output, recovered_path)

    assert recovered == lbb.read_bytes()
    assert recovered_path.read_bytes() == lbb.read_bytes()
    assert report["transport_status"] == "PASS"
    assert report["semantic_program_sha256"] == semantic_program_hash(lbb.read_bytes())
    manifest = json.loads(Path(artifact["manifest"]).read_text(encoding="utf-8"))
    assert manifest["sha256"] == artifact["sha256"]
    assert manifest["semantic_program_sha256"] == report["semantic_program_sha256"]
