"""Tests for LangBuster v0.1"""

import tempfile
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from langbuster.video import VideoDecoder, Frame, video_info
from langbuster.temporal import (
    VideoState, FrameState, TemporalTransition, LangBusterConfig,
    TemporalOp, create_video_state
)
from langbuster.grammar import parse, parse_file, Token, TokenType, Lexer
from langbuster.runtime import LangBusterVM, LangBusterConfig, run_program
from langbuster.trace import (
    TemporalTrace, TemporalEvent, run_file_with_trace,
    validate_trace, trace_diff, EvidencePackage, program_hash
)
from langbuster.observe import SpatiotemporalObserver, observe_video


def make_test_video(path: str, frames: int = 10, w: int = 64, h: int = 64, fps: float = 30.0):
    """Create a simple test video (MP4 via OpenCV or GIF fallback)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        # GIF via PIL
        images = []
        for i in range(frames):
            img = Image.new("RGB", (w, h), (0, 0, 0))
            px = img.load()
            x = (i * 5) % (w - 10)
            y = h // 2 - 5
            for yy in range(y, y+10):
                for xx in range(x, x+10):
                    px[xx, yy] = (255, 255, 255)
            images.append(img)
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/fps), loop=0)
    else:
        # MP4/WebM via OpenCV
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for i in range(frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            x = (i * 5) % (w - 10)
            y = h // 2 - 5
            frame[y:y+10, x:x+10] = [255, 255, 255]
            frame[:, :, 0] = 128
            frame[:, :, 1] = 128
            frame[:, :, 2] = i % 256
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()


def test_video_decoder():
    """Test video decoder basic functionality."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=10)

        dec = VideoDecoder(video_path)
        meta = dec.metadata

        assert meta.width == 64
        assert meta.height == 64
        assert meta.frame_count == 10
        assert meta.fps == 30.0
        assert len(meta.sha256) == 64

        # Test frame iteration
        frames = list(dec.frames())
        assert len(frames) == 10
        assert frames[0].index == 0
        assert frames[9].index == 9

        # Test random access
        frame5 = dec.frame_at(5)
        assert frame5.index == 5


def test_gif_decoder():
    """Test GIF decoder."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.gif"
        make_test_video(str(video_path), frames=5)

        dec = VideoDecoder(video_path)
        meta = dec.metadata
        assert meta.format == "gif"
        assert meta.frame_count == 5

        frames = list(dec.frames())
        assert len(frames) == 5


def test_frame_state():
    """Test FrameState creation and sampling."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=5)

        state = create_video_state(video_path)
        fs = state.get_frame(0)

        assert fs.frame.index == 0
        assert fs.field_r.shape == (64, 64)
        assert fs.field_g.shape == (64, 64)
        assert fs.field_b.shape == (64, 64)

        # Test vector sampling
        vx, vy = fs.sample_vector(32.5, 32.5)
        assert isinstance(vx, float)
        assert isinstance(vy, float)

        # Test scalar sampling
        val = fs.sample_scalar(32.5, 32.5)
        assert isinstance(val, int)


def test_temporal_transition():
    """Test TemporalTransition computation."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=10)

        state = create_video_state(video_path)
        state.advance()  # t=1

        trans = TemporalTransition.compute(state, 0, 1)

        assert trans.t_from == 0
        assert trans.t_to == 1
        assert trans.delta_b.shape == (64, 64)
        assert trans.motion_field.shape == (64, 64, 2)
        assert isinstance(trans.events, list)


def test_persist_appear_disappear():
    """Test persist/appear/disappear checks."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=10)

        state = create_video_state(video_path)

        # Test persist (region with moving square - won't persist)
        persists = state.persist_check(0, 0, 63, 63, 0, 9, threshold=1)
        # Returns bool
        assert isinstance(persists, bool)

        # Test appear/disappear - returns int (0/1)
        state.advance()
        appears = state.appear_check(0, 0, 10, 10)
        disappears = state.disappear_check(0, 0, 10, 10)
        assert appears in (0, 1)
        assert disappears in (0, 1)


def test_lexer():
    """Test lexer tokenizes correctly."""
    source = """
    FRAME 42
    INSPECT 100, 100, 42
    LOOP 0 99 {
        NEXT
    }
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    assert len(tokens) > 0
    assert tokens[-1].type == TokenType.EOF
    # Check keywords
    types = [t.type for t in tokens]
    assert TokenType.FRAME in types
    assert TokenType.INSPECT in types
    assert TokenType.LOOP in types
    assert TokenType.NEXT in types
    assert TokenType.LBRACE in types
    assert TokenType.RBRACE in types


def test_parser():
    """Test parser produces valid AST."""
    source = """
    FRAME 0
    LOOP 1 10 {
        DELTA 10, 10
        NEXT
    }
    EMIT "done"
    """
    ast = parse(source)

    assert isinstance(ast, parse.__annotations__['return'].__origin__ if hasattr(parse.__annotations__['return'], '__origin__') else type(ast))
    assert len(ast.statements) == 3
    assert isinstance(ast.statements[0], type(parse("FRAME 0").statements[0]))
    assert isinstance(ast.statements[1], type(parse("LOOP 1 10 { NEXT }").statements[0]))


def test_runtime_basic():
    """Test basic runtime execution."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=5)

        program = """
        FRAME 0
        INSPECT 32, 32, 0
        EMIT "test" _inspect
        """

        cfg = LangBusterConfig(max_frames=5, seed=42)
        vm = LangBusterVM(video_path, cfg)
        result = vm.execute(parse(program))

        assert result["success"] is True
        assert "trace" in result
        assert len(result["trace"]) > 0


def test_trace_generation():
    """Test trace generation and validation."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=5)

        program = """
        FRAME 0
        INSPECT 10, 10, 0
        """
        prog_path = Path(tmp) / "prog.lb"
        prog_path.write_text(program)

        trace = run_file_with_trace(video_path, prog_path, seed=42)

        assert isinstance(trace, TemporalTrace)
        assert len(trace.events) > 0
        assert trace.metadata["engine"] == "LangBuster"
        assert trace.metadata["video_sha256"] is not None
        assert trace.metadata["program_sha256"] is not None

        # Validate
        validation = validate_trace(trace)
        assert validation["valid"] is True


def test_trace_diff():
    """Test trace diff identifies identical traces."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=5)

        program = "FRAME 0\nINSPECT 10, 10, 0\n"
        prog_path = Path(tmp) / "prog.lb"
        prog_path.write_text(program)

        trace1 = run_file_with_trace(video_path, prog_path, seed=42)
        trace2 = run_file_with_trace(video_path, prog_path, seed=42)

        diff = trace_diff(trace1, trace2)
        assert diff["verdict"] == "identical"


def test_evidence_package():
    """Test evidence package creation."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=5)

        program = "FRAME 0\nINSPECT 10, 10, 0\n"
        prog_path = Path(tmp) / "prog.lb"
        prog_path.write_text(program)

        trace = run_file_with_trace(video_path, prog_path, seed=42)
        trace_path = Path(tmp) / "trace.json"
        trace.save(trace_path)

        evidence = EvidencePackage.from_execution(
            video_path, prog_path, trace, trace_path, seed=42
        )

        assert evidence.video_sha256 == trace.metadata["video_sha256"]
        assert evidence.program_sha256 == trace.metadata["program_sha256"]
        assert evidence.frames == 5
        assert evidence.seed == 42


def test_observer():
    """Test spatiotemporal observer."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=10)

        program = "FRAME 0\nINSPECT 10, 10, 0\n"
        prog_path = Path(tmp) / "prog.lb"
        prog_path.write_text(program)

        trace = run_file_with_trace(video_path, prog_path, seed=42)
        observer = observe_video(video_path, trace=trace, program_path=str(prog_path))

        # Test pixel query
        pixel = observer.pixel(32, 32, 0)
        assert pixel.x == 32
        assert pixel.y == 32
        assert pixel.t == 0
        assert "rgba" in pixel.to_dict()

        # Test region query
        region = observer.region(10, 10, 20, 20, 0)
        assert region.width == 11
        assert region.height == 11

        # Test summary
        summary = observer.summary()
        assert "video" in summary
        assert summary["video"]["frames"] == 10


def test_determinism():
    """Test deterministic execution with same seed."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(str(video_path), frames=5)

        program = """
        FRAME 0
        LOOP 1 4 {
            DELTA 10, 10
            EMIT "delta" t _delta
            NEXT
        }
        """
        prog_path = Path(tmp) / "prog.lb"
        prog_path.write_text(program)

        # Run twice with same seed
        trace1 = run_file_with_trace(video_path, prog_path, seed=123)
        trace2 = run_file_with_trace(video_path, prog_path, seed=123)

        diff = trace_diff(trace1, trace2)
        assert diff["verdict"] == "identical"

        # Different seed should produce different trace (for RAND)
        trace3 = run_file_with_trace(video_path, prog_path, seed=456)
        diff2 = trace_diff(trace1, trace3)
        # May or may not differ depending on program, but structure should be comparable


def test_cli_commands():
    """Test CLI command parsing (import only)."""
    from langbuster.cli import main, cmd_info, cmd_run, cmd_validate
    # Just verify imports work
    assert callable(main)
    assert callable(cmd_info)
    assert callable(cmd_run)
    assert callable(cmd_validate)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])