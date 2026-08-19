"""
LangBuster v0.1 — Spatiotemporal Video Esolang
SPACE × TIME computation on video.
"""

from .video import VideoDecoder, Frame, VideoMetadata, decode_video, video_info
from .temporal import (
    VideoState, FrameState, TemporalTransition, LangBusterConfig,
    TemporalOp, TEMPORAL_MNEMONIC, create_video_state
)
from .grammar import parse, parse_file, Program, Token, TokenType
from .runtime import LangBusterVM, LangBusterConfig, run_program, run_file
from .trace import (
    TemporalTrace, TemporalEvent, run_file_with_trace,
    validate_trace, trace_diff, EvidencePackage
)
from .observe import SpatiotemporalObserver, observe_video, SpatiotemporalPixel, SpatiotemporalRegion

__version__ = "0.1"
__all__ = [
    "VideoDecoder", "Frame", "VideoMetadata", "decode_video", "video_info",
    "VideoState", "FrameState", "TemporalTransition", "LangBusterConfig",
    "TemporalOp", "TEMPORAL_MNEMONIC", "create_video_state",
    "parse", "parse_file", "Program", "Token", "TokenType",
    "LangBusterVM", "run_program", "run_file",
    "TemporalTrace", "TemporalEvent", "run_file_with_trace",
    "validate_trace", "trace_diff", "EvidencePackage",
    "SpatiotemporalObserver", "observe_video", "SpatiotemporalPixel", "SpatiotemporalRegion",
]