"""Explicit-parameter CLI for adding a MoTeC video sync point."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout

from .video_sync import VideoAlignment, export_motec_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="track-telemetry motec-sync",
        description="Write an explicit video-to-telemetry alignment into a MoTeC LDX file.",
    )
    parser.add_argument("--ld", required=True, help="MoTeC LD file to synchronize")
    parser.add_argument("--rcz", required=True, help="RaceChrono source log")
    parser.add_argument("--video", required=True, help="Video file to bind to the LDX")
    parser.add_argument(
        "--offset",
        required=True,
        type=float,
        help="Seconds: RCZ time = source video time + offset",
    )
    parser.add_argument("--stint", type=int, default=None, help="RCZ stint index")
    parser.add_argument(
        "--video-start",
        type=float,
        default=0.0,
        help="Source-video time represented by output video time zero (seconds)",
    )
    parser.add_argument(
        "--video-start-utc",
        type=float,
        default=None,
        help="Optional Unix epoch seconds for source-video time zero",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Video playback rate (only 1.0 is currently supported)",
    )
    parser.add_argument(
        "--view-name", default="Camera View 1", help="MoTeC video view name",
    )
    parser.add_argument("--ldx", help="Source LDX (default: LD path with .ldx extension)")
    parser.add_argument("--output", help="Destination LDX (default: source LDX)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    alignment = VideoAlignment(
        rcz_path=args.rcz,
        video_path=args.video,
        stint=args.stint,
        offset=args.offset,
        rate=args.rate,
        video_start=args.video_start,
        video_start_utc=args.video_start_utc,
    )
    try:
        with redirect_stdout(sys.stderr):
            report = export_motec_sync(
                args.ld,
                alignment,
                ldx_path=args.ldx,
                output_path=args.output,
                view_name=args.view_name,
                dry_run=args.dry_run,
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
