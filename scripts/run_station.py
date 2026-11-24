#!/usr/bin/env python3
"""Start the edge station."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("VERIQC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VERIQC_PORT", "8080")))
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--source", choices=["folder", "video", "camera"], default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.source:
        if args.source in ("folder", "video") and not args.source_path:
            raise SystemExit("--source-path is required for folder/video sources")
        source = {"type": args.source, "interval_s": args.interval, "loop": True}
        if args.source in ("folder", "video"):
            source["path"] = args.source_path
        else:
            source["camera"] = {"kind": "usb", "index": int(args.source_path or 0)}
        os.environ["VERIQC_SOURCE_CONFIG"] = json.dumps(source)
    if args.host not in ("127.0.0.1", "::1", "localhost") and not os.environ.get("VERIQC_API_TOKEN"):
        raise SystemExit(
            "refusing a non-loopback bind without VERIQC_API_TOKEN; "
            "set the token or bind to 127.0.0.1"
        )
    import uvicorn

    uvicorn.run("veriqc.station.app:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
