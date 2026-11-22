#!/usr/bin/env python3
"""Start the edge station."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "::1", "localhost") and not os.environ.get("VERIQC_API_TOKEN"):
        raise SystemExit(
            "refusing a non-loopback bind without VERIQC_API_TOKEN; "
            "set the token or bind to 127.0.0.1"
        )
    import uvicorn

    uvicorn.run("veriqc.station.app:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
