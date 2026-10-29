"""Phục vụ gallery cục bộ, hỗ trợ chạy ẩn bằng pythonw trên Windows."""

import argparse
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    # pythonw không có stderr; BaseHTTPRequestHandler cần luồng log để trả response.
    with (root / "gallery-server.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(root)))
        print(f"Gallery: http://127.0.0.1:{args.port}/gallery.html", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
