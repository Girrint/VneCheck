from __future__ import annotations

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VneCheckHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local VneCheck benchmark web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser")
    args = parser.parse_args()

    web_root = PROJECT_ROOT / "web"
    handler = functools.partial(VneCheckHandler, directory=str(web_root))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"VneCheck Model Lab: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
