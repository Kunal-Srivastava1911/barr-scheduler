import argparse

from .server import main

ap = argparse.ArgumentParser(description="BARR live dashboard")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-browser", action="store_true")
a = ap.parse_args()
main(a.host, a.port, not a.no_browser)
