"""Start the app. Locally: uv run --locked --env-file .env python app.py (opens a browser tab unless --no-browser)."""

import argparse
import logging
import threading
import webbrowser

import uvicorn

from shortlist import config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shortlist",
        description="Daily arXiv papers screened by an AI agent, tuned by talking to an assistant.",
    )
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"
    if not (args.no_browser or config.HOSTED):
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"Shortlist is running at {url}  (Ctrl+C to stop)", flush=True)
    uvicorn.run("shortlist.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
