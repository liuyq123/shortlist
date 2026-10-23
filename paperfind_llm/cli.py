"""Start the app: paperfind-llm [--port 8000] [--no-browser]"""

import argparse
import logging
import threading
import webbrowser

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="paperfind-llm",
        description="Daily arXiv papers screened by Claude, tuned by talking to an assistant.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"paperfind is running at {url}  (Ctrl+C to stop)")
    uvicorn.run("paperfind_llm.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
