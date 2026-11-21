"""Entry point.

    technoforesight-assistant                  # open the command bar (GUI)
    technoforesight-assistant ask "question"   # answer in the terminal
    technoforesight-assistant do "open firefox"
    technoforesight-assistant doctor           # check the AI engine and model
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, actions
from .apps import load_apps
from .config import CONFIG_PATH, load_config
from .llm import OllamaClient, OllamaError
from .router import route


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="technoforesight-assistant", description="Offline AI assistant")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    p_ask = sub.add_parser("ask", help="ask the AI a question")
    p_ask.add_argument("text", nargs="+")
    p_do = sub.add_parser("do", help="run a command, same as typing it in the bar")
    p_do.add_argument("text", nargs="+")
    sub.add_parser("doctor", help="check that the local AI engine is ready")
    args = parser.parse_args(argv)

    cfg = load_config()
    llm = OllamaClient(cfg.ollama_url, cfg.model, cfg.assistant_name)

    if args.command == "doctor":
        print(f"Config file : {CONFIG_PATH} ({'found' if CONFIG_PATH.exists() else 'using defaults'})")
        print(f"AI engine   : {cfg.ollama_url}")
        print(f"Model       : {cfg.model}")
        print(f"Apps found  : {len(load_apps())}")
        problem = llm.check()
        print("Status      : " + (problem or "Ready ✔"))
        return 1 if problem else 0

    if args.command in ("ask", "do"):
        text = " ".join(args.text)
        action = route(text, load_apps()) if args.command == "do" else None
        try:
            if action is None:
                stream, msg, found = llm.chat_stream(text), "", []
            else:
                msg, stream, found = actions.run(action, cfg, llm)
            if msg:
                print(msg)
            for path in found:
                print("  ", path)
            if stream:
                for piece in stream:
                    print(piece, end="", flush=True)
                print()
        except (OllamaError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    from .ui import run_gui  # imported lazily so CLI mode works without GTK

    return run_gui(cfg)


if __name__ == "__main__":
    sys.exit(main())
