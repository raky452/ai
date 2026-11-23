"""Entry point.

    TechnoForesightAssistant.exe              # command bar (stays in background, hotkey Alt+Space)
    python -m tf_assistant ask "question"     # answer in the terminal
    python -m tf_assistant do "open notepad"  # run a command
    python -m tf_assistant doctor             # check the AI engine and model
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
    parser = argparse.ArgumentParser(prog="tf-assistant", description="TechnoForesight offline AI assistant")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("ask", help="ask the AI a question").add_argument("text", nargs="+")
    sub.add_parser("do", help="run a command, same as typing it in the bar").add_argument("text", nargs="+")
    sub.add_parser("doctor", help="check that the local AI engine is ready")
    args = parser.parse_args(argv)

    cfg = load_config()
    llm = OllamaClient(cfg.ollama_url, cfg.model, cfg.assistant_name, language=cfg.response_language)
    if args.command in ("ask", "do") and cfg.response_language != "English" and cfg.indic_model:
        try:  # same rule as the window: the Indian-language model once it's downloaded
            wanted = cfg.indic_model if ":" in cfg.indic_model else cfg.indic_model + ":latest"
            if wanted in llm.installed_models():
                llm.model = cfg.indic_model
        except OllamaError:
            pass

    if args.command == "doctor":
        print(f"Version     : {__version__}")
        print(f"Config file : {CONFIG_PATH} ({'found' if CONFIG_PATH.exists() else 'using defaults'})")
        print(f"AI engine   : {cfg.ollama_url}")
        print(f"Model       : {cfg.model}")
        print(f"Hotkey      : {cfg.hotkey if sys.platform == 'win32' else 'Ctrl+Alt+Space (desktop shortcut)'}")
        print(f"Language    : {cfg.response_language}")
        print(f"Apps found  : {len(load_apps())}")
        problem = llm.check()
        print("Status      : " + (problem or "Ready ✔"))
        return 1 if problem else 0

    if args.command in ("ask", "do"):
        text = " ".join(args.text)
        try:
            if args.command == "ask":
                res = actions.Result(stream=llm.chat_stream(text))
            else:
                res = actions.run(route(text, load_apps(), cfg.connectors), cfg, llm)
            if res.message:
                print(res.message)
            for p in res.files:
                print("  ", p)
            if res.stream:
                from .llm import join_stream
                print(join_stream(res.stream))
        except (OllamaError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    from .ui import run_gui  # lazy: CLI works without a display

    return run_gui(cfg)


if __name__ == "__main__":
    sys.exit(main())
