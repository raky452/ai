"""Opens the real window for a moment (runs on the Windows CI machine; skipped where Tk is missing)."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("tkinter") is None, reason="no tkinter")


def test_window_opens_and_routes(monkeypatch):
    monkeypatch.setenv("TF_NO_SHORTCUT", "1")  # never touch the developer's real desktop shortcuts
    from tf_assistant import ui
    from tf_assistant.config import Config

    class _Instance:
        is_primary = True

        def listen(self, _cb):
            pass

    try:
        bar = ui.CommandBar(Config(ollama_url="http://127.0.0.1:9", hotkey="ctrl+alt+shift+f12"), _Instance())
    except Exception as e:  # no display (e.g. headless Linux)
        if "display" in str(e).lower():
            pytest.skip("no display")
        raise
    bar.entry.delete(0, "end")
    bar.entry.insert(0, "settings")
    bar.show_results([("Notepad", object())])
    assert bar.listbox.size() == 1
    bar.root.after(1500, bar.root.destroy)
    bar.run()
