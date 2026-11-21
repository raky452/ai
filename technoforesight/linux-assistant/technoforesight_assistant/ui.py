"""GTK 4 command-bar window. Bind it to a keyboard shortcut (e.g. Super+Space) for instant access."""

from __future__ import annotations

import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk, Pango  # noqa: E402

from . import APP_ID, actions, files  # noqa: E402
from .apps import DesktopApp, launch, load_apps  # noqa: E402
from .config import Config  # noqa: E402
from .llm import OllamaClient, OllamaError  # noqa: E402
from .router import route  # noqa: E402

CSS = """
window.tf { background-color: rgba(15, 23, 42, 0.96); border-radius: 18px; }
entry.tf-bar { font-size: 20px; padding: 12px 16px; border-radius: 14px; }
label.tf-status { color: #7dd3fc; }
textview.tf-answer, textview.tf-answer text { background: transparent; color: #f1f5f9; font-size: 15px; }
row.tf-row { padding: 6px 10px; border-radius: 10px; }
"""


class CommandWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, cfg: Config, llm: OllamaClient):
        super().__init__(application=app, title=cfg.assistant_name)
        self.cfg, self.llm = cfg, llm
        self.apps: list[DesktopApp] = load_apps()
        self.add_css_class("tf")
        self.set_default_size(720, 120)
        self.set_decorated(False)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for side in ("top", "bottom", "start", "end"):
            getattr(box, f"set_margin_{side}")(16)
        self.set_child(box)

        self.entry = Gtk.Entry(placeholder_text="Open apps, find files, summarise ~/doc.pdf, or ask anything…")
        self.entry.add_css_class("tf-bar")
        self.entry.connect("activate", self.on_submit)
        self.entry.connect("changed", self.on_changed)
        box.append(self.entry)

        self.status = Gtk.Label(xalign=0, wrap=True)
        self.status.add_css_class("tf-status")
        box.append(self.status)

        self.results = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.results.connect("row-activated", self.on_row_activated)
        self.results_scroll = Gtk.ScrolledWindow(min_content_height=0, max_content_height=320,
                                                 propagate_natural_height=True, child=self.results,
                                                 visible=False)
        box.append(self.results_scroll)

        self.answer = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.answer.add_css_class("tf-answer")
        self.answer_scroll = Gtk.ScrolledWindow(min_content_height=0, max_content_height=420,
                                                propagate_natural_height=True, child=self.answer, visible=False)
        box.append(self.answer_scroll)

        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.on_key)
        self.add_controller(keys)

        self._generation = 0  # lets a new request cancel display of an older stream

    # ---- events -----------------------------------------------------------------
    def on_key(self, _ctrl, keyval, _code, _state):
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        if keyval == Gdk.KEY_Down and self.results.get_first_child():
            self.results.select_row(self.results.get_row_at_index(0))
            self.results.get_row_at_index(0).grab_focus()
            return True
        return False

    def on_changed(self, entry: Gtk.Entry):
        text = entry.get_text().strip()
        self.answer_scroll.set_visible(False)
        self.status.set_text("")
        matches = [a for a in self.apps if text and a.matches(text)][:8]
        self.show_results([(a.name, a) for a in matches])

    def on_row_activated(self, _box, row):
        target = row.target
        if isinstance(target, DesktopApp):
            launch(target)
            self.close()
        elif isinstance(target, Path):
            files.open_path(target)
            self.close()

    def on_submit(self, entry: Gtk.Entry):
        text = entry.get_text().strip()
        if not text:
            return
        action = route(text, self.apps)
        self._generation += 1
        generation = self._generation
        self.status.set_text("Thinking…")

        def worker():
            try:
                msg, stream, found = actions.run(action, self.cfg, self.llm)
                GLib.idle_add(self.status.set_text, msg)
                if found:
                    GLib.idle_add(self.show_results, [(str(p).replace(str(Path.home()), "~"), p) for p in found])
                if stream is None:
                    if msg.startswith(("Opening", "Searching")):
                        GLib.idle_add(self.close)
                    return
                GLib.idle_add(self.start_answer)
                for piece in stream:
                    if generation != self._generation:
                        return
                    GLib.idle_add(self.append_answer, piece)
            except (OllamaError, OSError, RuntimeError) as e:
                GLib.idle_add(self.status.set_text, str(e))

        threading.Thread(target=worker, daemon=True).start()

    # ---- view helpers -----------------------------------------------------------
    def show_results(self, items):
        while (child := self.results.get_first_child()) is not None:
            self.results.remove(child)
        for label, target in items:
            row = Gtk.ListBoxRow()
            row.add_css_class("tf-row")
            row.target = target
            row.set_child(Gtk.Label(label=label, xalign=0, ellipsize=Pango.EllipsizeMode.END))
            self.results.append(row)
        self.results_scroll.set_visible(bool(items))

    def start_answer(self):
        self.answer.get_buffer().set_text("")
        self.answer_scroll.set_visible(True)
        self.results_scroll.set_visible(False)
        if self.status.get_text() in ("Thinking…", ""):
            self.status.set_text("")

    def append_answer(self, piece: str):
        buf = self.answer.get_buffer()
        buf.insert(buf.get_end_iter(), piece)


class AssistantApp(Gtk.Application):
    def __init__(self, cfg: Config):
        super().__init__(application_id=APP_ID)
        self.cfg = cfg
        self.llm = OllamaClient(cfg.ollama_url, cfg.model, cfg.assistant_name)

    def do_startup(self):
        Gtk.Application.do_startup(self)
        provider = Gtk.CssProvider()
        if hasattr(provider, "load_from_string"):  # GTK 4.12+
            provider.load_from_string(CSS)
        else:  # older GTK 4 bindings differ in whether they take the length
            data = CSS.encode()
            try:
                provider.load_from_data(data, len(data))
            except TypeError:
                provider.load_from_data(data)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def do_activate(self):
        # Single instance: pressing the shortcut again just re-shows the window.
        win = self.props.active_window or CommandWindow(self, self.cfg, self.llm)
        win.present()
        if problem := self.llm.check():
            win.status.set_text("⚠ " + problem)


def run_gui(cfg: Config) -> int:
    return AssistantApp(cfg).run(None)
