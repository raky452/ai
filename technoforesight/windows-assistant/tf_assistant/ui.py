"""Command-bar window (tkinter: ships with Python, no extra dependencies)."""

from __future__ import annotations

import ctypes
import queue
import sys
import time
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk

from . import APP_NAME, WEBSITE, actions, files
from .apps import App, launch, load_apps
from .config import DEFAULT_MODEL, FAST_MODEL, SLOW_TOKENS_PER_SEC, Config, save_setting, user_set
from .llm import OllamaClient, OllamaError
from .router import route
from .winsys import SingleInstance, start_hotkey_listener

BG, PANEL, FG, MUTED, ACCENT, ERR = "#0f172a", "#1e293b", "#f1f5f9", "#94a3b8", "#38bdf8", "#fca5a5"
FONT = "Segoe UI"
WIDTH = 760


def _resource(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets" / name


class CommandBar:
    def __init__(self, cfg: Config, instance: SingleInstance):
        self.cfg = cfg
        self.llm = OllamaClient(cfg.ollama_url, cfg.model, cfg.assistant_name)
        self.apps: list[App] = []
        self.results: list[tuple[str, object]] = []
        self.events: queue.Queue = queue.Queue()
        self._generation = 0

        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        try:
            self.root.iconbitmap(default=str(_resource("icon.ico")))
        except tk.TclError:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        # Scale pixel sizes for high-DPI screens (fonts already scale; widths must too)
        self.scale = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        self.width = int(WIDTH * self.scale)
        self.busy_since: float | None = None   # set while waiting for the AI's first word
        self.streaming = False
        self._build()
        self._place()

        # Background work never touches tkinter directly; it posts to self.events.
        threading.Thread(target=self._load_apps, daemon=True).start()
        threading.Thread(target=self._check_engine, daemon=True).start()
        self.hotkey_ok = start_hotkey_listener(cfg.hotkey, lambda: self.events.put(("show",))) is not None
        instance.listen(lambda: self.events.put(("show",)))
        self.root.after(80, self._pump)
        hint = (f"Tip: press {cfg.hotkey.title()} anytime to open me." if self.hotkey_ok
                else f"Couldn't register {cfg.hotkey}; another app uses it. Change 'hotkey' in config.toml.")
        self.set_status(hint)
        self.show()

    # ---------- layout ----------
    def _build(self):
        outer = tk.Frame(self.root, bg=BG, padx=18, pady=16)
        outer.pack(fill="both", expand=True)

        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="✦ TechnoForesight", fg=ACCENT, bg=BG, font=(FONT, 11, "bold")).pack(side="left")
        tk.Label(head, text="offline AI · nothing leaves your PC", fg=MUTED, bg=BG, font=(FONT, 9)).pack(side="left", padx=10)

        self.entry = tk.Entry(outer, font=(FONT, 17), bg=PANEL, fg=FG, insertbackground=FG,
                              relief="flat", highlightthickness=2, highlightcolor=ACCENT, highlightbackground=PANEL)
        self.entry.pack(fill="x", pady=(10, 6), ipady=10)
        self.entry.bind("<Return>", self.on_submit)
        self.entry.bind("<KeyRelease>", self.on_key_release)
        self.entry.bind("<Down>", self.focus_results)
        self.root.bind("<Escape>", self.on_escape)

        self.placeholder = "Open apps · find files · summarise a PDF · settings wifi · or ask anything"
        self._show_placeholder()
        self.entry.bind("<FocusIn>", lambda e: self._clear_placeholder())

        self.status = tk.Label(outer, text="", fg=ACCENT, bg=BG, font=(FONT, 10), anchor="w", justify="left",
                               wraplength=int((WIDTH - 40) * self.scale))
        self.status.pack(fill="x")

        # Setup panel (shown only when Ollama or the model is missing)
        self.setup = tk.Frame(outer, bg=BG)
        self.setup_btn = tk.Button(self.setup, text="", font=(FONT, 10, "bold"), bg=ACCENT, fg=BG, relief="flat",
                                   activebackground="#7dd3fc", padx=12, pady=6, command=self.on_setup_click)
        self.setup_btn.pack(side="left")
        self.progress = ttk.Progressbar(self.setup, length=380, mode="determinate", maximum=1.0)

        self.listbox = tk.Listbox(outer, font=(FONT, 11), bg=PANEL, fg=FG, selectbackground=ACCENT,
                                  selectforeground=BG, relief="flat", highlightthickness=0, height=0,
                                  activestyle="none")
        self.listbox.bind("<Return>", self.on_result_open)
        self.listbox.bind("<Double-Button-1>", self.on_result_open)
        self.listbox.bind("<Up>", self.on_list_up)

        self.answer = tk.Text(outer, font=(FONT, 11), bg=PANEL, fg=FG, relief="flat", wrap="word",
                              height=12, padx=12, pady=10, highlightthickness=0)
        self.answer.configure(state="disabled")

        foot = tk.Frame(outer, bg=BG)
        foot.pack(fill="x", side="bottom", pady=(8, 0))
        self.foot_hint = tk.Label(foot, text="Esc to hide", fg=MUTED, bg=BG, font=(FONT, 8))
        self.foot_hint.pack(side="left")
        link = tk.Label(foot, text="technoforesight.in", fg=MUTED, bg=BG, font=(FONT, 8, "underline"), cursor="hand2")
        link.pack(side="right")
        link.bind("<Button-1>", lambda e: webbrowser.open(WEBSITE))

    def _place(self):
        self.root.update_idletasks()
        self._apply_geometry()

    def _resize(self):
        self.root.geometry("")  # let tk compute natural height
        self.root.update_idletasks()
        self._apply_geometry()

    def _apply_geometry(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(self.width, sw - 40)
        self.root.geometry(f"{w}x{self.root.winfo_reqheight()}+{max(0, (sw - w) // 2)}+{int(sh * 0.15)}")

    # ---------- placeholder ----------
    def _show_placeholder(self):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.configure(fg=MUTED)
            self._has_placeholder = True

    def _clear_placeholder(self):
        if getattr(self, "_has_placeholder", False):
            self.entry.delete(0, "end")
            self.entry.configure(fg=FG)
            self._has_placeholder = False

    def text(self) -> str:
        return "" if getattr(self, "_has_placeholder", False) else self.entry.get().strip()

    # ---------- show / hide ----------
    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.entry.focus_set()
        self._clear_placeholder()
        self.entry.select_range(0, "end")

    def hide(self):
        self.root.withdraw()

    # ---------- background → UI ----------
    def _pump(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "show":
                    self.show()
                elif kind == "status":
                    self.set_status(ev[1], ev[2] if len(ev) > 2 else ACCENT)
                elif kind == "apps":
                    self.apps = ev[1]
                elif kind == "answer_start" and ev[1] == self._generation:
                    self.busy_since = None
                    self.streaming = True
                    self._answer_start()
                elif kind == "done" and ev[1] == self._generation:
                    self.busy_since = None
                    self.streaming = False
                    self.foot_hint.configure(text="Esc to hide")
                elif kind == "answer" and ev[2] == self._generation:
                    self._answer_append(ev[1])
                elif kind == "files":
                    self.show_results([(str(p).replace(str(Path.home()), "~"), p) for p in ev[1]])
                elif kind == "hide":
                    self.hide()
                elif kind == "quit":
                    self.root.destroy()
                    return
                elif kind == "setup":
                    self._show_setup(ev[1])
                elif kind == "setup_hide":
                    self.setup.pack_forget()
                    self._resize()
                elif kind == "setup_enable":
                    self.setup_btn.configure(state="normal")
                elif kind == "progress":
                    self.progress["value"] = ev[1] if ev[1] >= 0 else 0
        except queue.Empty:
            pass
        self.root.after(60, self._pump)

    def _load_apps(self):
        self.events.put(("apps", load_apps()))

    def _check_engine(self):
        problem = self.llm.check()
        if problem is None:
            self._warm_up()
            return
        if "Can't reach" in problem:
            self.events.put(("setup", "install"))
        else:
            self.events.put(("setup", "pull"))
        self.events.put(("status", "⚠ " + problem, ERR))

    def _warm_up(self):
        """Loads the model and measures speed, so the first real question isn't a mystery wait,
        and slow laptops get offered the faster model."""
        self.events.put(("status", "Warming up the AI model (the first time can take up to a minute)…"))
        try:
            tps = self.llm.speed_test()
        except OllamaError as e:
            msg = str(e)
            if "memory" in msg.lower():
                msg += "  Close some apps (e.g. browser tabs) or switch to the smaller model."
                self.events.put(("setup", "fast"))
            self.events.put(("status", "⚠ AI model couldn't start: " + msg, ERR))
            return
        speed = f" ({tps:.1f} words/sec on this PC)" if tps else ""
        if tps and tps < SLOW_TOKENS_PER_SEC and self.llm.model == DEFAULT_MODEL and not user_set("model"):
            self.events.put(("setup", "fast"))
            self.events.put(("status", f"✔ Ready, but slow on this PC{speed}. A smaller model answers about twice as fast."))
        else:
            self.events.put(("status", f"✔ AI ready{speed}. Ask me anything!"))

    # ---------- setup panel ----------
    def _show_setup(self, mode: str):
        self.setup_mode = mode
        label = {"install": "Install Ollama (free AI engine)", "pull": f"Download AI model ({self.cfg.model}, ~2.5 GB)",
                 "recheck": "I've installed it, check again",
                 "fast": f"Switch to faster model ({FAST_MODEL}, ~1.4 GB)"}[mode]
        self.setup_btn.configure(text=label)
        self.setup.pack(fill="x", pady=(6, 4), before=self.status)
        self._resize()

    def on_setup_click(self):
        if self.setup_mode == "install":
            webbrowser.open("https://ollama.com/download/windows")
            self._show_setup("recheck")
        elif self.setup_mode == "recheck":
            self.setup.pack_forget()
            self.set_status("Checking…")
            threading.Thread(target=self._check_engine_then_ok, daemon=True).start()
        elif self.setup_mode in ("pull", "fast"):
            if self.setup_mode == "fast":
                self.llm.model = self.cfg.model = FAST_MODEL
                save_setting("model", FAST_MODEL)
            self.setup_btn.configure(state="disabled")
            self.progress.pack(side="left", padx=12)
            threading.Thread(target=self._pull_model, daemon=True).start()

    def _check_engine_then_ok(self):
        if self.llm.check() is None:
            self._warm_up()
        else:
            self._check_engine()

    def _pull_model(self):
        try:
            for status, frac in self.llm.pull_stream():
                pct = f" {frac * 100:.0f}%" if frac >= 0 else ""
                self.events.put(("status", f"Downloading model: {status}{pct}"))
                self.events.put(("progress", frac))
            self.events.put(("setup_hide",))
            self._warm_up()
        except OllamaError as e:
            self.events.put(("status", f"⚠ {e}", ERR))
            self.events.put(("setup_enable",))

    # ---------- typing ----------
    def on_key_release(self, event):
        if event.keysym in ("Return", "Down", "Up", "Escape"):
            return
        q = self.text()
        self._hide_answer()
        matches = [a for a in self.apps if q and a.matches(q)][:7]
        self.show_results([(a.name, a) for a in matches])

    def on_submit(self, _event=None):
        text = self.text()
        if not text:
            return
        action = route(text, self.apps)
        self._generation += 1
        gen = self._generation
        self.busy_since = time.monotonic()
        self._tick(gen)

        def worker():
            try:
                res = actions.run(action, self.cfg, self.llm)
                self.events.put(("status", res.message or ""))
                if res.files:
                    self.events.put(("files", res.files))
                if res.quit_app:
                    self.events.put(("quit",))
                    return
                if res.hide_window:
                    self.events.put(("hide",))
                    return
                if res.stream is not None:
                    first = True
                    try:
                        for piece in res.stream:
                            if gen != self._generation:
                                return  # cancelled: leaving the loop closes the request
                            if first:
                                self.events.put(("answer_start", gen))
                                first = False
                            self.events.put(("answer", piece, gen))
                    finally:
                        res.stream.close()
                        self.events.put(("done", gen))
                    if first and gen == self._generation:
                        self.events.put(("status", "The AI returned an empty answer. Try rephrasing.", ERR))
            except (OllamaError, OSError, RuntimeError) as e:
                if gen == self._generation:
                    self.events.put(("done", gen))
                    self.events.put(("status", f"⚠ {e}", ERR))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- results list ----------
    def show_results(self, items):
        self.results = items
        self.listbox.delete(0, "end")
        for label, target in items:
            prefix = "📄  " if isinstance(target, Path) else "▸  "
            self.listbox.insert("end", prefix + label)
        if items:
            self.listbox.configure(height=min(len(items), 8))
            self.listbox.pack(fill="x", pady=(6, 0))
        else:
            self.listbox.pack_forget()
        self._resize()

    def focus_results(self, _event=None):
        if self.results:
            self.listbox.focus_set()
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(0)
            self.listbox.activate(0)
        return "break"

    def on_list_up(self, _event=None):
        if self.listbox.curselection() == (0,):
            self.entry.focus_set()
            return "break"
        return None

    def on_result_open(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        _, target = self.results[sel[0]]
        try:
            if isinstance(target, App):
                launch(target)
            elif isinstance(target, Path):
                files.open_path(target)
            self.hide()
        except OSError as e:
            self.set_status(f"⚠ {e}", ERR)

    # ---------- answer area ----------
    def _answer_start(self):
        self.listbox.pack_forget()
        self.answer.configure(state="normal")
        self.answer.delete("1.0", "end")
        self.answer.configure(state="disabled")
        self.answer.pack(fill="both", expand=True, pady=(6, 0))
        if self.status.cget("text").startswith("Thinking…"):
            self.set_status("")
        self._resize()

    def _answer_append(self, piece: str):
        self.answer.configure(state="normal")
        self.answer.insert("end", piece)
        self.answer.see("end")
        self.answer.configure(state="disabled")

    def _hide_answer(self):
        if self.answer.winfo_ismapped():
            self.answer.pack_forget()
            self._resize()

    def _tick(self, gen: int):
        """Shows 'Thinking… 12s' until the first word arrives, so the app never looks frozen."""
        if gen != self._generation or self.busy_since is None:
            return
        secs = int(time.monotonic() - self.busy_since)
        hint = "  (older laptops can take 20-60s for the first answer)" if secs >= 8 else ""
        self.set_status(f"Thinking… {secs}s{hint}")
        self.foot_hint.configure(text="Esc to stop")
        self.root.after(1000, lambda: self._tick(gen))

    def on_escape(self, _event=None):
        if self.busy_since is not None or self.streaming:
            self._generation += 1  # worker sees the change and closes the request
            self.busy_since = None
            self.streaming = False
            self.foot_hint.configure(text="Esc to hide")
            self.set_status("Stopped.")
        else:
            self.hide()

    def set_status(self, text: str, color: str = ACCENT):
        self.status.configure(text=text, fg=color)

    def run(self):
        self.root.mainloop()


def _enable_dpi_awareness() -> None:
    """Without this, Windows bitmap-scales the window on high-DPI screens: blurry and off-centre."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system-DPI aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def run_gui(cfg: Config) -> int:
    _enable_dpi_awareness()
    instance = SingleInstance()
    if not instance.is_primary:
        instance.signal_primary()  # already running: just pop its window up
        return 0
    CommandBar(cfg, instance).run()
    return 0
