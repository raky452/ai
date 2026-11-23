"""Command-bar window: borderless, rounded "glass" panel with a neon gradient edge.

Built on tkinter (ships with Python, no extra dependencies). All background work posts to
`self.events`; only the UI thread touches widgets.
"""

from __future__ import annotations

import ctypes
import math
import queue
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk

from . import APP_NAME, WEBSITE, actions, files
from .apps import App, launch, load_apps
from .config import (DEFAULT_MODEL, FAST_MODEL, INDIC_MODEL_SIZE, SLOW_TOKENS_PER_SEC, Config, save_setting,
                     user_set)
from .llm import LANGUAGE_BY_NAME, LANGUAGES, RESET, OllamaClient, OllamaError
from .router import UseModel, route
if sys.platform == "win32":
    from .winsys import SingleInstance, start_hotkey_listener
else:
    from .linuxsys import SingleInstance, start_hotkey_listener

WINDOWS = sys.platform == "win32"

# ---------- design tokens ----------
# Windows: transparent colour key outside the rounded panel. Linux window managers don't support
# colour-key transparency, so there the area around the panel is a near-black frame.
KEY = "#010203" if sys.platform == "win32" else "#05070d"
BG = "#0a0f1e"             # panel
SURFACE = "#111a2e"        # input, cards
SURFACE_HI = "#16223b"     # hover
LINE = "#1e2c48"
FG = "#e8eefc"
MUTED = "#7f8db0"
DIM = "#4b587a"
CYAN, VIOLET, PINK = "#22d3ee", "#8b5cf6", "#ec4899"
GREEN, AMBER, ERR = "#34d399", "#fbbf24", "#fca5a5"
if sys.platform == "win32":
    FONT, MONO, INDIC = "Segoe UI", "Consolas", "Nirmala UI"
else:  # fontconfig falls back per character, so Noto covers every Indian script it has installed
    FONT, MONO, INDIC = "Noto Sans", "DejaVu Sans Mono", "Noto Sans"
WIDTH, RADIUS, PAD = 780, 22, 14

CHIPS = [
    ("✦  Ask AI", ""),
    ("▶  Open app", "open "),
    ("⌕  Find files", "find files "),
    ("▤  Summarise", "summarise "),
    ("⚙  Settings", "settings "),
    ("🔗  Connectors", "connectors"),
    ("⏻  Lock PC", "lock pc"),
]


def _resource(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets" / name


def _mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colours."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def _gradient(t: float) -> str:
    """Cyan → violet → pink."""
    return _mix(CYAN, VIOLET, t * 2) if t < 0.5 else _mix(VIOLET, PINK, (t - 0.5) * 2)


def _round_rect(cv: tk.Canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return cv.create_polygon(pts, smooth=True, **kw)


class Chip(tk.Label):
    """Flat pill button with hover state."""

    def __init__(self, master, text, command, *, fg=MUTED, bg=SURFACE, hover=SURFACE_HI, font=(FONT, 9), padx=10, pady=4):
        super().__init__(master, text=text, fg=fg, bg=bg, font=font, padx=padx, pady=pady, cursor="hand2")
        self._bg, self._hover, self._fg = bg, hover, fg
        self.bind("<Enter>", lambda e: self.configure(bg=self._hover, fg=FG))
        self.bind("<Leave>", lambda e: self.configure(bg=self._bg, fg=self._fg))
        self.bind("<Button-1>", lambda e: command())


class Pulse(tk.Canvas):
    """Animated 'thinking' indicator: three pulsing dots and a scanning gradient bar."""

    def __init__(self, master, scale: float):
        self.s = scale
        super().__init__(master, width=int(170 * scale), height=int(14 * scale), bg=BG, highlightthickness=0)
        self.running = False
        self.t0 = 0.0

    def start(self):
        if not self.running:
            self.running, self.t0 = True, time.monotonic()
            self._frame()

    def stop(self):
        self.running = False
        self.delete("all")

    def _frame(self):
        if not self.running:
            return
        self.delete("all")
        t = time.monotonic() - self.t0
        s = self.s
        for i in range(3):
            k = (math.sin(t * 6 - i * 0.9) + 1) / 2
            r = (2.5 + 2 * k) * s
            cx, cy = (8 + i * 13) * s, 7 * s
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=_mix(DIM, _gradient(i / 2), k), outline="")
        x0, w = 50 * s, 115 * s
        self.create_rectangle(x0, 6 * s, x0 + w, 8 * s, fill=LINE, outline="")
        head = (t * 0.8) % 1.0
        seg = 30 * s
        steps = 12
        for j in range(steps):
            frac = head - j / steps * 0.25
            if 0 <= frac <= 1:
                x = x0 + frac * (w - seg / steps)
                self.create_rectangle(x, 5.5 * s, x + seg / steps + 1, 8.5 * s,
                                      fill=_mix(BG, _gradient(frac), 1 - j / steps), outline="")
        self.after(33, self._frame)


class CommandBar:
    def __init__(self, cfg: Config, instance: SingleInstance):
        self.cfg = cfg
        self.llm = OllamaClient(cfg.ollama_url, cfg.model, cfg.assistant_name, language=cfg.response_language)
        self.apps: list[App] = []
        self.results: list[tuple[str, object]] = []
        self.events: queue.Queue = queue.Queue()
        self._generation = 0
        self.busy_since: float | None = None
        self.streaming = False
        self.speed: float | None = None
        self.answer_text = ""

        # className sets WM_CLASS on Linux so the dock groups the window under the app's icon
        self.root = tk.Tk(className="TechnoForesightAssistant")
        self.root.title(APP_NAME)
        self.root.configure(bg=KEY)
        if WINDOWS:
            self.root.overrideredirect(True)      # borderless: we draw our own rounded panel
            self.root.attributes("-topmost", True)
            try:
                self.root.attributes("-transparentcolor", KEY)
            except tk.TclError:
                pass
            try:
                self.root.iconbitmap(default=str(_resource("icon.ico")))
            except tk.TclError:
                pass
        else:
            # Keep the window manager's frame on Linux: borderless (override-redirect) windows can't
            # receive keyboard focus reliably on X11/XWayland.
            self.root.resizable(False, False)
            try:
                self.root.iconphoto(True, tk.PhotoImage(file=str(_resource("icon.png"))))
            except tk.TclError:
                pass
        self.scale = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        self.width = int(WIDTH * self.scale)
        self._style()
        self._build()

        threading.Thread(target=self._load_apps, daemon=True).start()
        threading.Thread(target=self._check_engine, daemon=True).start()
        self.hotkey_ok = start_hotkey_listener(cfg.hotkey, lambda: self.events.put(("show",))) is not None
        instance.listen(lambda: self.events.put(("show",)))
        self.root.after(60, self._pump)
        if WINDOWS:
            self.set_status(f"Press {self._hk()} anywhere to summon me." if self.hotkey_ok
                            else f"{self._hk()} is used by another app. Change 'hotkey' in config.toml.", MUTED)
        else:
            from .linuxsys import SHORTCUT_LABEL, ensure_gnome_shortcut
            self.hotkey_ok = ensure_gnome_shortcut()
            self.set_status(f"Press {SHORTCUT_LABEL} anywhere to summon me." if self.hotkey_ok
                            else "Tip: add a keyboard shortcut for 'technoforesight-assistant' in your desktop's "
                                 "keyboard settings.", MUTED)
        self.show()

    # ================= construction =================
    def _px(self, v: float) -> int:
        return int(v * self.scale)

    def _hk(self) -> str:
        return "+".join(p.strip().title() for p in self.cfg.hotkey.split("+"))

    def _style(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Neon.Horizontal.TProgressbar", troughcolor=SURFACE, background=CYAN,
                     bordercolor=SURFACE, lightcolor=CYAN, darkcolor=VIOLET, thickness=self._px(6))

    def _build(self):
        self.canvas = tk.Canvas(self.root, bg=KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=BG)
        self.body_win = self.canvas.create_window(self._px(PAD + 4), self._px(PAD + 8), window=self.body,
                                                  anchor="nw", width=self.width - 2 * self._px(PAD + 4))
        self.body.bind("<Configure>", lambda e: self._layout())

        self._build_header()
        self._build_input()
        self._build_chips()
        self._build_status()
        self._build_setup()
        self._build_results()
        self._build_answer()
        self._build_footer()

        self.root.bind("<Escape>", self.on_escape)

    def _build_header(self):
        head = tk.Frame(self.body, bg=BG)
        head.pack(fill="x")
        logo = tk.Canvas(head, width=self._px(26), height=self._px(26), bg=BG, highlightthickness=0)
        logo.pack(side="left")
        s = self._px(26)
        for i in range(12, 0, -1):  # glowing gradient diamond
            k = i / 12
            r = s / 2 * k
            logo.create_polygon(s / 2, s / 2 - r, s / 2 + r, s / 2, s / 2, s / 2 + r, s / 2 - r, s / 2,
                                fill=_gradient(1 - k), outline="")
        logo.create_text(s / 2, s / 2, text="T", fill=BG, font=(FONT, 9, "bold"))
        tk.Label(head, text="TechnoForesight", fg=FG, bg=BG, font=(FONT, 12, "bold")).pack(side="left", padx=(8, 4))
        badge = tk.Label(head, text=" AI ", fg=BG, bg=CYAN, font=(MONO, 8, "bold"))
        badge.pack(side="left", pady=2)

        close = Chip(head, "✕", self.hide, fg=MUTED, bg=BG, hover=SURFACE, font=(FONT, 10), padx=8, pady=2)
        close.pack(side="right")
        self.lang_chip = Chip(head, self._lang_label(), self.open_language_menu, fg=FG, bg=SURFACE,
                              hover=SURFACE_HI, font=(FONT, 9), padx=10, pady=3)
        self.lang_chip.pack(side="right", padx=(6, 6))
        status = tk.Frame(head, bg=SURFACE, padx=self._px(8), pady=self._px(2))
        status.pack(side="right")
        self.dot = tk.Canvas(status, width=self._px(10), height=self._px(10), bg=SURFACE, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 6))
        self.engine_label = tk.Label(status, text="CONNECTING", fg=MUTED, bg=SURFACE, font=(MONO, 8, "bold"))
        self.engine_label.pack(side="left")
        self._dot_color = AMBER
        self._blink()

        for w in (head, logo, badge, status, self.engine_label):  # drag the window by its header
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)
        tk.Frame(self.body, bg=BG, height=self._px(10)).pack(fill="x")

    def _build_input(self):
        self.input_frame = tk.Frame(self.body, bg=SURFACE, highlightthickness=self._px(2),
                                    highlightbackground=LINE, highlightcolor=CYAN)
        self.input_frame.pack(fill="x")
        tk.Label(self.input_frame, text="❯", fg=CYAN, bg=SURFACE, font=(MONO, 18, "bold")).pack(side="left", padx=(12, 4))
        self.entry = tk.Entry(self.input_frame, font=(FONT, 16), bg=SURFACE, fg=FG, insertbackground=CYAN,
                              insertwidth=self._px(2), relief="flat", highlightthickness=0, bd=0)
        self.entry.pack(side="left", fill="x", expand=True, ipady=self._px(11))
        tk.Label(self.input_frame, text="⏎ enter", fg=DIM, bg=SURFACE, font=(MONO, 9)).pack(side="right", padx=12)
        self.entry.bind("<Return>", self.on_submit)
        self.entry.bind("<KeyRelease>", self.on_key_release)
        self.entry.bind("<Down>", self.focus_results)
        self.entry.bind("<FocusIn>", lambda e: (self._clear_placeholder(), self.input_frame.configure(highlightbackground=CYAN)))
        self.entry.bind("<FocusOut>", lambda e: self.input_frame.configure(highlightbackground=LINE))
        self.placeholder = "Ask anything, or try: open chrome · find files invoice · yt telugu songs"
        self._show_placeholder()

    def _build_chips(self):
        row = tk.Frame(self.body, bg=BG)
        row.pack(fill="x", pady=(self._px(10), 0))
        for label, template in CHIPS:
            Chip(row, label, lambda t=template: self._use_template(t)).pack(side="left", padx=(0, 6))

    def _build_status(self):
        row = tk.Frame(self.body, bg=BG)
        row.pack(fill="x", pady=(self._px(10), 0))
        self.pulse = Pulse(row, self.scale)
        self.pulse.pack(side="left")
        self.pulse.pack_forget()
        self.status = tk.Label(row, text="", fg=CYAN, bg=BG, font=(FONT, 10), anchor="w", justify="left",
                               wraplength=self.width - self._px(80))
        self.status.pack(side="left", fill="x", expand=True)
        self.status_row = row

    def _build_setup(self):
        self.setup = tk.Frame(self.body, bg=BG)
        self.setup_btn = tk.Label(self.setup, text="", fg=BG, bg=CYAN, font=(FONT, 10, "bold"),
                                  padx=14, pady=7, cursor="hand2")
        self.setup_btn.pack(side="left")
        self.setup_btn.bind("<Button-1>", lambda e: self.on_setup_click())
        self.setup_btn.bind("<Enter>", lambda e: self.setup_btn.configure(bg="#67e8f9"))
        self.setup_btn.bind("<Leave>", lambda e: self.setup_btn.configure(bg=CYAN))
        self.progress = ttk.Progressbar(self.setup, length=self._px(360), mode="determinate", maximum=1.0,
                                        style="Neon.Horizontal.TProgressbar")
        self.setup_mode = ""
        self._setup_busy = False

    def _build_results(self):
        self.listbox = tk.Listbox(self.body, font=(FONT, 11), bg=SURFACE, fg=FG, selectbackground=_mix(SURFACE, VIOLET, 0.7),
                                  selectforeground="#ffffff", relief="flat", highlightthickness=0, height=0,
                                  activestyle="none", bd=0)
        self.listbox.bind("<Return>", self.on_result_open)
        self.listbox.bind("<Double-Button-1>", self.on_result_open)
        self.listbox.bind("<Up>", self.on_list_up)

    def _build_answer(self):
        self.card = tk.Frame(self.body, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        bar = tk.Frame(self.card, bg=SURFACE)
        bar.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(bar, text="◆ AI RESPONSE", fg=CYAN, bg=SURFACE, font=(MONO, 8, "bold")).pack(side="left")
        self.answer_lang = tk.Label(bar, text="", fg=MUTED, bg=SURFACE, font=(MONO, 8))
        self.answer_lang.pack(side="left", padx=8)
        Chip(bar, "⧉ Copy", self.copy_answer, bg=SURFACE, hover=SURFACE_HI, font=(FONT, 8), padx=8, pady=1).pack(side="right")
        self.answer = tk.Text(self.card, font=self._answer_font(), bg=SURFACE, fg=FG, relief="flat", wrap="word",
                              height=11, padx=14, pady=10, highlightthickness=0, bd=0, spacing2=3, spacing3=4,
                              insertbackground=CYAN)
        self.answer.pack(fill="both", expand=True)
        self.answer.tag_configure("cursor", foreground=CYAN)
        self.answer.configure(state="disabled")

    def _build_footer(self):
        foot = tk.Frame(self.body, bg=BG)
        foot.pack(fill="x", side="bottom", pady=(self._px(12), 0))
        self.foot_hint = tk.Label(foot, text=self._hint_text(), fg=DIM, bg=BG, font=(MONO, 8))
        self.foot_hint.pack(side="left")
        link = tk.Label(foot, text="technoforesight.in", fg=DIM, bg=BG, font=(MONO, 8, "underline"), cursor="hand2")
        link.pack(side="right")
        link.bind("<Button-1>", lambda e: webbrowser.open(WEBSITE))
        self.model_label = tk.Label(foot, text=f"⚡ {self.llm.model}", fg=MUTED, bg=BG, font=(MONO, 8))
        self.model_label.pack(side="right", padx=12)

    # ================= painting the rounded shell =================
    def _layout(self):
        if getattr(self, "_in_layout", False):
            return
        self._in_layout = True
        try:
            self._paint()
        finally:
            self._in_layout = False

    def _paint(self):
        self.root.update_idletasks()
        bh = self.body.winfo_reqheight()
        p = self._px(PAD)
        w, h = self.width, bh + 2 * p + self._px(20)
        self.canvas.configure(width=w, height=h)
        cv = self.canvas
        cv.delete("shell")
        r = self._px(RADIUS)
        # soft outer glow
        for i, k in enumerate((0.10, 0.18, 0.30)):
            off = self._px(6 - i * 2)
            _round_rect(cv, p - off, p - off, w - p + off, h - p + off, r + off,
                        fill=_mix(KEY, VIOLET, k * 0.5) if i < 2 else _mix(BG, CYAN, k * 0.5), outline="", tags="shell")
        _round_rect(cv, p, p, w - p, h - p, r, fill=BG, outline=_mix(LINE, CYAN, 0.35), width=1, tags="shell")
        # neon gradient edge along the top
        x1, x2, y = p + r, w - p - r, p + 1
        n = 90
        for i in range(n):
            xa = x1 + (x2 - x1) * i / n
            cv.create_line(xa, y, xa + (x2 - x1) / n + 1, y, fill=_gradient(i / (n - 1)), width=self._px(2), tags="shell")
        cv.tag_lower("shell")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if not getattr(self, "_moved", False):
            self._x, self._y = max(0, (sw - w) // 2), int(sh * 0.14)
        self.root.geometry(f"{w}x{h}+{self._x}+{self._y}")

    def _drag_start(self, e):
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _drag_move(self, e):
        dx, dy = self._drag
        self._x, self._y, self._moved = e.x_root - dx, e.y_root - dy, True
        self.root.geometry(f"+{self._x}+{self._y}")

    def _blink(self):
        self.dot.delete("all")
        k = (math.sin(time.monotonic() * 3) + 1) / 2
        s = self._px(10)
        self.dot.create_oval(0, 0, s, s, fill=_mix(SURFACE, self._dot_color, 0.35 + 0.25 * k), outline="")
        self.dot.create_oval(s * 0.25, s * 0.25, s * 0.75, s * 0.75, fill=self._dot_color, outline="")
        self.root.after(80, self._blink)

    def _set_engine(self, text: str, color: str):
        self.engine_label.configure(text=text, fg=color)
        self._dot_color = color

    # ================= language =================
    def _lang_label(self) -> str:
        name, native, _ = LANGUAGE_BY_NAME.get(self.llm.language, LANGUAGES[0])
        return f"🌐  {native if name == 'English' else native + ' · ' + name}  ▾"

    def _answer_font(self):
        return (FONT if self.llm.language in ("English",) else INDIC, 11)

    def open_language_menu(self):
        menu = tk.Menu(self.root, tearoff=0, bg=SURFACE, fg=FG, activebackground=VIOLET, activeforeground="#ffffff",
                       font=(INDIC, 10), bd=0, relief="flat")
        menu.add_command(label="Response language", state="disabled")
        menu.add_separator()
        for name, native, _ in LANGUAGES:
            mark = "●  " if name == self.llm.language else "    "
            menu.add_command(label=f"{mark}{native}   ({name})" if name != "English" else f"{mark}English",
                             command=lambda n=name: self.set_language(n))
        x, y = self.lang_chip.winfo_rootx(), self.lang_chip.winfo_rooty() + self.lang_chip.winfo_height()
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    # ---------- model choice ----------
    def _has_model(self, name: str) -> bool:
        installed = getattr(self, "_installed", set())
        return (name if ":" in name else name + ":latest") in installed

    def _refresh_installed(self):
        try:
            self._installed = set(self.llm.installed_models())
        except OllamaError:
            pass

    def _model_for_language(self) -> str:
        """Indian languages use the Indian-language model once it's downloaded; until then the
        main model keeps answering so the app never stops working."""
        if self.llm.language != "English" and self.cfg.indic_model and self._has_model(self.cfg.indic_model):
            return self.cfg.indic_model
        return self.cfg.model

    def _offer_indic_model(self):
        """Background: if an Indian language is selected and its better model is missing, offer it."""
        name = self.cfg.indic_model
        if self.llm.language == "English" or not name or name == self.cfg.model:
            return
        self._refresh_installed()
        if self._has_model(name):
            self._apply_model_for_language_bg()
            self.events.put(("speed", 0))
            return
        self._pull_target = name
        native = LANGUAGE_BY_NAME[self.llm.language][1]
        self.events.put(("setup", "indic"))
        self.events.put(("status", f"Answers work now. For much better {native}, download {name} once "
                                   f"(~{INDIC_MODEL_SIZE}, free, works offline).", AMBER))

    def _apply_model_for_language_bg(self):
        """Thread-safe variant: only changes the client model (label updates on next submit)."""
        self.llm.model = self._model_for_language()

    def _apply_model_for_language(self):
        want = self._model_for_language()
        if self.llm.model != want:
            self.llm.model = want
            self.model_label.configure(text=f"⚡ {want}")

    def use_model(self, action: "UseModel"):
        """'use model X' (all languages) or 'use model X for indian languages'."""
        if action.scope == "indic":
            self.cfg.indic_model = action.name
            save_setting("indic_model", action.name)
            where = "for Indian-language answers"
        else:
            self.cfg.model = action.name
            save_setting("model", action.name)
            where = "for all answers"
        self.entry.delete(0, "end")
        self._apply_model_for_language()
        self.set_status(f"Switching to {action.name} {where}…", CYAN)
        threading.Thread(target=self._ensure_model, args=(action.name,), daemon=True).start()

    def _ensure_model(self, name: str):
        """Download the model if needed (with progress), then warm it up."""
        probe = OllamaClient(self.cfg.ollama_url, name, self.cfg.assistant_name)
        problem = probe.check()
        if problem and "isn't downloaded" in problem:
            self._pull_target = name
            self.events.put(("setup", "pull"))
            self.events.put(("status", f"⬇ {name} isn't downloaded yet. Click the button to download it.", AMBER))
            return
        if problem:
            self.events.put(("status", "⚠ " + problem, ERR))
            return
        self._refresh_installed()
        self._apply_model_for_language_bg()
        saved = self.llm.model
        self.llm.model = name
        self._warm_up()
        self.llm.model = saved
        self.events.put(("speed", 0))
        self.events.put(("status", f"✔ Now using {name}.", GREEN))

    def set_language(self, name: str):
        self.llm.language = self.cfg.response_language = name
        try:
            save_setting("response_language", name)
        except OSError:
            pass
        self.lang_chip.configure(text=self._lang_label())
        self.answer.configure(font=self._answer_font())
        self._apply_model_for_language()
        native = LANGUAGE_BY_NAME[name][1]
        self.set_status(f"✔ I'll answer in {native} from now on.", GREEN)
        self.entry.focus_set()
        if name != "English":
            threading.Thread(target=self._offer_indic_model, daemon=True).start()
        elif self.setup.winfo_ismapped() and self.setup_mode == "indic":
            self.setup.pack_forget()
            self._layout()

    # ================= placeholder =================
    def _show_placeholder(self):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.configure(fg=DIM)
            self._has_placeholder = True

    def _clear_placeholder(self):
        if getattr(self, "_has_placeholder", False):
            self.entry.delete(0, "end")
            self.entry.configure(fg=FG)
            self._has_placeholder = False

    def text(self) -> str:
        return "" if getattr(self, "_has_placeholder", False) else self.entry.get().strip()

    def _use_template(self, template: str):
        self._clear_placeholder()
        self.entry.delete(0, "end")
        self.entry.insert(0, template)
        self.entry.focus_set()
        self.entry.icursor("end")
        if template.strip() == "lock pc":
            self.set_status("Press Enter to lock your PC.", MUTED)

    # ================= show / hide =================
    def show(self):
        self.root.deiconify()
        self._layout()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.entry.focus_set()
        self._clear_placeholder()
        self.entry.select_range(0, "end")

    def hide(self):
        self.root.withdraw()

    # ================= background → UI =================
    def _pump(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "show":
                    self.show()
                elif kind == "status":
                    self.set_status(ev[1], ev[2] if len(ev) > 2 else CYAN)
                elif kind == "engine":
                    self._set_engine(ev[1], ev[2])
                elif kind == "speed":
                    self.speed = ev[1]
                    self.model_label.configure(text=f"⚡ {self.llm.model}" + (f" · {ev[1]:.1f} tok/s" if ev[1] else ""))
                elif kind == "apps":
                    self.apps = ev[1]
                elif kind == "answer_start" and ev[1] == self._generation:
                    self.busy_since = None
                    self.streaming = True
                    self.pulse.stop()
                    self.pulse.pack_forget()
                    self._answer_start()
                elif kind == "answer" and ev[2] == self._generation:
                    if ev[1] == RESET:
                        self._answer_clear()
                    else:
                        self._answer_append(ev[1])
                elif kind == "done" and ev[1] == self._generation:
                    self._answer_done()
                    self.busy_since = None
                    self.streaming = False
                    self.pulse.stop()
                    self.pulse.pack_forget()
                    self.foot_hint.configure(text=self._hint_text())
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
                    self._setup_busy = False
                    self.progress.pack_forget()
                    self.setup.pack_forget()
                    self._layout()
                elif kind == "setup_enable":
                    self._setup_busy = False
                    self.setup_btn.configure(bg=CYAN)
                elif kind == "progress":
                    self.progress["value"] = ev[1] if ev[1] >= 0 else 0
        except queue.Empty:
            pass
        self.root.after(50, self._pump)

    def _load_apps(self):
        self.events.put(("apps", load_apps()))

    def _check_engine(self):
        self.llm.model = self.cfg.model
        problem = self.llm.check()
        if problem is None:
            self._refresh_installed()
            self._apply_model_for_language_bg()
            self._warm_up()
            self._offer_indic_model()
            return
        if "Can't reach" in problem:
            self.events.put(("engine", "ENGINE OFF", ERR))
            self.events.put(("setup", "install"))
        else:
            self.events.put(("engine", "NO MODEL", AMBER))
            self.events.put(("setup", "pull"))
        self.events.put(("status", "⚠ " + problem, ERR))

    def _warm_up(self):
        self.events.put(("engine", "WARMING UP", AMBER))
        self.events.put(("status", "Warming up the AI model (the first time can take up to a minute)…", MUTED))
        try:
            tps = self.llm.speed_test()
        except OllamaError as e:
            msg = str(e)
            if "memory" in msg.lower():
                msg += "  Close some apps (e.g. browser tabs) or switch to the smaller model."
                self.events.put(("setup", "fast"))
            self.events.put(("engine", "MODEL ERROR", ERR))
            self.events.put(("status", "⚠ AI model couldn't start: " + msg, ERR))
            return
        self.events.put(("engine", "LOCAL · OFFLINE", GREEN))
        self.events.put(("speed", tps))
        if tps and tps < SLOW_TOKENS_PER_SEC and self.llm.model == DEFAULT_MODEL and not user_set("model"):
            self.events.put(("setup", "fast"))
            self.events.put(("status", f"✔ Ready, but slow on this PC ({tps:.1f} tok/s). A smaller model answers about twice as fast.", AMBER))
        else:
            self.events.put(("status", "✔ AI ready. Ask me anything, it all stays on this PC.", GREEN))

    # ================= setup panel =================
    def _show_setup(self, mode: str):
        self.setup_mode = mode
        label = {"install": "⬇  Install Ollama (free AI engine)",
                 "pull": f"⬇  Download AI model ({self._pending_model()})",
                 "recheck": "↻  I've installed it, check again",
                 "fast": f"⚡  Switch to faster model ({FAST_MODEL}, ~1.4 GB)",
                 "indic": f"⬇  Better Indian languages: get {self._pending_model()} (~{INDIC_MODEL_SIZE})"}[mode]
        self.setup_btn.configure(text=label)
        self.setup.pack(fill="x", pady=(self._px(8), 0), after=self.status_row)
        self._layout()

    def _pending_model(self) -> str:
        return getattr(self, "_pull_target", None) or self.cfg.model

    def on_setup_click(self):
        if self._setup_busy:
            return
        if self.setup_mode == "install":
            if WINDOWS:
                webbrowser.open("https://ollama.com/download/windows")
            else:  # Ollama's official one-line installer; copy it so the user can paste it in a terminal
                cmd = "curl -fsSL https://ollama.com/install.sh | sh"
                self.root.clipboard_clear()
                self.root.clipboard_append(cmd)
                self.set_status(f"⧉ Copied. Paste in a terminal and press Enter:  {cmd}", AMBER)
                webbrowser.open("https://ollama.com/download/linux")
            self._show_setup("recheck")
        elif self.setup_mode == "recheck":
            self.setup.pack_forget()
            self.set_status("Checking…", MUTED)
            threading.Thread(target=self._check_engine, daemon=True).start()
        elif self.setup_mode in ("pull", "fast", "indic"):
            if self.setup_mode == "fast":
                self.llm.model = self.cfg.model = FAST_MODEL
                save_setting("model", FAST_MODEL)
                self.model_label.configure(text=f"⚡ {FAST_MODEL}")
            self._setup_busy = True
            self.setup_btn.configure(bg=DIM)
            self.progress.pack(side="left", padx=12)
            threading.Thread(target=self._pull_model, daemon=True).start()

    def _pull_model(self):
        try:
            self.events.put(("engine", "DOWNLOADING", AMBER))
            target = self._pending_model()
            puller = OllamaClient(self.cfg.ollama_url, target, self.cfg.assistant_name)
            for status, frac in puller.pull_stream():
                pct = f" {frac * 100:.0f}%" if frac >= 0 else ""
                self.events.put(("status", f"Downloading model: {status}{pct}", MUTED))
                self.events.put(("progress", frac))
            self._pull_target = None
            self.events.put(("setup_hide",))
            self._refresh_installed()
            self._apply_model_for_language_bg()
            self._warm_up()
        except OllamaError as e:
            self.events.put(("status", f"⚠ {e}", ERR))
            self.events.put(("setup_enable",))

    # ================= typing & running =================
    def on_key_release(self, event):
        if event.keysym in ("Return", "Down", "Up", "Escape"):
            return
        q = self.text()
        if self.answer_text and not self.streaming and self.busy_since is None:
            self._hide_answer()
        matches = [a for a in self.apps if q and len(q) >= 2 and a.matches(q)][:6]
        self.show_results([(a.name, a) for a in matches])

    def on_submit(self, _event=None):
        text = self.text()
        if not text:
            return
        action = route(text, self.apps, self.cfg.connectors)
        if isinstance(action, UseModel):
            self.use_model(action)
            return
        self._apply_model_for_language()
        self._generation += 1
        gen = self._generation
        self.busy_since = time.monotonic()
        self.show_results([])
        self.pulse.pack(side="left", padx=(0, 10), before=self.status)
        self.pulse.start()
        self._tick(gen)

        def worker():
            try:
                res = actions.run(action, self.cfg, self.llm)
                if res.message:
                    self.events.put(("status", res.message))
                if res.files:
                    self.events.put(("files", res.files))
                if res.quit_app:
                    self.events.put(("quit",))
                    return
                if res.stream is None:
                    self.events.put(("done", gen))
                    if res.hide_window:
                        time.sleep(0.4)
                        self.events.put(("hide",))
                    return
                first = True
                try:
                    for piece in res.stream:
                        if gen != self._generation:
                            return
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

    def _tick(self, gen: int):
        if gen != self._generation or self.busy_since is None:
            return
        secs = int(time.monotonic() - self.busy_since)
        lang = self.llm.language
        where = f" in {LANGUAGE_BY_NAME[lang][1]}" if lang != "English" else ""
        hint = "  ·  older laptops can take 20-60s" if secs >= 8 else ""
        self.set_status(f"Thinking{where}…  {secs}s{hint}", CYAN)
        self.foot_hint.configure(text="Esc  stop")
        self.root.after(1000, lambda: self._tick(gen))

    def on_escape(self, _event=None):
        if self.busy_since is not None or self.streaming:
            self._generation += 1
            self.busy_since = None
            self.streaming = False
            self.pulse.stop()
            self.pulse.pack_forget()
            self._answer_done()
            self.foot_hint.configure(text=self._hint_text())
            self.set_status("■ Stopped.", MUTED)
        else:
            self.hide()

    def _hint_text(self) -> str:
        key = self._hk() if WINDOWS else "Ctrl+Alt+Space"
        return f"{key}  summon   ·   ↑↓  select   ·   Esc  hide"

    # ================= results list =================
    def show_results(self, items):
        self.results = items
        self.listbox.delete(0, "end")
        for label, target in items:
            prefix = "    ▤    " if isinstance(target, Path) else "    ›    "
            self.listbox.insert("end", prefix + label)
        if items:
            self.listbox.configure(height=min(len(items), 7))
            self.listbox.pack(fill="x", pady=(self._px(8), 0), after=self.status_row)
        else:
            self.listbox.pack_forget()
        self._layout()

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

    # ================= answer card =================
    def _answer_start(self):
        self.listbox.pack_forget()
        self._answer_clear()
        lang = self.llm.language
        self.answer_lang.configure(text="" if lang == "English" else f"· {LANGUAGE_BY_NAME[lang][1]}")
        self.card.pack(fill="both", expand=True, pady=(self._px(10), 0), after=self.status_row)
        if self.status.cget("text").startswith("Thinking"):
            self.set_status("")
        self._layout()

    def _answer_clear(self):
        self.answer_text = ""
        self.answer.configure(state="normal")
        self.answer.delete("1.0", "end")
        self.answer.insert("end", "▍", "cursor")   # glowing cursor while the answer streams in
        self.answer.configure(state="disabled", height=3)

    def _answer_append(self, piece: str):
        self.answer_text += piece
        self.answer.configure(state="normal")
        self.answer.insert("cursor.first" if self.answer.tag_ranges("cursor") else "end", piece)
        self.answer.see("end")
        self.answer.configure(state="disabled")
        self._fit_answer()

    def _answer_done(self):
        if self.answer.tag_ranges("cursor"):
            self.answer.configure(state="normal")
            self.answer.delete("cursor.first", "cursor.last")
            self.answer.configure(state="disabled")
        self._fit_answer()

    def _fit_answer(self):
        """Grow the card with the answer (3 to 14 lines), then scroll."""
        self.answer.update_idletasks()
        try:
            lines = int(self.answer.count("1.0", "end", "displaylines")[0])
        except (TypeError, tk.TclError):
            lines = int(self.answer.index("end-1c").split(".")[0])
        want = max(3, min(14, lines))
        if int(self.answer.cget("height")) != want:
            self.answer.configure(height=want)
            self._layout()

    def _hide_answer(self):
        if self.card.winfo_ismapped():
            self.card.pack_forget()
            self._layout()

    def copy_answer(self):
        if self.answer_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.answer_text.strip())
            self.set_status("⧉ Copied to clipboard.", GREEN)

    def set_status(self, text: str, color: str = CYAN):
        self.status.configure(text=text, fg=color)

    def run(self):
        self.root.mainloop()


def _enable_dpi_awareness() -> None:
    """Without this, Windows bitmap-scales the window on high-DPI screens: blurry and off-centre."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def run_gui(cfg: Config) -> int:
    _enable_dpi_awareness()
    instance = SingleInstance()
    if not instance.is_primary:
        instance.signal_primary()
        return 0
    CommandBar(cfg, instance).run()
    return 0
