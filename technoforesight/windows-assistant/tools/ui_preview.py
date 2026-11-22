"""Renders the command bar in several states and saves screenshots (run on Windows CI).

    python tools/ui_preview.py out_dir
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageGrab  # noqa: E402

from tf_assistant import ui  # noqa: E402
from tf_assistant.apps import App  # noqa: E402
from tf_assistant.config import Config  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "preview")
OUT.mkdir(parents=True, exist_ok=True)

TELUGU = ("ప్రియమైన సార్,\n\nనాకు జ్వరం ఉన్నందున రేపు ఆఫీసుకు రాలేను. దయచేసి ఒక రోజు సెలవు మంజూరు చేయగలరు.\n\n"
          "ధన్యవాదాలు,\nరాకేష్")
ENGLISH = ("Dear Sir,\n\nI have a fever and won't be able to come to the office tomorrow. "
           "Kindly grant me one day of leave.\n\nThank you,\nRakesh")


class _Inst:
    is_primary = True

    def listen(self, _cb):
        pass


class Preview(ui.CommandBar):
    def _check_engine(self):
        self.events.put(("engine", "LOCAL · OFFLINE", ui.GREEN))
        self.events.put(("speed", 9.1))

    def _load_apps(self):
        self.events.put(("apps", [App(n, n) for n in ("Calculator", "Camera", "Calendar", "Google Chrome",
                                                        "WhatsApp", "Microsoft Word", "Notepad")]))


bar = Preview(Config(hotkey="ctrl+alt+shift+f11"), _Inst())
root = bar.root


def settle(sec=0.6):
    end = time.time() + sec
    while time.time() < end:
        root.update()
        time.sleep(0.02)


def shot(name):
    settle(0.8)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(OUT / f"{name}.png")
    print("saved", name, w, h)


settle(1.2)
bar.set_status("✔ AI ready. Ask me anything, it all stays on this PC.", ui.GREEN)
shot("1_idle")

bar._clear_placeholder()
bar.entry.insert(0, "ca")
bar.show_results([(a.name, a) for a in bar.apps if "ca" in a.name.lower()])
shot("2_search")

bar.show_results([])
bar.entry.delete(0, "end")
bar.entry.insert(0, "write a 3 line leave letter for tomorrow, I have a fever")
bar._generation += 1
bar.busy_since = time.monotonic() - 4
bar.pulse.pack(side="left", padx=(0, 10), before=bar.status)
bar.pulse.start()
bar._tick(bar._generation)
shot("3_thinking")

bar.events.put(("answer_start", bar._generation))
bar.events.put(("answer", ENGLISH, bar._generation))
bar.events.put(("done", bar._generation))
shot("4_answer_english")

bar.llm.language = "Telugu"
bar.lang_chip.configure(text=bar._lang_label())
bar.answer.configure(font=bar._answer_font())
bar._generation += 1
bar.events.put(("answer_start", bar._generation))
bar.events.put(("answer", TELUGU, bar._generation))
bar.events.put(("done", bar._generation))
shot("5_answer_telugu")
root.destroy()
