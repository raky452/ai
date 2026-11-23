import json
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tf_assistant import files
from tf_assistant.apps import App, dedupe_sort, find_app, scan_shortcuts
from tf_assistant.llm import OllamaClient, OllamaError, ThinkFilter
from tf_assistant.router import (
    AskAssistant, FindFiles, OpenApp, OpenSettings, OpenUrl, Summarize, System, WebSearch, route,
)
from tf_assistant.winsys import MOD_NOREPEAT, parse_hotkey

APPS = dedupe_sort([
    App("Google Chrome", "Chrome"),
    App("Notepad", "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"),
    App("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
    App("Microsoft Word", "Microsoft.Office.WINWORD.EXE.15"),
    App("WhatsApp", "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"),
])


# ---------- router ----------
@pytest.mark.parametrize("text,name", [
    ("open chrome", "Google Chrome"),
    ("launch word", "Microsoft Word"),      # word-start match
    ("start calc", "Calculator"),
    ("notepad", "Notepad"),                 # bare exact name
    ("WhatsApp", "WhatsApp"),
])
def test_open_app(text, name):
    a = route(text, APPS)
    assert isinstance(a, OpenApp) and a.app.name == name


def test_settings_lock_quit(monkeypatch):
    monkeypatch.setattr("tf_assistant.router.sys.platform", "win32")
    assert route("settings wifi", APPS) == OpenSettings("wifi", "ms-settings:network-wifi")
    assert route("open settings bluetooth", APPS) == OpenSettings("bluetooth", "ms-settings:bluetooth")
    assert route("settings", APPS) == OpenSettings("home", "ms-settings:")
    monkeypatch.setattr("tf_assistant.router.sys.platform", "linux")
    assert route("settings wifi", APPS) == OpenSettings("wifi", "linux-settings:wifi")
    assert route("lock pc", APPS) == System("lock")
    assert route("exit assistant", APPS) == System("quit")


def test_files_and_paths():
    assert route("find files invoice", APPS) == FindFiles("invoice")
    s = route(r"summarise C:\Users\Rakesh\Downloads\report.pdf", APPS)
    assert isinstance(s, Summarize) and s.path.name.endswith("report.pdf")


def test_web():
    assert route("technoforesight.in", APPS) == OpenUrl("https://technoforesight.in")
    assert route("google ipl score", APPS) == WebSearch("ipl score")


def test_assistant_fallback():
    assert route("write a leave letter for tomorrow", APPS) == AskAssistant("write a leave letter for tomorrow")
    assert isinstance(route("explain blockchain simply", APPS), AskAssistant)


def test_find_app_exact():
    assert find_app("calc", APPS, exact_only=True) is None


# ---------- start menu scan ----------
def test_scan_shortcuts(tmp_path):
    (tmp_path / "Tools").mkdir()
    (tmp_path / "Tools" / "VLC media player.lnk").write_bytes(b"x")
    (tmp_path / "Uninstall VLC.lnk").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [a.name for a in scan_shortcuts([tmp_path])] == ["VLC media player"]


# ---------- hotkey ----------
def test_parse_hotkey():
    assert parse_hotkey("alt+space") == (0x0001 | MOD_NOREPEAT, 0x20)
    assert parse_hotkey("Ctrl+Shift+K") == (0x0002 | 0x0004 | MOD_NOREPEAT, ord("K"))
    assert parse_hotkey("win+f2")[1] == 0x71
    with pytest.raises(ValueError):
        parse_hotkey("hyper+space")


# ---------- documents ----------
def test_read_docx_and_text(tmp_path):
    docx = tmp_path / "letter.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml",
                   "<w:document><w:body><w:p><w:r><w:t>Hello Rakesh</w:t></w:r></w:p>"
                   "<w:p><w:r><w:t>Line two</w:t></w:r></w:p></w:body></w:document>")
    assert files.read_document(docx, 1000) == "Hello Rakesh\nLine two"
    (tmp_path / "Invoice_Nov.txt").write_text("Total: Rs 4,500")
    found = files.find_files("invoice", (str(tmp_path),))
    assert [p.name for p in found] == ["Invoice_Nov.txt"]
    with pytest.raises(RuntimeError):
        (tmp_path / "pic.png").write_bytes(b"x")
        files.read_document(tmp_path / "pic.png", 10)


@pytest.mark.skipif("pypdf" not in sys.modules and __import__("importlib").util.find_spec("pypdf") is None,
                    reason="pypdf not installed")
def test_read_pdf(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, StreamObject, DictionaryObject

    w = PdfWriter()
    page = w.add_blank_page(width=300, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    content = StreamObject()
    content.set_data(b"BT /F1 12 Tf 20 100 Td (Fee due 5 Dec) Tj ET")
    page[NameObject("/Contents")] = w._add_object(content)
    pdf = tmp_path / "fee.pdf"
    with pdf.open("wb") as fh:
        w.write(fh)
    assert "Fee due 5 Dec" in files.read_document(pdf, 1000)


# ---------- streaming filter ----------
def test_think_filter():
    f = ThinkFilter()
    out = "".join("".join(f.feed(c)) for c in ["<thi", "nk>x</th", "ink>\nHi ", "there"]) + f.flush()
    assert out == "Hi there"


def test_thinking_without_opening_tag():
    """Thinking-only models (Ollama's plain qwen3:4b) send reasoning, '</think>', then the answer."""
    from tf_assistant.llm import RESET, join_stream, strip_thinking
    f = ThinkFilter()
    pieces = []
    for c in ["Okay, the user said hi. Let me th", "ink about it.\n</th", "ink>\n\nHello! How can I help?"]:
        pieces += f.feed(c)
    pieces.append(f.flush())
    assert RESET in pieces
    assert join_stream(pieces) == "Hello! How can I help?"
    assert strip_thinking("reasoning...\n</think>\n\nHi!") == "Hi!"


# ---------- Ollama client ----------
class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": "qwen3:4b"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.end_headers()
        _Fake.last_body = body
        if self.path == "/api/generate":
            self.wfile.write(json.dumps({"response": "Hello!", "eval_count": 20, "eval_duration": 4_000_000_000}).encode())
            return
        if self.path == "/api/pull":
            events = [{"status": "pulling manifest"}, {"status": "downloading", "total": 100, "completed": 50},
                      {"status": "success"}]
        else:
            events = [{"message": {"content": "<think>x</think>Namaste"}, "done": False},
                      {"message": {"content": "!"}, "done": True}]
        for e in events:
            self.wfile.write((json.dumps(e) + "\n").encode())


@pytest.fixture()
def server():
    s = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def test_ollama(server):
    c = OllamaClient(server, "qwen3:4b")
    assert c.check() is None
    assert c.chat("hi") == "Namaste!"
    # thinking is switched off and answers are capped so slow PCs never stall silently
    assert _Fake.last_body["think"] is False
    assert _Fake.last_body["options"]["num_predict"] == 700
    assert _Fake.last_body["options"]["presence_penalty"] > 0   # anti-repetition for Indian languages
    assert c.speed_test() == pytest.approx(5.0)
    assert list(c.pull_stream())[1] == ("downloading", 0.5)
    assert "ollama pull" in OllamaClient(server, "qwen3:8b").check()
    with pytest.raises(OllamaError):
        OllamaClient("http://127.0.0.1:9", "x").installed_models()


def test_chat_cancel(server):
    import threading as _t
    cancel = _t.Event()
    stream = OllamaClient(server, "qwen3:4b").chat_stream("hi", cancel=cancel)
    cancel.set()
    assert list(stream) == []


def test_save_setting(tmp_path):
    from tf_assistant.config import load_config, save_setting, user_set
    cfg = tmp_path / "config.toml"
    cfg.write_text('# my settings\nhotkey = "ctrl+space"\n')
    assert not user_set("model", cfg)
    save_setting("model", "qwen3:1.7b", cfg)
    save_setting("model", "qwen3:0.6b", cfg)
    assert user_set("model", cfg)
    text = cfg.read_text()
    assert text.count("model =") == 1 and "# my settings" in text
    loaded = load_config(cfg)
    assert loaded.model == "qwen3:0.6b" and loaded.hotkey == "ctrl+space"


def test_response_language_prompt():
    from tf_assistant.llm import LANGUAGES, language_instruction
    names = [n for n, _, _ in LANGUAGES]
    assert {"English", "Telugu", "Hindi", "Tamil", "Kannada", "Malayalam"} <= set(names)
    c = OllamaClient("http://x", "m", language="Telugu")
    assert "Telugu (తెలుగు)" in c.system_prompt and "Telugu script" in c.system_prompt
    c.language = "Hindi"
    assert "हिन्दी" in c.system_prompt
    assert "English" in language_instruction("English")


def test_connectors_routing_and_saving(tmp_path):
    from tf_assistant.config import DEFAULT_CONNECTORS, load_config, save_connector, save_setting
    from tf_assistant.router import Connector
    conns = dict(DEFAULT_CONNECTORS, store="https://technoforesight.in")
    assert route("store", APPS, conns) == Connector("store", "https://technoforesight.in")
    yt = route("yt telugu songs", APPS, conns)
    assert yt.url == "https://www.youtube.com/results?search_query=telugu+songs"
    assert route("add connector shop technoforesight.in/converters", APPS, conns) == \
        Connector("shop", "https://technoforesight.in/converters", manage="add")
    assert route("remove connector shop", APPS, conns).manage == "remove"
    assert route("connectors", APPS, conns).manage == "list"
    # apps and AI still work when no connector matches
    assert isinstance(route("open chrome", APPS, conns), OpenApp)
    assert isinstance(route("what is GST", APPS, conns), AskAssistant)

    cfg = tmp_path / "config.toml"
    save_connector("store", "https://technoforesight.in", cfg)
    save_setting("model", "qwen3:1.7b", cfg)         # top-level key must stay above the table
    save_connector("amz", "https://example.com/s?q={q}", cfg)
    save_connector("store", None, cfg)
    loaded = load_config(cfg)
    assert loaded.model == "qwen3:1.7b"
    assert loaded.connectors["amz"] == "https://example.com/s?q={q}"
    assert "store" not in loaded.connectors and "yt" in loaded.connectors


def test_use_model_command():
    from tf_assistant.router import UseModel
    assert route("use model gemma4:e4b for indian languages", APPS) == UseModel("gemma4:e4b", "indic")
    assert route("use model qwen3:8b", APPS) == UseModel("qwen3:8b", "all")
    assert route("switch to model gemma4:e2b for other languages", APPS) == UseModel("gemma4:e2b", "indic")


def test_markdown_cleaner_handles_split_chunks():
    from tf_assistant.llm import MarkdownCleaner
    src = "**Subject:** [మీ పేరు]\n## Heading\n* one\n- two\nkeep 2*3 and file_name.txt\n"
    want = "Subject: [మీ పేరు]\nHeading\n• one\n• two\nkeep 2*3 and file_name.txt\n"
    for size in (1, 2, 3, 5, 64):
        m = MarkdownCleaner()
        out = "".join(m.feed(src[i:i + size]) for i in range(0, len(src), size)) + m.flush()
        assert out == want, size


def test_indian_language_prompt_forbids_other_scripts():
    from tf_assistant.llm import language_instruction
    text = language_instruction("Telugu")
    assert "Telugu script" in text and "Vietnamese" in text and "Markdown" in text


def test_indic_model_default_and_fallback():
    import importlib.util

    import pytest
    if importlib.util.find_spec("tkinter") is None:
        pytest.skip("no tkinter")
    from tf_assistant.config import Config
    from tf_assistant.ui import CommandBar
    cfg = Config()
    assert cfg.indic_model == "gemma4:e4b"

    class Fake:  # just enough of CommandBar for the model choice
        _has_model = CommandBar._has_model
        _model_for_language = CommandBar._model_for_language

    bar = Fake()
    bar.cfg = cfg
    bar.llm = type("L", (), {"language": "Telugu"})()
    bar._installed = {"qwen3:4b-instruct"}
    assert bar._model_for_language() == "qwen3:4b-instruct"   # not downloaded yet: keep working
    bar._installed.add("gemma4:e4b")
    assert bar._model_for_language() == "gemma4:e4b"
    bar.llm.language = "English"
    assert bar._model_for_language() == "qwen3:4b-instruct"
