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


def test_settings_lock_quit():
    assert route("settings wifi", APPS) == OpenSettings("wifi", "ms-settings:network-wifi")
    assert route("open settings bluetooth", APPS) == OpenSettings("bluetooth", "ms-settings:bluetooth")
    assert route("settings", APPS) == OpenSettings("home", "ms-settings:")
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
    out = "".join(f.feed(c) for c in ["<thi", "nk>x</th", "ink>\nHi ", "there"]) + f.flush()
    assert out == "Hi there"


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
    assert list(c.pull_stream())[1] == ("downloading", 0.5)
    assert "ollama pull" in OllamaClient(server, "qwen3:8b").check()
    with pytest.raises(OllamaError):
        OllamaClient("http://127.0.0.1:9", "x").installed_models()
