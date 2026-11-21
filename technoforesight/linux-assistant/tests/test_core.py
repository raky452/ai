import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from technoforesight_assistant import files
from technoforesight_assistant.apps import DesktopApp, build_command, find_app, load_apps, parse_desktop_file
from technoforesight_assistant.llm import OllamaClient, OllamaError, ThinkFilter, strip_thinking
from technoforesight_assistant.router import (
    AskAssistant, FindFiles, OpenApp, OpenUrl, Summarize, WebSearch, route,
)

APPS = [
    DesktopApp("Firefox Web Browser", "firefox %u", "/x/firefox.desktop", ("browser", "internet")),
    DesktopApp("Files", "nautilus --new-window %U", "/x/nautilus.desktop"),
    DesktopApp("LibreOffice Writer", "libreoffice --writer %U", "/x/writer.desktop", ("word",)),
]


# ---------------- router ----------------
@pytest.mark.parametrize("text,expected_app", [
    ("open firefox", "Firefox Web Browser"),
    ("launch writer", "LibreOffice Writer"),
    ("start browser", "Firefox Web Browser"),   # keyword match
    ("files", "Files"),                         # bare exact name
])
def test_open_app(text, expected_app):
    action = route(text, APPS)
    assert isinstance(action, OpenApp) and action.app.name == expected_app


def test_find_files():
    assert route("find files invoice", APPS) == FindFiles("invoice")
    assert route("search for resume", APPS) == FindFiles("resume")


def test_summarize_path():
    action = route("summarise ~/Documents/report.pdf", APPS)
    assert isinstance(action, Summarize) and action.path == Path.home() / "Documents/report.pdf"


def test_urls_and_web():
    assert route("www.technoforesight.in", APPS) == OpenUrl("https://www.technoforesight.in")
    assert route("google weather hyderabad", APPS) == WebSearch("weather hyderabad")


def test_falls_back_to_assistant():
    assert route("what is the capital of france", APPS) == AskAssistant("what is the capital of france")
    # "explain" with no path is a question, not a summarise command
    assert isinstance(route("explain quantum computing", APPS), AskAssistant)


# ---------------- apps ----------------
def test_parse_desktop_files(tmp_path):
    apps_dir = tmp_path / "applications"
    apps_dir.mkdir()
    (apps_dir / "good.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Good App\nExec=good --flag %F\nKeywords=a;b;\n"
        "[Desktop Action new]\nName=Ignored\n"
    )
    (apps_dir / "hidden.desktop").write_text("[Desktop Entry]\nType=Application\nName=H\nExec=h\nNoDisplay=true\n")
    (apps_dir / "link.desktop").write_text("[Desktop Entry]\nType=Link\nName=L\nURL=x\n")
    apps = load_apps([apps_dir])
    assert [a.name for a in apps] == ["Good App"]
    assert apps[0].keywords == ("a", "b")
    assert build_command(apps[0].exec_line) == ["good", "--flag"]
    assert parse_desktop_file(apps_dir / "hidden.desktop") is None


def test_find_app_exact_only():
    assert find_app("fire", APPS, exact_only=True) is None
    assert find_app("fire", APPS).name == "Firefox Web Browser"


# ---------------- files ----------------
def test_find_and_read_files(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "Invoice_Nov.txt").write_text("Total: Rs 4,500")
    (tmp_path / ".secret").mkdir()
    (tmp_path / ".secret" / "invoice_hidden.txt").write_text("x")
    found = files.find_files("invoice", (str(tmp_path),))
    assert [p.name for p in found] == ["Invoice_Nov.txt"]
    assert files.read_document(found[0], 100) == "Total: Rs 4,500"
    with pytest.raises(RuntimeError):
        (tmp_path / "pic.png").write_bytes(b"\x89PNG")
        files.read_document(tmp_path / "pic.png", 100)


# ---------------- thinking filter ----------------
def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>\nHello") == "Hello"


def test_think_filter_streaming_split_tags():
    f = ThinkFilter()
    chunks = ["<thi", "nk>secret ", "reasoning</th", "ink>\n\nHel", "lo <", "b>world</b>"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "Hello <b>world</b>"


# ---------------- Ollama client against a fake local server ----------------
class _FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen3:4b"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert payload["messages"][-1]["content"] == "hi"
        self.send_response(200)
        self.end_headers()
        for piece, done in [("<think>x</think>", False), ("Namaste", False), ("!", True)]:
            self.wfile.write((json.dumps({"message": {"content": piece}, "done": done}) + "\n").encode())


@pytest.fixture()
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_chat_and_check(fake_server):
    client = OllamaClient(fake_server, "qwen3:4b")
    assert client.check() is None
    assert client.chat("hi") == "Namaste!"
    assert "ollama pull" in OllamaClient(fake_server, "gemma3:4b").check()


def test_ollama_unreachable():
    client = OllamaClient("http://127.0.0.1:9", "qwen3:4b")
    with pytest.raises(OllamaError):
        client.installed_models()
