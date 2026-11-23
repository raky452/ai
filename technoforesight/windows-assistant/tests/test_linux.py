"""Linux layer: .desktop discovery, launching command, single instance, settings panels."""

import sys
import threading

import pytest

from tf_assistant import apps

linux_only = pytest.mark.skipif(sys.platform == "win32", reason="Linux only")


def test_parse_desktop_files(tmp_path):
    (tmp_path / "firefox.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Firefox Web Browser\nExec=firefox %u\nKeywords=web;browser;\n")
    (tmp_path / "hidden.desktop").write_text("[Desktop Entry]\nType=Application\nName=Hidden\nExec=x\nNoDisplay=true\n")
    (tmp_path / "link.desktop").write_text("[Desktop Entry]\nType=Link\nName=Site\nURL=https://x\n")
    found = apps.scan_desktop_files([tmp_path])
    assert [a.name for a in found] == ["Firefox Web Browser"]
    assert found[0].matches("browser") and found[0].matches("fire")
    assert apps.build_command(found[0].exec_line) == ["firefox"]
    assert apps.find_app("firefox", found) is found[0]


@linux_only
def test_single_instance_signal():
    from tf_assistant.linuxsys import SingleInstance
    first = SingleInstance("tf-test-instance")
    second = SingleInstance("tf-test-instance")
    try:
        assert first.is_primary and not second.is_primary
        got = threading.Event()
        first.listen(got.set)
        second.signal_primary()
        assert got.wait(3)
    finally:
        first.close()


def test_settings_panels_cover_every_page():
    from tf_assistant.linuxsys import SETTINGS_PANELS
    from tf_assistant.router import SETTINGS_PAGES
    assert set(SETTINGS_PAGES) <= set(SETTINGS_PANELS)


@linux_only
def test_gnome_shortcut_appends_without_touching_others(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.write_text("['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/']")
    log = tmp_path / "log"
    fake = tmp_path / "gsettings"
    fake.write_text(f"""#!/bin/sh
echo "$@" >> {log}
if [ "$1" = get ]; then cat {store}; fi
if [ "$1" = set ] && [ "$3" = custom-keybindings ]; then echo "$4" > {store}; fi
""")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.delenv("TF_NO_SHORTCUT", raising=False)
    from tf_assistant.linuxsys import ensure_gnome_shortcut
    assert ensure_gnome_shortcut()
    assert store.read_text().strip() == ("['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/', "
                                         "'/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/technoforesight/']")
    assert "<Control><Alt>space" in log.read_text()
    calls = log.read_text().count("\n")
    assert ensure_gnome_shortcut()                      # second run: already there, no new writes
    assert log.read_text().count("\n") == calls + 1     # just the one 'get'
