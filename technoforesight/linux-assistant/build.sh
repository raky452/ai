#!/usr/bin/env bash
# Builds the Linux packages from the shared source in ../windows-assistant/tf_assistant:
#   dist/technoforesight-assistant_<version>_all.deb   (Ubuntu, Debian, Mint, Pop!_OS...)
#   dist/TechnoForesightAssistant-Linux.tar.gz          (any distro: ./install.sh, no root)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../windows-assistant"
VERSION="$(cd "$SRC" && python3 -c 'import tf_assistant; print(tf_assistant.__version__)')"
DIST="$HERE/dist"
rm -rf "$DIST" && mkdir -p "$DIST"

stage_app() {  # $1 = destination folder for the app files
  mkdir -p "$1/assets"
  cp -r "$SRC/tf_assistant" "$1/"
  find "$1/tf_assistant" -name '__pycache__' -prune -exec rm -rf {} +
  rm -f "$1/tf_assistant/winsys.py"
  cp "$SRC/assets/icon.png" "$SRC/assets/icon.svg" "$1/assets/"
}

# ---------- .deb ----------
ROOT="$(mktemp -d)/pkg"
stage_app "$ROOT/opt/technoforesight-assistant"
install -Dm755 "$HERE/data/technoforesight-assistant" "$ROOT/usr/bin/technoforesight-assistant"
install -Dm644 "$HERE/data/in.technoforesight.Assistant.desktop" "$ROOT/usr/share/applications/in.technoforesight.Assistant.desktop"
install -Dm644 "$SRC/assets/icon.png" "$ROOT/usr/share/icons/hicolor/256x256/apps/in.technoforesight.Assistant.png"
install -Dm644 "$SRC/assets/icon.svg" "$ROOT/usr/share/icons/hicolor/scalable/apps/in.technoforesight.Assistant.svg"
mkdir -p "$ROOT/DEBIAN"
SIZE="$(du -sk "$ROOT" | cut -f1)"
cat > "$ROOT/DEBIAN/control" <<CTRL
Package: technoforesight-assistant
Version: $VERSION
Section: utils
Priority: optional
Architecture: all
Depends: python3 (>= 3.10), python3-tk, xdg-utils
Recommends: python3-pypdf | poppler-utils, fonts-noto-core, libglib2.0-bin, python3-tomli | python3 (>= 3.11)
Installed-Size: $SIZE
Maintainer: TechnoForesight <support@technoforesight.in>
Homepage: https://technoforesight.in/assistant-linux
Description: Offline AI command bar for Linux
 Press Ctrl+Alt+Space, type what you want: open apps, find files, summarise
 PDFs and Word files, or ask the AI. Answers in English, Telugu, Hindi and
 9 more Indian languages. The AI runs locally through Ollama, so questions
 and documents never leave the computer.
CTRL
cat > "$ROOT/DEBIAN/postinst" <<'POST'
#!/bin/sh
set -e
command -v update-desktop-database >/dev/null && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
exit 0
POST
chmod 755 "$ROOT/DEBIAN/postinst"
dpkg-deb --root-owner-group --build "$ROOT" "$DIST/technoforesight-assistant_${VERSION}_all.deb" >/dev/null
cp "$DIST/technoforesight-assistant_${VERSION}_all.deb" "$DIST/TechnoForesightAssistant-Linux.deb"

# ---------- portable tarball ----------
T="$(mktemp -d)/TechnoForesightAssistant-Linux"
stage_app "$T/app"
cp "$HERE/install.sh" "$HERE/data/technoforesight-assistant" "$HERE/data/in.technoforesight.Assistant.desktop" "$T/"
tar -C "$(dirname "$T")" -czf "$DIST/TechnoForesightAssistant-Linux-$VERSION.tar.gz" TechnoForesightAssistant-Linux
cp "$DIST/TechnoForesightAssistant-Linux-$VERSION.tar.gz" "$DIST/TechnoForesightAssistant-Linux.tar.gz"
ls -l "$DIST"
