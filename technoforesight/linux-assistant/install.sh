#!/usr/bin/env bash
# Installs TechnoForesight Assistant for the current user from the portable tarball (any distro).
# Ubuntu/Debian users can install the .deb instead.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/technoforesight-assistant"
say() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

say "Checking Python and Tk"
if ! python3 -c 'import sys, tkinter; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo "Python 3.10+ with Tk is needed. Install it, then run ./install.sh again:"
  echo "  Ubuntu/Debian/Mint : sudo apt install python3-tk"
  echo "  Fedora             : sudo dnf install python3-tkinter"
  echo "  Arch/Manjaro       : sudo pacman -S tk"
  echo "  openSUSE           : sudo zypper install python3-tk"
  exit 1
fi

say "Installing to $APP_HOME"
rm -rf "$APP_HOME" && mkdir -p "$APP_HOME" "$HOME/.local/bin"
cp -r "$HERE/app/." "$APP_HOME/"
sed "s#/opt/technoforesight-assistant#$APP_HOME#" "$HERE/technoforesight-assistant" > "$HOME/.local/bin/technoforesight-assistant"
chmod +x "$HOME/.local/bin/technoforesight-assistant"

say "Adding the menu entry and icon"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
mkdir -p "$APPS" "$ICONS/256x256/apps" "$ICONS/scalable/apps"
sed "s#^Exec=.*#Exec=$HOME/.local/bin/technoforesight-assistant#" "$HERE/in.technoforesight.Assistant.desktop" > "$APPS/in.technoforesight.Assistant.desktop"
cp "$APP_HOME/assets/icon.png" "$ICONS/256x256/apps/in.technoforesight.Assistant.png"
cp "$APP_HOME/assets/icon.svg" "$ICONS/scalable/apps/in.technoforesight.Assistant.svg"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$APPS" || true

say "Done!"
command -v ollama >/dev/null || echo "The AI engine (Ollama, free) isn't installed yet: the app shows a button with the one-line installer."
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "Note: add ~/.local/bin to your PATH to run it from a terminal.";; esac
echo "Open 'TechnoForesight Assistant' from your app menu, or run: technoforesight-assistant"
