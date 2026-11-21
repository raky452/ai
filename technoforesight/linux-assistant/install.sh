#!/usr/bin/env bash
# Installs TechnoForesight Assistant for the current user (no root needed except for apt packages).
# Tested target: Ubuntu / Debian / Linux Mint. Fedora users: see README.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${TF_MODEL:-qwen3:4b}"

say() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

say "Checking system packages (GTK 4 bindings, PDF reader, pipx)"
missing=()
python3 -c "import gi; gi.require_version('Gtk','4.0')" 2>/dev/null || missing+=(python3-gi gir1.2-gtk-4.0)
command -v pdftotext >/dev/null || missing+=(poppler-utils)
command -v pipx >/dev/null || missing+=(pipx)
if ((${#missing[@]})); then
  echo "Installing: ${missing[*]}"
  sudo apt-get update -qq && sudo apt-get install -y "${missing[@]}"
fi

say "Installing the assistant"
# --system-site-packages lets it use the system's GTK bindings
pipx install --force --system-site-packages "$HERE"

say "Adding menu entry and icon"
mkdir -p ~/.local/share/applications ~/.local/share/icons/hicolor/scalable/apps
cp "$HERE/data/in.technoforesight.Assistant.desktop" ~/.local/share/applications/
cp "$HERE/data/in.technoforesight.Assistant.svg" ~/.local/share/icons/hicolor/scalable/apps/

say "Checking the local AI engine (Ollama)"
if ! command -v ollama >/dev/null; then
  cat <<EOF
Ollama is not installed. Install it from https://ollama.com/download
(review the script first), then run:
    ollama pull $MODEL
EOF
else
  ollama pull "$MODEL"
fi

say "Done!"
cat <<EOF
Try it:      technoforesight-assistant doctor
Open bar:    technoforesight-assistant
Shortcut:    Settings → Keyboard → Custom Shortcuts → add
             command 'technoforesight-assistant', e.g. Super+Space
EOF
