#!/usr/bin/env bash
# End-to-end test of the INSTALLED Linux app against a real local Ollama (run by CI after installing the .deb).
# Needs: technoforesight-assistant on PATH, Ollama running with qwen3:4b-instruct and gemma4:e4b pulled,
# Xvfb + xdotool + ImageMagick for the window part. Screenshots go to $OUT and are printed as base64.
set -uo pipefail
OUT="${1:-e2e-out}"; mkdir -p "$OUT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP=technoforesight-assistant
CONF="$HOME/.config/technoforesight/config.toml"
fail=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
setconf() { mkdir -p "$(dirname "$CONF")"; printf '%s\n' "$@" > "$CONF"; }

echo "=== CLI ==="
setconf 'response_language = "English"'
$APP doctor | tee "$OUT/doctor.txt"; check "doctor reports ready" 'grep -q "Ready" "$OUT/doctor.txt"'

$APP ask "What is the capital of India? Answer in one word." | tee "$OUT/ask_en.txt"
check "English answer is correct" 'grep -qi "delhi" "$OUT/ask_en.txt"'

python3 "$HERE/make_test_pdf.py" /tmp/Sample-Fee-Notice.pdf
$APP do "summarise /tmp/Sample-Fee-Notice.pdf" | tee "$OUT/summary.txt"
check "PDF summary mentions the fee or due date" 'grep -Eqi "12,?500|december|15" "$OUT/summary.txt"'
check "no Markdown symbols in answers" '! grep -q "\*\*" "$OUT/summary.txt" "$OUT/ask_en.txt"'

$APP do "add connector store https://technoforesight.in" | tee "$OUT/conn.txt"
check "connector saved" 'grep -q "store" "$CONF"'

setconf 'response_language = "Telugu"' '' '[connectors]' 'store = "https://technoforesight.in"'
for q in "explain what UPI is in 3 simple lines" \
         "write a professional email to a food delivery company asking them to open a delivery centre in my area, with blanks for my details"; do
  f="$OUT/te_$(echo "$q" | cut -c1-12 | tr ' ' '_').txt"
  start=$(date +%s); $APP ask "$q" > "$f"; secs=$(( $(date +%s) - start ))
  cat "$f"; echo "($secs s)"
  check "Telugu answer is clean Telugu: $q" 'python3 "$HERE/check_indic.py" "$f" telugu'
done

echo "=== Window ==="
export DISPLAY=:99
Xvfb :99 -screen 0 1366x768x24 >/dev/null 2>&1 & XVFB=$!
sleep 2
shot() { import -window root "$OUT/$1.png"; echo "===PNG $1.png==="; base64 -w 8000 "$OUT/$1.png" | sed 's/^/B64:/'; echo "===END==="; }
export TF_ANSWER_LOG="$OUT/window_answers.txt"; : > "$TF_ANSWER_LOG"
$APP & PID=$!
for _ in $(seq 60); do WID=$(xdotool search --class TechnoForesightAssistant 2>/dev/null | head -1); [ -n "$WID" ] && break; sleep 1; done
check "window opens" '[ -n "${WID:-}" ]'
sleep 25; shot ui_ready
$APP; sleep 1
check "second launch hands over to the running copy" '[ "$(pgrep -fc "python3 -m tf_assistant$")" = "1" ]'
xdotool windowactivate --sync "$WID" 2>/dev/null || xdotool windowfocus "$WID"
xdotool type --delay 40 "explain what UPI is in 3 simple lines"; xdotool key Return
waitans() { for _ in $(seq "$2"); do [ "$(grep -c '^=====$' "$TF_ANSWER_LOG")" -ge "$1" ] && return 0; sleep 2; done; return 1; }
waitans 1 90; sleep 1; shot ui_telugu_upi
sed -n '1,/^=====$/p' "$TF_ANSWER_LOG" | sed '$d' > "$OUT/win_te.txt"; cat "$OUT/win_te.txt"
check "window: Telugu answer is clean Telugu" 'python3 "$HERE/check_indic.py" "$OUT/win_te.txt" telugu'
xdotool key ctrl+a; xdotool type --delay 20 "summarise /tmp/Sample-Fee-Notice.pdf"; xdotool key Return
waitans 2 120; sleep 1; shot ui_pdf_summary
awk 'BEGIN{n=0} /^=====$/{n++; next} n==1' "$TF_ANSWER_LOG" > "$OUT/win_pdf.txt"; cat "$OUT/win_pdf.txt"
check "window: Ctrl+A replaced the text and the PDF summary is in Telugu" 'python3 "$HERE/check_indic.py" "$OUT/win_pdf.txt" telugu && grep -Eq "12,?500|15" "$OUT/win_pdf.txt"'
check "app still running after the session" 'kill -0 $PID'
kill $PID $XVFB 2>/dev/null
exit $fail
