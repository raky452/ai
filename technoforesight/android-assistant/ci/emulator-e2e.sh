#!/usr/bin/env bash
# Runs inside the booted emulator job (see .github/workflows/android-assistant.yml).
# 1) "use my PC" mode against Ollama on the host (10.0.2.2) in Telugu and English, plus connectors
# 2) on-device Gemma 4 in Telugu (model file pushed into the app's private storage)
set -uo pipefail
APP=in.technoforesight.assistant
RUNNER=$APP.test/androidx.test.runner.AndroidJUnitRunner
OUT=e2e-out
mkdir -p "$OUT"
fail=0

adb wait-for-device
adb shell getprop ro.product.cpu.abi
adb install -r -g app/build/outputs/apk/debug/app-debug.apk
adb install -r -g app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk

run() {  # $1 = label, rest = am instrument args
  local label=$1; shift
  echo "=== $label ==="
  adb shell am instrument -w "$@" "$RUNNER" | tee "$OUT/$label.txt"
  if grep -q "FAILURES!!!\|INSTRUMENTATION_FAILED\|Process crashed" "$OUT/$label.txt" || ! grep -q "^OK (" "$OUT/$label.txt"; then
    echo "FAIL  $label"; fail=1
  else
    echo "PASS  $label"
  fi
}

run pc-mode -e class "$APP.EndToEndTest#pcEngine_teluguAndEnglishAnswers,$APP.EndToEndTest#connectorsWithoutAi" \
  -e pcUrl http://10.0.2.2:11434 -e pcModel gemma4:e4b

# Free the host's memory for the on-device run
ollama stop gemma4:e4b 2>/dev/null || true
sudo systemctl stop ollama 2>/dev/null || true

echo "=== pushing the on-device model ==="
adb push "$RUNNER_TEMP/gemma.litertlm" /data/local/tmp/model.litertlm
adb shell "run-as $APP sh -c 'mkdir -p files/models && cp /data/local/tmp/model.litertlm files/models/gemma-4-E2B-it.litertlm'"
adb shell rm /data/local/tmp/model.litertlm
adb shell "run-as $APP ls -l files/models"
run on-device -e class "$APP.EndToEndTest#onDevice_teluguAnswer" -e onDevice true

adb pull "/sdcard/Android/data/$APP/files/e2e" "$OUT/" || true
adb logcat -d -t 3000 > "$OUT/logcat.txt" || true
for f in "$OUT"/e2e/*.txt; do echo "--- $f"; cat "$f"; echo; done
for f in "$OUT"/e2e/*.png; do
  [ -f "$f" ] || continue
  echo "===PNG $(basename "$f")==="; base64 -w 8000 "$f" | sed 's/^/B64:/'; echo "===END==="
done
exit $fail
