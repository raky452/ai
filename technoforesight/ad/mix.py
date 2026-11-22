"""Builds the soundtrack: voice-over lines placed on their scenes, music ducked under the voice.

    python3 mix.py te|en  -> audio_te.m4a / audio_en.m4a
"""
import subprocess
import sys
from pathlib import Path

from render import DUR, SCENES, VO_OFFSET

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "assets" / "media"
L = sys.argv[1]

inputs, filters, labels = ["-i", str(MEDIA / "music.mp3")], [], []
for i, (name, start, _end) in enumerate(SCENES):
    inputs += ["-i", str(MEDIA / f"{L}_{i}.mp3")]
    ms = int((start + VO_OFFSET) * 1000)
    filters.append(f"[{i + 1}:a]aresample=48000,loudnorm=I=-15:TP=-1.5:LRA=7,adelay={ms}|{ms},apad[v{i}]")
    labels.append(f"[v{i}]")
filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,atrim=0:{DUR}[vo]")
filters.append("[vo]asplit=2[vo1][vo2]")
filters.append(f"[0:a]aresample=48000,volume=0.55,apad,atrim=0:{DUR},afade=t=out:st={DUR - 1.2}:d=1.2[mus]")
filters.append("[mus][vo1]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[ducked]")
filters.append("[ducked][vo2]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]")
cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(ROOT / f"audio_{L}.m4a")]
subprocess.run(cmd, check=True)
print("ok", L)
