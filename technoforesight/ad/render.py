"""TechnoForesight Assistant ad renderer: futuristic motion graphics + real app footage.

    python3 render.py te|en [--preview t1,t2,...] [--out file.mp4] [--scale 0.5]
"""
from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
FOOT = ROOT / "footage"
W, H, FPS = 1920, 1080, 30
DUR = 57.7

BG = (4, 6, 14)
CYAN, VIOLET, PINK = (34, 211, 238), (139, 92, 246), (236, 72, 153)
WHITE, MUTED, GREEN = (240, 246, 255), (140, 155, 190), (52, 211, 153)

F_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
F_MED = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
F_LIGHT = "/usr/share/fonts/opentype/inter/Inter-Light.otf"
F_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
NOTO = ASSETS / "fonts"
SCRIPT_FONT = {  # unicode block start -> font stem
    0x0C00: "NotoSansTelugu", 0x0900: "NotoSansDevanagari", 0x0B80: "NotoSansTamil",
    0x0C80: "NotoSansKannada", 0x0D00: "NotoSansMalayalam", 0x0980: "NotoSansBengali",
    0x0A80: "NotoSansGujarati", 0x0A00: "NotoSansGurmukhi", 0x0B00: "NotoSansOriya",
    0x0600: "NotoNaskhArabic",
}

# ---------------------------------------------------------------- timeline
SCENES = [  # name, start, end
    ("spark", 0.0, 4.5), ("words", 4.5, 9.5), ("shield", 9.5, 13.5), ("reveal", 13.5, 17.5),
    ("email", 17.5, 26.0), ("langs", 26.0, 33.0), ("offline", 33.0, 36.5), ("montage", 36.5, 42.5),
    ("connect", 42.5, 46.5), ("free", 46.5, 52.0), ("end", 52.0, DUR),
]
VO_OFFSET = 0.3

TEXT = {
    "en": {
        "words": "Where do your words go?",
        "shield": "Nowhere. They stay with you.",
        "reveal": "Alt + Space. Anywhere.",
        "email": ("PRIVATE", "Type in English. Get it in Telugu. Runs on your PC."),
        "langs": "12 languages. One assistant.",
        "offline": ("WORKS OFFLINE", "after a one-time setup"),
        "montage": ("GETS THINGS DONE", "apps · files · PDFs · settings"),
        "connect": ("CONNECTS", "your websites, with one word"),
        "free": ("Free", "Runs on your PC  ·  Your data stays with you"),
        "tag": "Your AI. Your PC. Your language.",
        "tag2": "మీ AI. మీ PC. మీ భాష.",
        "download": "Download free:",
        "steps": ["Download", "Install Ollama (free)", "Press Alt + Space"],
        "fine": ("Windows 10/11 (64-bit), 8 GB RAM recommended. Needs the free Ollama engine and a one-time ~2.5 GB model "
                 "download; works offline after that. Web search and links use the internet. Answer quality varies by "
                 "language and PC speed. Uses open-source Qwen models (Apache 2.0). Not affiliated with Microsoft, Ollama or Alibaba."),
    },
    "te": {
        "words": "మీ మాటలు ఎక్కడికి వెళ్తున్నాయి?",
        "shield": "ఎక్కడికీ కాదు. మీ దగ్గరే.",
        "reveal": "Alt + Space. ఎక్కడైనా.",
        "email": ("ప్రైవేట్", "English లో అడగండి, తెలుగులో పొందండి"),
        "langs": "12 భాషలు. ఒకే అసిస్టెంట్.",
        "offline": ("ఇంటర్నెట్ లేకపోయినా పని చేస్తుంది", "ఒక్కసారి సెటప్ తర్వాత"),
        "montage": ("యాప్స్ · ఫైల్స్ · PDF లు · సెట్టింగ్స్", ""),
        "connect": ("మీ వెబ్‌సైట్లు", "ఒక్క మాటతో"),
        "free": ("పూర్తిగా ఉచితం", "మీ PC లోనే  ·  మీ డేటా మీ దగ్గరే"),
        "tag": "మీ AI. మీ PC. మీ భాష.",
        "tag2": "Your AI. Your PC. Your language.",
        "download": "ఉచితంగా డౌన్‌లోడ్ చేసుకోండి:",
        "steps": ["డౌన్‌లోడ్", "Ollama ఇన్‌స్టాల్ (ఉచితం)", "Alt + Space నొక్కండి"],
        "fine": ("Windows 10/11 (64-bit), 8 GB RAM సిఫార్సు. ఉచిత Ollama ఇంజిన్, ఒక్కసారి ~2.5 GB మోడల్ డౌన్‌లోడ్ అవసరం; "
                 "తర్వాత ఆఫ్‌లైన్‌లో పని చేస్తుంది. వెబ్ సెర్చ్, లింకులకు ఇంటర్నెట్ అవసరం. భాష, PC వేగాన్ని బట్టి జవాబుల నాణ్యత మారవచ్చు. "
                 "ఓపెన్-సోర్స్ Qwen మోడల్స్ (Apache 2.0). Microsoft, Ollama, Alibaba లతో ఎలాంటి సంబంధం లేదు."),
    },
}
LANG_WALL = ["తెలుగు", "हिन्दी", "தமிழ்", "ಕನ್ನಡ", "മലയാളം", "मराठी", "বাংলা", "ગુજરાતી", "ਪੰਜਾਬੀ", "ଓଡ଼ିଆ", "اردو", "English"]


# ---------------------------------------------------------------- helpers
def ease(x):  # smooth in-out
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def grad(t):
    t = t % 1.0
    return mix(CYAN, VIOLET, t * 2) if t < 0.5 else mix(VIOLET, PINK, (t - 0.5) * 2)


@lru_cache(maxsize=64)
def font(path, size):
    return ImageFont.truetype(str(path), size)


def script_font(ch, bold):
    cp = ord(ch)
    if ch == "‌" or ch == "‍":
        return None
    for start, stem in SCRIPT_FONT.items():
        if start <= cp < start + 0x80 or (stem == "NotoNaskhArabic" and 0x0600 <= cp < 0x0700):
            return NOTO / f"{stem}-{'Bold' if bold else 'Regular'}.ttf"
    return "latin"


def runs(text, bold):
    """Split text into (substring, fontpath) runs by script."""
    out, cur, cur_f = [], "", None
    for ch in text:
        f = script_font(ch, bold)
        if f is None or (ch == " " and cur_f is not None):
            cur += ch
            continue
        if cur_f is None:
            cur_f = f
        if f != cur_f:
            out.append((cur, cur_f))
            cur, cur_f = "", f
        cur += ch
    if cur:
        out.append((cur, cur_f or "latin"))
    return out


@lru_cache(maxsize=256)
def text_img(text, size, color=WHITE, weight="bold", glow=0, glow_color=None):
    """RGBA image of text (mixed scripts), optional neon glow. Cached."""
    latin = {"black": F_BLACK, "bold": F_BOLD, "med": F_MED, "light": F_LIGHT, "mono": F_MONO}[weight]
    bold = weight in ("black", "bold", "mono")
    parts = []
    for s, f in runs(text, bold):
        fp = latin if f == "latin" else f
        sz = int(size * (0.92 if f != "latin" else 1.0))
        fo = font(fp, sz)
        bb = fo.getbbox(s, language=None)
        parts.append((s, fo, bb))
    asc = int(size * 1.25)
    width = sum(max(1, p[2][2]) for p in parts) + 4
    pad = glow * 3 + 8
    img = Image.new("RGBA", (width + 2 * pad, asc + int(size * 0.5) + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad
    for s, fo, bb in parts:
        d.text((x, pad + asc - int(size * 1.0)), s, font=fo, fill=color + (255,))
        x += max(1, bb[2])
    img = img.crop(img.getbbox() or (0, 0, 1, 1))
    if glow:
        gpad = glow * 7
        base = Image.new("RGBA", (img.width + 2 * gpad, img.height + 2 * gpad), (0, 0, 0, 0))
        base.paste(img, (gpad, gpad), img)
        gc = glow_color or color
        tint = Image.new("RGBA", base.size, gc + (0,))
        tint.putalpha(base.getchannel("A"))
        g1 = tint.filter(ImageFilter.GaussianBlur(glow))
        g2 = tint.filter(ImageFilter.GaussianBlur(glow * 2.5))
        out = Image.alpha_composite(g2, g1)
        out = Image.alpha_composite(out, base)
        return out
    return img


def paste_center(frame, img, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.01:
        return
    if abs(scale - 1.0) > 0.01:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    if alpha < 0.999:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    frame.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))


def wrap(text, size, weight, max_w):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_img(trial, size, MUTED, weight).width > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- background
rng = random.Random(7)
STARS = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.15, 1.0)) for _ in range(380)]


@lru_cache(maxsize=1)
def vignette():
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
    v = np.clip(1.15 - r * 0.75, 0.25, 1.0)
    return v[..., None]


@lru_cache(maxsize=1)
def nebula():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    for (cx, cy, rad, col, k) in [(0.25, 0.3, 0.45, VIOLET, 0.10), (0.78, 0.7, 0.5, CYAN, 0.07), (0.6, 0.15, 0.3, PINK, 0.05)]:
        d = np.exp(-(((x / W - cx) ** 2 + (y / H - cy) ** 2) / (rad ** 2)) * 3)
        img += d[..., None] * np.array(col, np.float32) * k
    return img


def background(t, warp=0.0):
    base = np.full((H, W, 3), BG, np.float32) + nebula()
    frame = Image.fromarray(np.clip(base * vignette(), 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(frame)
    speed = 0.04 + warp
    for sx, sy, z in STARS:
        zz = (z - t * speed) % 1.0 + 0.02
        px, py = W / 2 + sx / zz * W * 0.18, H / 2 + sy / zz * H * 0.18
        if not (0 <= px < W and 0 <= py < H):
            continue
        b = int(min(255, (1.1 - zz) * 230))
        r = max(0.6, (1.1 - zz) * 2.2)
        col = (b, b, min(255, b + 25), 255)
        if warp > 0.05:
            tail = warp * 60 * (1.1 - zz)
            ang = math.atan2(py - H / 2, px - W / 2)
            d.line([(px, py), (px - math.cos(ang) * tail, py - math.sin(ang) * tail)], fill=col, width=max(1, int(r)))
        else:
            d.ellipse([px - r, py - r, px + r, py + r], fill=col)
    return frame


# ---------------------------------------------------------------- shapes
def glow_layer(draw_fn, blur=10, strength=1.6):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    g = layer.filter(ImageFilter.GaussianBlur(blur))
    out = Image.alpha_composite(g, layer)
    if strength > 1:
        out = Image.alpha_composite(out, g)
    return out


def diamond(d, cx, cy, r, t=0.0, width=0):
    for i in range(16, 0, -1):
        k = i / 16
        rr = r * k
        col = grad(1 - k + t * 0.2) + (255,)
        d.polygon([(cx, cy - rr), (cx + rr, cy), (cx, cy + rr), (cx - rr, cy)], fill=col)


def laptop(d, cx, cy, s, col, width=4):
    w, h = 300 * s, 190 * s
    d.rounded_rectangle([cx - w / 2, cy - h, cx + w / 2, cy], radius=int(14 * s), outline=col, width=width)
    d.rounded_rectangle([cx - w / 2 + 14 * s, cy - h + 14 * s, cx + w / 2 - 14 * s, cy - 12 * s], radius=int(6 * s),
                        outline=mix(col[:3], BG, 0.5) + (255,), width=max(1, width // 2))
    d.polygon([(cx - w / 2 - 40 * s, cy + 22 * s), (cx + w / 2 + 40 * s, cy + 22 * s), (cx + w / 2, cy), (cx - w / 2, cy)],
              outline=col, width=width)


def cloud(d, cx, cy, s, col, width=4):
    for (ox, oy, r) in [(-90, 10, 55), (-30, -30, 75), (50, -15, 65), (100, 20, 45)]:
        d.arc([cx + (ox - r) * s, cy + (oy - r) * s, cx + (ox + r) * s, cy + (oy + r) * s], 160, 380, fill=col, width=width)
    d.line([(cx - 140 * s, cy + 60 * s), (cx + 140 * s, cy + 60 * s)], fill=col, width=width)


def shield(d, cx, cy, r, col, width=5, prog=1.0):
    pts = []
    for i in range(7):
        a = -math.pi / 2 + i * 2 * math.pi / 6
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r * 1.05))
    total = int(6 * prog * 100)
    for i in range(6):
        seg = min(100, max(0, total - i * 100)) / 100
        if seg <= 0:
            break
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        d.line([(x1, y1), (x1 + (x2 - x1) * seg, y1 + (y2 - y1) * seg)], fill=col, width=width)


def bezier(p0, p1, p2, t):
    return ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])


def chip(frame, title, sub, lt, y=930, accent=CYAN):
    """Lower-third benefit chip that slides in."""
    a = ease(lt / 0.5)
    if a <= 0:
        return
    ti = text_img(title, 46, WHITE, "black", glow=6, glow_color=accent)
    si = text_img(sub, 30, MUTED, "med") if sub else None
    w = max(ti.width, si.width if si else 0) + 120
    h = 150 if si else 100
    x0 = W / 2 - w / 2 + (1 - a) * 80
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([x0, y - h / 2, x0 + w, y + h / 2], radius=26, fill=(12, 18, 36, int(215 * a)),
                        outline=accent + (int(200 * a),), width=2)
    for i in range(60):  # gradient top edge
        xx = x0 + 26 + (w - 52) * i / 60
        d.line([(xx, y - h / 2 + 1), (xx + (w - 52) / 60 + 1, y - h / 2 + 1)], fill=grad(i / 60) + (int(255 * a),), width=3)
    frame.alpha_composite(layer)
    paste_center(frame, ti, x0 + w / 2, y - (22 if si else 0), a)
    if si:
        paste_center(frame, si, x0 + w / 2, y + 40, a)


def big_title(frame, text, lt, y=H / 2, size=86, dur=10.0, color=WHITE, glow_color=CYAN):
    a = ease(lt / 0.6) * (1 - ease((lt - (dur - 0.4)) / 0.4))
    img = text_img(text, size, color, "black", glow=10, glow_color=glow_color)
    sc = 0.92 + 0.08 * ease_out(lt / 0.8) + lt * 0.006
    paste_center(frame, img, W / 2, y, a, sc)


# ---------------------------------------------------------------- footage
@lru_cache(maxsize=8)
def clip_frames(name):
    d = FOOT / name
    return sorted(d.glob("*.png")) if d.exists() else []


@lru_cache(maxsize=4)
def placeholder():
    p = ROOT / "ui_idle.png"
    return Image.open(p).convert("RGBA") if p.exists() else Image.new("RGBA", (780, 244), (10, 15, 30, 255))


def footage_frame(name, lt, speed=1.0):
    frames = clip_frames(name)
    if not frames:
        return placeholder()
    i = min(len(frames) - 1, int(lt * FPS * speed))
    return Image.open(frames[i]).convert("RGBA")


def screen_panel(frame, img, lt, cx=W / 2, cy=H * 0.46, max_w=1240, max_h=640, intro=0.5):
    """Footage in a floating glass panel with neon edge and reflection glow."""
    a = ease(lt / intro)
    sc = min(max_w / img.width, max_h / img.height) * (0.94 + 0.06 * ease_out(lt / 0.8))
    im = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
    x0, y0 = int(cx - im.width / 2), int(cy - im.height / 2 + (1 - a) * 40)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.rounded_rectangle([x0 - 6, y0 - 6, x0 + im.width + 6, y0 + im.height + 6], radius=26,
                         outline=VIOLET + (int(170 * a),), width=10)
    glow = glow.filter(ImageFilter.GaussianBlur(22))
    frame.alpha_composite(glow)
    if a < 0.999:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
    frame.alpha_composite(im, (x0, y0))


# ---------------------------------------------------------------- scenes
def s_spark(t, lt, dur, L):
    fr = background(t)
    cx, cy = W / 2, H / 2 - 30
    conv = ease(lt / 3.0)

    def dr(d):
        rr = random.Random(3)
        for i in range(90):
            a0 = rr.uniform(0, 2 * math.pi)
            r0 = rr.uniform(300, 900)
            ang = a0 + (1 - conv) * 3.5
            r = r0 * (1 - conv) + 6
            px, py = cx + math.cos(ang) * r, cy + math.sin(ang) * r * 0.6
            s = rr.uniform(1.5, 4) * (0.5 + conv)
            d.ellipse([px - s, py - s, px + s, py + s], fill=grad(i / 90) + (int(255 * min(1, lt / 0.6)),))
        k = ease((lt - 2.6) / 1.0)
        if k > 0:
            diamond(d, cx, cy, 120 * ease_out(k), t)
            d.polygon([(cx, cy - 40 * k), (cx + 40 * k, cy), (cx, cy + 40 * k), (cx - 40 * k, cy)], fill=BG + (255,))
        if lt < 0.5:
            r = 6 + lt * 30
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CYAN + (255,))
    fr.alpha_composite(glow_layer(dr, 14))
    if lt > 3.2:
        paste_center(fr, text_img("TechnoForesight", 64, WHITE, "black", glow=8, glow_color=VIOLET),
                     W / 2, cy + 200, ease((lt - 3.2) / 0.6))
    return fr


LAP = (W * 0.24, H * 0.72)
CLD = (W * 0.76, H * 0.32)


def s_words(t, lt, dur, L):
    fr = background(t)

    def dr(d):
        laptop(d, *LAP, 1.0, CYAN + (255,))
        cloud(d, *CLD, 1.3, (120, 130, 170, 255), 4)
        rr = random.Random(11)
        for i in range(40):
            ph = (lt * 0.35 + i / 40) % 1.0
            p0 = (LAP[0] + rr.uniform(-60, 60), LAP[1] - 110)
            p2 = (CLD[0] + rr.uniform(-80, 80), CLD[1] + 30)
            p1 = (W / 2 + rr.uniform(-200, 200), H * 0.18 + rr.uniform(-80, 80))
            x, y = bezier(p0, p1, p2, ph)
            x2, y2 = bezier(p0, p1, p2, max(0, ph - 0.05))
            col = grad(i / 40) + (int(255 * math.sin(ph * math.pi)),)
            d.line([(x2, y2), (x, y)], fill=col, width=4)
    fr.alpha_composite(glow_layer(dr, 9))
    big_title(fr, TEXT[L]["words"], lt, y=H * 0.5, size=80, dur=dur)
    return fr


def s_shield(t, lt, dur, L):
    fr = background(t)
    mv = ease(lt / 1.4)
    cx, cy = LAP[0] + (W / 2 - LAP[0]) * mv, LAP[1] + (H * 0.6 - LAP[1]) * mv

    def dr(d):
        laptop(d, cx, cy, 1.0 + 0.2 * mv, CYAN + (255,))
        rr = random.Random(11)
        back = ease(lt / 1.2)
        for i in range(40):
            ph = 1 - back - i / 160
            if ph <= 0.01:
                continue
            p0 = (cx + rr.uniform(-60, 60), cy - 110)
            p2 = (CLD[0] + rr.uniform(-80, 80), CLD[1] + 30)
            p1 = (W / 2 + rr.uniform(-200, 200), H * 0.18)
            x, y = bezier(p0, p1, p2, ph)
            x2, y2 = bezier(p0, p1, p2, min(1, ph + 0.06))
            d.line([(x2, y2), (x, y)], fill=grad(i / 40) + (220,), width=4)
        sp = ease((lt - 1.2) / 1.2)
        if sp > 0:
            shield(d, cx, cy - 100, 300, CYAN + (255,), 6, sp)
            shield(d, cx, cy - 100, 325, VIOLET + (int(150 * sp),), 3, sp)
    fr.alpha_composite(glow_layer(dr, 12))
    if 2.2 < lt < 2.6:  # flash
        k = 1 - abs(lt - 2.4) / 0.2
        fr.alpha_composite(Image.new("RGBA", (W, H), (180, 240, 255, int(70 * k))))
    big_title(fr, TEXT[L]["shield"], lt - 1.6, y=H * 0.2, size=76, dur=dur - 1.6)
    return fr


def keycap(fr, label, cx, cy, w, pressed):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    dy = 8 * pressed
    col = mix((40, 50, 80), CYAN, pressed)
    d.rounded_rectangle([cx - w / 2, cy - 55 + dy, cx + w / 2, cy + 55 + dy], radius=18, fill=(14, 20, 40, 255),
                        outline=col + (255,), width=4)
    g = layer.filter(ImageFilter.GaussianBlur(14 * (0.3 + pressed)))
    fr.alpha_composite(g)
    fr.alpha_composite(layer)
    paste_center(fr, text_img(label, 40, WHITE, "bold"), cx, cy + dy)


def s_reveal(t, lt, dur, L):
    fr = background(t, warp=0.5 * ease(1 - abs(lt - 1.0) / 1.0) if lt < 2 else 0)
    if lt < 1.3:
        p = ease((lt - 0.4) / 0.25)
        keycap(fr, "Alt", W / 2 - 260, H / 2, 200, p)
        paste_center(fr, text_img("+", 60, MUTED, "bold"), W / 2 - 110, H / 2)
        keycap(fr, "Space", W / 2 + 120, H / 2, 360, ease((lt - 0.55) / 0.25))
    if 1.0 < lt < 1.4:
        k = 1 - abs(lt - 1.2) / 0.2
        fr.alpha_composite(Image.new("RGBA", (W, H), (200, 245, 255, int(120 * k))))
    if lt >= 1.2:
        screen_panel(fr, footage_frame("reveal", lt - 1.2), lt - 1.2, cy=H * 0.44, max_w=1300, max_h=430)
        big_title(fr, TEXT[L]["reveal"], lt - 1.5, y=H * 0.83, size=70, dur=dur - 1.5)
    return fr


def s_email(t, lt, dur, L):
    fr = background(t)
    screen_panel(fr, footage_frame("email", lt), lt, cy=H * 0.43, max_w=1180, max_h=720)
    title, sub = TEXT[L]["email"]
    chip(fr, title, sub, lt - 1.2, y=H - 110)
    return fr


def s_langs(t, lt, dur, L):
    fr = background(t, warp=0.15)
    rr = random.Random(5)
    n = len(LANG_WALL)
    for i, word in enumerate(LANG_WALL):
        start = i * 0.32
        k = ease_out((lt - start) / 1.1)
        if k <= 0:
            continue
        col, row = i % 4, i // 4
        tx, ty = W * (0.17 + col * 0.22), H * (0.25 + row * 0.2)
        sx, sy = W / 2 + rr.uniform(-80, 80), H / 2 + rr.uniform(-60, 60)
        x, y = sx + (tx - sx) * k, sy + (ty - sy) * k
        img = text_img(word, 78, WHITE, "bold", glow=9, glow_color=grad(i / n))
        drift = math.sin(t * 1.3 + i) * 6
        out = 1 - ease((lt - 5.2) / 0.6)
        paste_center(fr, img, x, y + drift, min(1, k * 1.4) * out, 0.4 + 0.6 * k)
    if lt > 5.0:
        big_title(fr, TEXT[L]["langs"], lt - 5.0, y=H / 2, size=96, dur=dur - 5.0, glow_color=VIOLET)
    return fr


def wifi(d, cx, cy, s, col, on):
    for i, r in enumerate((150, 105, 60)):
        c = col if on > i / 3 else (60, 66, 90, 255)
        d.arc([cx - r * s, cy - r * s, cx + r * s, cy + r * s], 225, 315, fill=c, width=int(14 * s))
    d.ellipse([cx - 14 * s, cy - 14 * s, cx + 14 * s, cy + 14 * s], fill=col if on > 0 else (60, 66, 90, 255))


def s_offline(t, lt, dur, L):
    fr = background(t)
    screen_panel(fr, footage_frame("offline", lt), lt, cx=W * 0.6, cy=H * 0.45, max_w=1000, max_h=560)
    on = 1 - ease((lt - 0.4) / 0.9)

    def dr(d):
        wifi(d, W * 0.17, H * 0.5, 1.0, (CYAN if on > 0.3 else (255, 120, 140)) + (255,), on)
        if on < 0.5:
            k = ease((0.5 - on) / 0.3)
            d.line([(W * 0.17 - 150 * k, H * 0.5 - 150 * k), (W * 0.17 + 150 * k, H * 0.5 + 40 * k)], fill=(255, 110, 140, 255), width=12)
    fr.alpha_composite(glow_layer(dr, 10))
    title, sub = TEXT[L]["offline"]
    chip(fr, title, sub, lt - 0.8, y=H - 110, accent=GREEN)
    return fr


def s_montage(t, lt, dur, L):
    fr = background(t)
    clips = ["summary", "settings", "search"]
    seg = dur / len(clips)
    i = min(len(clips) - 1, int(lt / seg))
    ll = lt - i * seg
    screen_panel(fr, footage_frame(clips[i], ll), ll, cy=H * 0.43, max_w=1180, max_h=700, intro=0.25)
    title, sub = TEXT[L]["montage"]
    chip(fr, title, sub, lt - 0.3, y=H - 110, accent=VIOLET)
    return fr


def s_connect(t, lt, dur, L):
    fr = background(t, warp=0.8 * ease((lt - 1.6) / 0.6) * (1 - ease((lt - 2.6) / 0.8)))
    screen_panel(fr, footage_frame("connect", lt), lt, cy=H * 0.42, max_w=1100, max_h=420)
    k = ease((lt - 1.6) / 0.5) * (1 - ease((lt - 2.8) / 0.6))
    if k > 0:
        def dr(d):
            for i in range(28):
                a = i / 28 * 2 * math.pi
                r1, r2 = 120 + 500 * k, 200 + 900 * k
                d.line([(W / 2 + math.cos(a) * r1, H * 0.42 + math.sin(a) * r1 * 0.6),
                        (W / 2 + math.cos(a) * r2, H * 0.42 + math.sin(a) * r2 * 0.6)], fill=grad(i / 28) + (int(255 * k),), width=4)
        fr.alpha_composite(glow_layer(dr, 8))
    title, sub = TEXT[L]["connect"]
    chip(fr, title, sub, lt - 0.6, y=H - 110, accent=PINK)
    return fr


def s_free(t, lt, dur, L):
    fr = background(t)
    cx, cy = W / 2, H * 0.4
    shatter = ease((lt - 1.0) / 1.2)
    if lt < 1.0:
        k = ease(lt / 0.5)
        paste_center(fr, text_img("₹ 999", 170, WHITE, "black", glow=10, glow_color=PINK), cx, cy, k)
    else:
        def dr(d):
            rr = random.Random(9)
            for i in range(160):
                a = rr.uniform(0, 2 * math.pi)
                sp = rr.uniform(150, 650)
                x = cx + rr.uniform(-200, 200) + math.cos(a) * sp * shatter
                y = cy + rr.uniform(-60, 60) + math.sin(a) * sp * shatter * 0.7 + 200 * shatter ** 2
                s = rr.uniform(2, 6) * (1 - shatter * 0.7)
                d.rectangle([x - s, y - s, x + s, y + s], fill=grad(i / 160) + (int(255 * (1 - shatter)),))
        fr.alpha_composite(glow_layer(dr, 6))
        k = ease((lt - 1.5) / 0.6)
        paste_center(fr, text_img("₹0", 230, WHITE, "black", glow=16, glow_color=CYAN), cx, cy, k, 0.7 + 0.3 * ease_out(k))
    title, sub = TEXT[L]["free"]
    if lt > 2.0:
        a = ease((lt - 2.0) / 0.5)
        paste_center(fr, text_img(title, 70, WHITE, "black", glow=8, glow_color=VIOLET), cx, H * 0.66, a)
        paste_center(fr, text_img(sub, 40, MUTED, "med"), cx, H * 0.76, ease((lt - 2.4) / 0.5))
    return fr


def s_end(t, lt, dur, L):
    fr = background(t)
    tx = TEXT[L]

    def dr(d):
        diamond(d, W / 2, H * 0.16, 62 * ease_out(lt / 0.6), t)
        d.polygon([(W / 2, H * 0.16 - 20), (W / 2 + 20, H * 0.16), (W / 2, H * 0.16 + 20), (W / 2 - 20, H * 0.16)], fill=BG + (255,))
    fr.alpha_composite(glow_layer(dr, 12))
    a = ease(lt / 0.6)
    paste_center(fr, text_img("TechnoForesight Assistant", 60, WHITE, "black"), W / 2, H * 0.29, a)
    paste_center(fr, text_img(tx["tag"], 76, WHITE, "black", glow=10, glow_color=CYAN), W / 2, H * 0.41, ease((lt - 0.3) / 0.6))
    paste_center(fr, text_img(tx["tag2"], 38, MUTED, "med"), W / 2, H * 0.49, ease((lt - 0.6) / 0.6))
    paste_center(fr, text_img(tx["download"] + "  technoforesight.in/assistant", 46, CYAN, "bold", glow=6),
                 W / 2, H * 0.6, ease((lt - 0.9) / 0.5))
    xs = [W * 0.22, W * 0.47, W * 0.77]
    for i, (x, s) in enumerate(zip(xs, tx["steps"])):
        k = ease((lt - 1.3 - i * 0.35) / 0.4)
        if k <= 0:
            continue
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse([x - 230 - 30, H * 0.72 - 30, x - 230 + 30, H * 0.72 + 30], outline=grad(i / 2) + (int(255 * k),), width=4)
        fr.alpha_composite(layer.filter(ImageFilter.GaussianBlur(6)))
        fr.alpha_composite(layer)
        paste_center(fr, text_img(str(i + 1), 34, WHITE, "black"), x - 230, H * 0.72, k)
        img = text_img(s, 36, WHITE, "bold")
        paste_center(fr, img, x - 180 + img.width / 2, H * 0.72, k)
    fa = ease((lt - 1.8) / 0.5)
    for j, line in enumerate(wrap(tx["fine"], 21, "light", W - 260)):
        paste_center(fr, text_img(line, 21, (120, 132, 165), "light"), W / 2, H * 0.84 + j * 30, fa)
    return fr


SCENE_FN = {"spark": s_spark, "words": s_words, "shield": s_shield, "reveal": s_reveal, "email": s_email,
            "langs": s_langs, "offline": s_offline, "montage": s_montage, "connect": s_connect, "free": s_free,
            "end": s_end}


def frame_at(t, L):
    for name, a, b in SCENES:
        if a <= t < b or (name == "end" and t >= a):
            fr = SCENE_FN[name](t, t - a, b - a, L)
            # cross-fade from black at scene boundaries
            edge = min(t - a, b - t)
            if edge < 0.18 and t > 0.2:
                k = edge / 0.18
                fr = Image.blend(Image.new("RGBA", (W, H), BG + (255,)), fr, 0.35 + 0.65 * k)
            if t < 0.3:
                fr = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), fr, t / 0.3)
            return fr.convert("RGB")
    return Image.new("RGB", (W, H), BG)


def render_range(L, f0, f1, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(frame_at(f / FPS, L).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    L = sys.argv[1]
    if "--preview" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]
        os.makedirs(ROOT / "preview", exist_ok=True)
        for tt in ts:
            frame_at(tt, L).save(ROOT / "preview" / f"{L}_{tt:05.1f}.jpg", quality=88)
        print("ok")
    elif "--part" in sys.argv:
        i, n = map(int, sys.argv[sys.argv.index("--part") + 1].split("/"))
        total = int(DUR * FPS)
        f0, f1 = total * i // n, total * (i + 1) // n
        render_range(L, f0, f1, str(ROOT / f"part_{L}_{i}.mp4"))
