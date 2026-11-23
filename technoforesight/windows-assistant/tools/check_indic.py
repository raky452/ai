"""Checks that an answer file is clean Indian-language text (used by the end-to-end tests).

    python tools/check_indic.py answer.txt [telugu]

Passes when most words use the language's script (English technical terms are allowed), no
characters from unrelated scripts slipped in (Vietnamese, Chinese, Cyrillic) and no Markdown remains.
"""
import sys

RANGES = {"telugu": ("ఀ", "౿"), "hindi": ("ऀ", "ॿ"), "tamil": ("஀", "௿")}
FOREIGN = (("Ḁ", "ỿ"), ("一", "鿿"), ("Ѐ", "ӿ"), ("぀", "ヿ"))


def check(text: str, lang: str = "telugu") -> tuple[bool, str]:
    lo, hi = RANGES[lang]
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    native = sum(any(lo <= c <= hi for c in w) for w in words)
    foreign = [c for c in text if any(a <= c <= b for a, b in FOREIGN)]
    share = native / max(1, len(words))
    ok = bool(words) and share > 0.5 and not foreign and "**" not in text
    return ok, f"{lang} words {share:.0%}, foreign characters {len(foreign)}, markdown {'**' in text}"


if __name__ == "__main__":
    ok, info = check(open(sys.argv[1], encoding="utf-8").read(), sys.argv[2] if len(sys.argv) > 2 else "telugu")
    print(info)
    sys.exit(0 if ok else 1)
