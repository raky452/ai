# TechnoForesight Assistant ad (Telugu + English)

Motion-graphics renderer for the 58-second launch ad.

- `render.py` draws every frame (starfield, neon shapes, glowing mixed-script text, benefit chips, end card)
  and composites real app footage from `footage/<scene>/*.png`.
- `mix.py` places the voice-over lines on their scenes and ducks the music under the voice.

Assets (fonts, voice-overs, music) live on the `ad-assets` branch:
- Noto fonts for Indian scripts, SIL Open Font License
- Voice-overs and music generated with vidIQ (royalty-free)

```bash
git clone -b ad-assets https://github.com/raky452/ai assets
python3 mix.py te && python3 render.py te --part 0/1
```
