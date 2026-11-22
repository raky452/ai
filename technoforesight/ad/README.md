# TechnoForesight Assistant launch ad (Telugu + English)

58-second futuristic launch ad, rendered entirely in code. **Live:** YouTube [Telugu](https://youtu.be/6oqI3PJZu7c) · [English](https://youtu.be/US3ozZIHie0) · Instagram [Telugu Reel](https://www.instagram.com/reel/DgDUsQSgX8r/) · [English Reel](https://www.instagram.com/reel/DgDUuqwCl76/)

| File | What it does |
|---|---|
| `render.py` | Draws every frame at 1080p30: starfield, neon laptop, cloud and shield, glowing mixed-script text (Latin + Indian scripts via raqm shaping), benefit chips, the 12-language wall, ₹0 shatter and end card. Composites real app footage from `footage/<scene>/*.png` (falls back to `ui_idle.png`) |
| `mix.py` | Places each voice-over line on its scene and ducks the music under the voice (about −15 LUFS) |

## Scenes

`spark → words → shield → reveal → email → langs → offline → montage → connect → free → end`, timed in `SCENES` in `render.py`. On-screen text for both languages is in `TEXT`.

## Assets (`ad-assets` branch)

- `fonts/`: Noto Sans for Indian scripts (SIL Open Font License), fetched by `fetch-fonts.yml`
- `media/`: voice-over lines `te_0…10.mp3` (Bharadwaj) and `en_0…10.mp3` (Brian), plus `music.mp3`, all AI-generated with vidIQ (royalty-free), fetched by `fetch-media.yml`
- `renders/`: final MP4s (16:9 and 9:16 Reels)

## Render

```bash
git clone -b ad-assets https://github.com/raky452/ai assets
python3 mix.py te                                            # audio_te.m4a
python3 render.py te --part 0/2 & python3 render.py te --part 1/2 & wait
printf "file 'part_te_0.mp4'\nfile 'part_te_1.mp4'\n" > l.txt
ffmpeg -f concat -safe 0 -i l.txt -i audio_te.m4a -map 0:v -map 1:a -c copy -shortest final_te.mp4
python3 render.py te --preview 2,15,29,50,55                 # quick still frames
```

## Compliance rules followed

- Every claim is qualified: AI runs locally; offline after a one-time setup; free (app, Ollama and model).
- Fine print shows requirements, notes that web features use the internet and that quality varies by language, and says TechnoForesight is not affiliated with Microsoft, Ollama or Alibaba.
- No third-party logos; app-output scenes must be **genuine AI output**, never typed in by hand.
- AI voice and music are disclosed in descriptions; YouTube's "altered or synthetic content" box is set to **Yes**.
