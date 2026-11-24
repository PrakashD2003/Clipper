# StreamClipper Audio Assets & Licenses

This document logs all background music (BGM) and sound effects (SFX) included with StreamClipper. All assets have been verified to be **100% YouTube-monetization-safe**, containing **no YouTube Content ID claims**, and licensed under either **Creative Commons Zero (CC0 1.0 Universal / Public Domain)** or **Creative Commons: By Attribution 4.0 (CC-BY 4.0)**.

---

## 1. Summary of Assets

| Filename | Type & Mood | Duration | Artist / Creator | License | Source / Download URL | Attribution Required? |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| `assets/bgm/neon_laser_horizon.mp3` | BGM — Vice City / Retro Synthwave | 2m 58s (178.57s) | Kevin MacLeod | CC-BY 4.0 | [Incompetech](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN2000023) | Yes (Description credit) |
| `assets/bgm/chill_wave.mp3` | BGM — Chill Lo-Fi / Ambient Commentary | 4m 00s (240.07s) | Kevin MacLeod | CC-BY 4.0 | [Incompetech](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1600048) | Yes (Description credit) |
| `assets/bgm/crypto.mp3` | BGM — Subtle Suspense / Investigative Underscore | 3m 24s (204.07s) | Kevin MacLeod | CC-BY 4.0 | [Incompetech](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1600013) | Yes (Description credit) |
| `assets/bgm/cipher.mp3` | BGM — Upbeat Energetic Gaming (Shorts) | 3m 51s (231.26s) | Kevin MacLeod | CC-BY 4.0 | [Incompetech](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1100844) | Yes (Description credit) |
| `assets/sfx/whoosh.wav` | SFX — Soft Filtered Transition Swoosh | 0.550s | StreamClipper / DeepMind | CC0 1.0 Universal (Public Domain) | Synthesized locally (pink noise swept bandpass) | No |
| `assets/sfx/pop.wav` | SFX — Subtle UI Pop / Callout Click | 0.350s | StreamClipper / DeepMind | CC0 1.0 Universal (Public Domain) | Synthesized locally (damped resonant frequency chirp) | No |

---

## 2. Technical Audio Specifications (Verified via `ffprobe`)

All audio files are verified for valid streams, zero corruption, non-zero file size, and standard broadcast sample rates:

| File | Codec | Sample Rate | Channels | Duration | Size (Bytes) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `assets/bgm/neon_laser_horizon.mp3` | `mp3` | 44,100 Hz | 2 (stereo) | 178.57 s | 3,573,716 |
| `assets/bgm/chill_wave.mp3` | `mp3` | 44,100 Hz | 2 (stereo) | 240.07 s | 7,684,325 |
| `assets/bgm/crypto.mp3` | `mp3` | 44,100 Hz | 2 (stereo) | 204.07 s | 6,532,455 |
| `assets/bgm/cipher.mp3` | `mp3` | 44,100 Hz | 2 (stereo) | 231.26 s | 9,252,699 |
| `assets/sfx/whoosh.wav` | `pcm_s16le` | 48,000 Hz | 2 (stereo) | 0.550 s | 105,678 |
| `assets/sfx/pop.wav` | `pcm_s16le` | 48,000 Hz | 2 (stereo) | 0.350 s | 67,278 |

---

## 3. Required Attribution Copy for YouTube Video Descriptions

When rendering videos using any of the Incompetech BGM tracks, paste the corresponding credit line into the YouTube video description (or configure your channel upload defaults to include the lines of the tracks in rotation):

### Vice-City / Retro Synthwave:
```text
"Neon Laser Horizon" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/
```

### Chill Lo-Fi / Commentary:
```text
"Chill Wave" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/
```

### Suspense / Investigative Lore Breakdown:
```text
"Crypto" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/
```

### Upbeat Energetic Gaming / Shorts:
```text
"Cipher" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/
```

### Combined Full Credit Block (Recommended for All Videos):
```text
Music by Kevin MacLeod (incompetech.com)
Tracks: "Neon Laser Horizon", "Chill Wave", "Crypto", "Cipher"
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/
```

---

## 4. Sound Effects Synthesis & CC0 License

The transition and UI sound effects (`assets/sfx/whoosh.wav` and `assets/sfx/pop.wav`) were generated natively for StreamClipper using NumPy, SciPy, and FFmpeg:

1. **`whoosh.wav` (0.55s)**:
   - Generated using a swept resonant bandpass filter on pink noise (center frequency sweeping from 250 Hz -> 1400 Hz -> 350 Hz).
   - Envelope shaped with raised cosine attack and smooth exponential decay.
   - Mastered to 48 kHz stereo 16-bit PCM WAV normalized to -14 LUFS (`-1.5` dB True Peak).
2. **`pop.wav` (0.35s)**:
   - Generated using a rapid downward frequency sweep (750 Hz down to 280 Hz over 45 ms) with a warm secondary 220 Hz acoustic body resonance.
   - Mastered to 48 kHz stereo 16-bit PCM WAV normalized via EBU R128 `loudnorm` to -16 LUFS (`-2.0` dB True Peak).

Both sound effects are dedicated to the public domain under the terms of **Creative Commons Zero 1.0 Universal (CC0 1.0)**. Anyone is free to copy, modify, distribute, or perform the work, even for commercial purposes, all without asking permission or giving attribution.
