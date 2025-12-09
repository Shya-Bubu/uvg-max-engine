# SFX Assets

Sound effects library for automated video enhancement.

## Required Files

| File | Format | Purpose | Duration |
|------|--------|---------|----------|
| whoosh_fast.wav | WAV/44.1kHz | Fast transition | 0.3-0.5s |
| whoosh_slow.wav | WAV/44.1kHz | Slow transition | 0.5-1.0s |
| impact_deep.wav | WAV/44.1kHz | Climax hit | 0.5-1.0s |
| tension_riser.wav | WAV/44.1kHz | Building tension | 2-4s |
| bright_ding.wav | WAV/44.1kHz | Happy moment | 0.3-0.5s |
| pop_soft.wav | WAV/44.1kHz | Text appear | 0.1-0.3s |
| fade_soft.wav | WAV/44.1kHz | Fade transition | 0.5-1.0s |
| click_soft.wav | WAV/44.1kHz | UI click | 0.1s |

## Freesound API

Set `FREESOUND_API_KEY` in .env to auto-download missing SFX.

Get key: https://freesound.org/apiv2/apply/

## Free Sources

- [Freesound](https://freesound.org/)
- [Mixkit](https://mixkit.co/free-sound-effects/)
- [Zapsplat](https://www.zapsplat.com/)

## Usage

```python
from uvg_core.sfx_engine import SFXEngine

engine = SFXEngine(freesound_key="YOUR_KEY")
sfx = engine.get_sfx("transition:whip_pan")
```
