# Overlay Assets

This directory contains video overlay files for cinematic effects.

## Required Files

| File | Format | Purpose |
|------|--------|---------|
| grain_35mm.webm | VP9/WebM | Film grain (subtle) |
| grain_16mm.webm | VP9/WebM | Film grain (grainier) |
| grain_super8.webm | VP9/WebM | Vintage film grain |
| flare_anamorphic.webm | VP9/WebM | Blue anamorphic lens flare |
| flare_warm.webm | VP9/WebM | Warm orange lens flare |
| leak_orange.webm | VP9/WebM | Light leak (warm) |
| leak_cool.webm | VP9/WebM | Light leak (cool) |
| dust_vintage.webm | VP9/WebM | Dust particles |

## Specifications

- Resolution: 1920x1080 (scaled as needed)
- Duration: 10-30 seconds (looped)
- Alpha: Transparent background preferred
- Codec: VP9 with alpha or ProRes 4444

## Free Sources

- [Film Composite](https://www.filmcomposite.com/free-film-overlays)
- [Pexels](https://www.pexels.com/search/videos/film%20grain/)
- [Videezy](https://www.videezy.com/free-video/film-grain)

## Usage

```python
from uvg_core.visual_overlays import VisualOverlayEngine

engine = VisualOverlayEngine()
engine.apply_grain("input.mp4", intensity=0.08)
```
