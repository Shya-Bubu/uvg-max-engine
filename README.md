# UVG MAX – Premium AI Video Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shya-bubu/uvg-max-engine/blob/main/uvg_colab.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎬 Overview

**UVG MAX** is a professional-grade AI video generation engine that creates **$300 Fiverr-quality cinematic videos** from JSON scripts. It combines multiple AI services to automatically:

- 📝 Load structured scripts via JSON Schema v2.1
- 🗣️ **Fish-Speech S1** TTS (50+ emotions, zero cost)
- 🎥 CLIP-scored stock footage selection
- ✨ Emotion-driven VFX, SFX, and music sync
- 📱 Render videos for TikTok, YouTube, Instagram
- 🖼️ Auto-generate trending thumbnails

**No Gemini or Azure required!** Fish-Speech S1 is completely free and runs locally.

---

## 🚀 Quick Start (Colab)

1. Open the [Colab Notebook](https://colab.research.google.com/github/shya-bubu/uvg-max-engine/blob/main/uvg_colab.ipynb)
2. Paste your Pexels API key (free at [pexels.com/api](https://www.pexels.com/api/))
3. Run all cells
4. Use the Gradio UI to generate your video!

---

## 📁 Folder Structure

```
uvg-max-engine/
├── uvg_core/                    # Core engine (50+ modules)
│   ├── uvg_pipeline.py          # Main pipeline orchestrator
│   ├── schema_v2.py             # JSON schema v2.1 definition
│   ├── gradio_ui.py             # Gradio web interface
│   ├── fish_speech_adapter.py   # Fish-Speech S1 TTS
│   ├── voice_presets.py         # 50+ voice emotion presets
│   ├── scene_emotion.py         # Emotion-driven processing
│   ├── deterministic.py         # Reproducible outputs
│   ├── license_metadata.py      # Attribution tracking
│   ├── vfx_engine.py            # Visual effects
│   ├── music_engine.py          # Beat sync & ducking
│   ├── speed_ramp.py            # Speed ramping
│   └── ...                      # And 40+ more modules
├── uvg_colab.ipynb              # Google Colab notebook
├── requirements.txt             # Dependencies
└── README.md
```

---

## 🔑 API Keys

| Key | Service | Required |
|-----|---------|----------|
| `PEXELS_KEY` | [Pexels](https://www.pexels.com/api/) | ✅ Required |
| `PIXABAY_KEY` | [Pixabay](https://pixabay.com/api/docs/) | ⚪ Optional |
| `FREESOUND_KEY` | [Freesound](https://freesound.org/apiv2/apply) | ⚪ Optional (SFX) |

**No TTS API key needed!** Fish-Speech S1 runs locally for free.

---

## 📖 Usage

### Option 1: Gradio UI (Recommended)

```python
from uvg_core.gradio_ui import launch
launch(share=True)  # Opens web UI
```

### Option 2: JSON Script

```python
from uvg_core.uvg_pipeline import run_from_json

script = {
    "version": "2.1",
    "video_meta": {
        "title": "Mountain Sunrise",
        "narrative_style": "cinematic",
        "orientation": "portrait",
        "resolution": {"width": 1080, "height": 1920},
        "include_captions": True,
        "thumbnail_enabled": True
    },
    "scenes": [
        {
            "scene_id": 1,
            "text": "A peaceful sunrise over majestic mountains.",
            "duration_seconds": 5,
            "emotion": "calm",
            "voice_style": "documentary",
            "search_keywords": "mountain sunrise peaceful"
        }
    ]
}

result = run_from_json(script)
print(f"Video saved to: {result.output_path}")
```

---

## ✨ Features

### 🗣️ Fish-Speech S1 TTS
- **50+ emotion markers**: happy, sad, excited, dramatic, whisper, etc.
- **7 voice presets**: documentary, motivational, cinematic, warm, energetic, calm, dramatic
- **Zero cost** - runs completely locally

### 🎭 Emotion-Driven Processing
Each scene's emotion automatically influences:
- **VFX**: Preset selection, bloom, contrast
- **SFX**: Volume and trigger intensity
- **Music**: Volume and sync mode

### 🎬 Music Sync Modes
- **DUCKING**: Lower music under voice (professional standard)
- **EMOTIONAL**: Volume follows scene emotions
- **BEAT_REACTIVE**: Pumping effect for TikTok energy

### 🔄 Deterministic Mode
Set `deterministic_mode: true` to get identical outputs from the same script.

### 📜 License Tracking
Auto-generates `attribution.txt` for YouTube description with all media sources.

---

## 📋 JSON Schema v2.1

See [schema_v2.py](uvg_core/schema_v2.py) for full schema definition.

Key fields:
```json
{
    "version": "2.1",
    "video_meta": {
        "title": "...",
        "narrative_style": "cinematic",
        "orientation": "portrait",
        "resolution": {"width": 1080, "height": 1920},
        "include_captions": true,
        "thumbnail_enabled": true,
        "deterministic_mode": false
    },
    "scenes": [...]
}
```

---

## 📋 Requirements

- Python 3.9+
- FFmpeg (system)
- 8GB+ RAM (Colab T4 recommended)
- GPU optional but recommended

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Fish-Speech](https://github.com/fishaudio/fish-speech) for amazing open-source TTS
- [OpenCLIP](https://github.com/mlfoundations/open_clip) for vision-language models
- [Pexels](https://www.pexels.com), [Pixabay](https://pixabay.com) for stock footage
