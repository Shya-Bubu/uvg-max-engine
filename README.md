# UVG MAX – Premium AI Video Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shya-bubu/uvg-max-engine/blob/main/uvg_colab.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎬 Overview

**UVG MAX** is a professional-grade AI video generation engine that creates **Fiverr-quality cinematic videos** from text prompts. It combines multiple AI services to automatically:

- 📝 Generate structured scripts with cinematic story arcs
- 🎥 Search & select the most relevant stock footage
- 🎤 Synthesize natural voiceover with word-level timing
- ✨ Apply premium VFX, transitions, and color grading
- 📱 Render vertical/portrait videos optimized for social media
- 🖼️ Generate trending-style thumbnails

---

## 📁 Folder Structure

```
uvg-max-engine/
├── uvg_core/                    # Core engine modules
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration & presets
│   ├── orchestrator.py          # Master pipeline controller
│   ├── script_generator.py      # AI script generation
│   ├── script_structure.py      # Cinematic story arc
│   ├── creative_director.py     # Scene visualization
│   ├── media_search.py          # Multi-provider stock search
│   ├── vision_scorer.py         # CLIP-based relevance scoring
│   ├── scene_relevance.py       # Semantic validation
│   ├── clip_trimmer.py          # Intelligent clip extraction
│   ├── clip_preparer.py         # Motion & aspect handling
│   ├── tts_engine.py            # Azure TTS with word timing
│   ├── subtitle_engine.py       # Caption generation
│   ├── caption_animation.py     # Animated captions
│   ├── audio_engine.py          # Audio mastering
│   ├── music_engine.py          # Beat detection & sync
│   ├── vfx_engine.py            # Visual effects presets
│   ├── transition_engine.py     # Premium transitions
│   ├── pacing_engine.py         # Beat-level editing
│   ├── ffmpeg_assembler.py      # Final video assembly
│   ├── thumbnail_generator.py   # Trending thumbnails
│   ├── hardware_detector.py     # GPU/CPU optimization
│   ├── gpu_memory_manager.py    # VRAM management
│   ├── disk_watchdog.py         # Storage cleanup
│   └── visual_density_score.py  # Visual quality scoring
├── assets/                      # LUTs, fonts, audio
│   ├── luts/
│   ├── fonts/
│   └── sfx/
├── examples/                    # Example scripts
├── uvg_output/                  # Generated outputs
├── uvg_colab.ipynb              # Google Colab notebook
├── requirements.txt             # Dependencies
├── .env.example                 # API key template
└── README.md
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/shya-bubu/uvg-max-engine.git
cd uvg-max-engine

# Install dependencies
pip install -r requirements.txt

# FFmpeg is required (system dependency)
# Ubuntu: sudo apt install ffmpeg
# Mac: brew install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

---

## 🔑 Environment Setup

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

| Key | Service | Required |
|-----|---------|----------|
| `PEXELS_KEY` | [Pexels](https://www.pexels.com/api/) | ✅ |
| `PIXABAY_KEY` | [Pixabay](https://pixabay.com/api/docs/) | ✅ |
| `UNSPLASH_KEY` | [Unsplash](https://unsplash.com/developers) | Optional |
| `GEMINI_API_KEY` | [Google AI](https://makersuite.google.com/app/apikey) | ✅ |
| `AZURE_TTS_KEY` | [Azure Speech](https://azure.microsoft.com/en-us/products/ai-services/text-to-speech) | ✅ |
| `AZURE_TTS_REGION` | Azure region (e.g., `eastus`) | ✅ |
| `FREESOUND_KEY` | [Freesound](https://freesound.org/apiv2/apply) | Optional |

---

## 📖 Basic Usage

```python
from uvg_core.orchestrator import Orchestrator
from uvg_core.config import UVGConfig

# Load configuration
config = UVGConfig.from_env()
config.validate()

# Initialize orchestrator
orch = Orchestrator(config=config.to_dict())

# Generate video
result = orch.run_pipeline(
    script={
        "title": "Mountain Sunrise",
        "scenes": [
            {"text": "A peaceful sunrise over majestic mountains.", "emotion": "awe"},
            {"text": "Golden light spreads across the valley.", "emotion": "peace"},
            {"text": "A new day begins with endless possibilities.", "emotion": "hope"},
        ]
    }
)

print(f"Video saved to: {result['output_path']}")
```

---

## ✨ Features

### 🎬 Script Generation
- **Gemini-powered** script writing with fallback chain
- **Cinematic structure**: Hook → Buildup → Peak → Resolution → CTA
- **Scene-specific** visual descriptors for better clip matching

### 🔍 Intelligent Media Search
- **Multi-provider**: Pexels, Pixabay, Unsplash
- **HEAD pre-filter**: Skip oversized files before download
- **CLIP-based scoring**: 50% semantic relevance + quality metrics

### 🎤 Professional Audio
- **Azure TTS** with word-level timing
- **Voice styles**: calm, energetic, dramatic, inspirational
- **Audio mastering**: -14 LUFS normalization, de-esser, compressor

### 🎨 Premium VFX
- **12 emotional presets**: cinematic, dramatic, travel, romantic, etc.
- **13+ transitions**: fade, dissolve, wipe, zoom, radial
- **Film grain, bloom, LUT support**

### 📝 Smart Captions
- **6 style presets**: TikTok, YouTube, Instagram, elegant
- **Word-by-word animation**: pop, bounce, typewriter
- **Face-safe placement** with auto-contrast

### 🖼️ Trending Thumbnails
- **Auto hero frame** extraction at golden ratio
- **Face detection** for subject focus
- **Gradient overlays** and text styling

### 🔧 Robust Engineering
- **Auto-repair**: `redo_scene()` on failure
- **Degrade mode**: Simplify processing after repeated failures
- **Scene hashing**: Resume incomplete projects
- **GPU/CPU fallback**: Works on any hardware

---

## 🎨 Style Presets

| Preset | Description |
|--------|-------------|
| `cinematic` | Warm LUTs, film grain, slow zoom |
| `motivational` | High contrast, bold captions |
| `tiktok` | Fast pacing, vibrant colors |
| `corporate` | Clean, professional look |
| `travel` | Saturated colors, natural feel |
| `documentary` | Minimal processing |
| `romantic` | Soft glow, warm tones |
| `tech` | Cool tones, sharp contrast |

---

## 📋 Requirements

- Python 3.9+
- FFmpeg (system)
- 4GB+ RAM (8GB+ recommended)
- GPU optional (CUDA for faster encoding)

---

## ⚠️ Disclaimer

This project is for **educational and personal use only**. Ensure compliance with:
- Stock footage provider terms of service
- API usage limits and quotas
- Copyright and fair use guidelines

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [OpenCLIP](https://github.com/mlfoundations/open_clip) for vision-language models
- [Pexels](https://www.pexels.com), [Pixabay](https://pixabay.com) for stock footage
- [Azure Cognitive Services](https://azure.microsoft.com/en-us/products/ai-services/) for TTS
- [Google Gemini](https://deepmind.google/technologies/gemini/) for script generation
