# UVG MAX - How to Run

Complete guide for running UVG MAX in mock mode and real API mode.

---

## Quick Start (Mock Mode)

No API keys required! Perfect for testing and development.

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (uses mock mode by default)
python scripts/run_demo.py --mock

# Run tests
python -m pytest tests/ -v
```

**Output:**
- `uvg_output/final/sample.mp4`
- `uvg_output/thumbnails/sample_thumb.png`
- `uvg_output/audio/scene_*.wav`

---

## Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

#### Mock Mode (default)
```bash
UVG_MOCK_MODE=true       # Use mock APIs
UVG_DEBUG_SEED=42        # Deterministic output
UVG_TTS_PROVIDER=mock    # Use MockTTSAdapter
```

#### Real API Mode
```bash
UVG_MOCK_MODE=false
UVG_TTS_PROVIDER=azure   # or gemini

# Required for real mode:
GEMINI_API_KEY=your_key
AZURE_TTS_KEY=your_key
AZURE_TTS_REGION=eastus
PEXELS_KEY=your_key
PIXABAY_KEY=your_key
```

---

## Run Modes

### 1. Mock Mode (Default)
```bash
python scripts/run_demo.py --mock
```
- All APIs are mocked
- Deterministic output (seed=42)
- No external calls
- Audio: silence WAV files

### 2. Real API Mode
```bash
python scripts/run_demo.py --real
```
- Requires API keys in `.env`
- Real Gemini script generation
- Real Azure/Gemini TTS
- Real stock footage search

---

## Gemini Model Selection

Configure which Gemini models to use:

```bash
# Script generation (structured JSON)
UVG_GEMINI_SCRIPT_MODEL=gemini-2.5-flash

# Creative direction (visual descriptors)
UVG_GEMINI_CREATIVE_MODEL=gemini-2.5-flash-live

# TTS (when available)
UVG_GEMINI_TTS_MODEL=gemini-2.5-flash-tts
```

> **Note:** If gemini-2.5 models are unavailable, the system uses mock mode automatically. There is **NO fallback to gemini-1.5-flash**.

---

## TTS Provider Options

### Mock (default)
```bash
UVG_TTS_PROVIDER=mock
```
- Generates silence WAV with word timings
- No API key required

### Azure TTS
```bash
UVG_TTS_PROVIDER=azure
AZURE_TTS_KEY=your_key
AZURE_TTS_REGION=eastus
```

### Gemini TTS (when available)
```bash
UVG_TTS_PROVIDER=gemini
UVG_GEMINI_TTS_KEY=your_key
```
> Currently uses mock fallback until API is available.

---

## Testing

### Run all unit tests
```bash
python -m pytest tests/ -v
```

### Run specific test module
```bash
python -m pytest tests/test_tts_engine.py -v
```

### Run integration tests (requires API keys)
```bash
UVG_RUN_INTEGRATION=true python -m pytest tests/ -v -m integration
```

---

## Output Structure

```
uvg_output/
├── audio/
│   └── scene_*.wav       # TTS audio per scene
├── clips/
│   └── *.mp4             # Downloaded stock clips
├── final/
│   └── sample.mp4        # Final assembled video
├── frames/
│   └── *.jpg             # Keyframes
└── thumbnails/
    └── sample_thumb.png  # Generated thumbnail
```

---

## Troubleshooting

### UnicodeEncodeError on Windows
```bash
$env:PYTHONIOENCODING='utf-8'; python scripts/run_demo.py
```

### FFmpeg not found
Install FFmpeg and add to PATH:
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

### API rate limits
- Set `MAX_DOWNLOAD_WORKERS=3` to reduce parallel requests
- Enable mock mode for burst development

---

## Premium Features

The following premium features are available:

| Feature | Description |
|---------|-------------|
| Kinetic Captions | Animated text overlays (slide, fade, pop, bounce) |
| Emotion Mapping | Scene text → emotion → creative decisions |
| Camera Motion | zoom-in, zoom-out, pan, tilt, drone, handheld |
| Color Grades | warm, cold, cinematic, documentary, dramatic |
| Smart Thumbnails | Best frame extraction with effects |

---

## API Integration TODOs

When real Gemini 2.5 APIs become available:

1. Set `UVG_MOCK_MODE=false`
2. Implement real calls in:
   - `GeminiTTSAdapter.synthesize()` 
   - `NativeAudioAdapter.synthesize()`
3. Test with: `python -m pytest tests/ -v -m integration`
