"""
Pytest fixtures for UVG MAX tests.

All external APIs are mocked by default.
Set UVG_RUN_INTEGRATION=true for real API tests.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# AUTO-ENABLE MOCK MODE
# =============================================================================

@pytest.fixture(autouse=True)
def auto_mock_mode(monkeypatch):
    """Automatically enable mock mode for all tests."""
    monkeypatch.setenv("UVG_MOCK_MODE", "true")
    monkeypatch.setenv("UVG_DEBUG_SEED", "42")
    monkeypatch.setenv("UVG_TTS_PROVIDER", "mock")


# =============================================================================
# TEMP DIRECTORIES
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_clips_dir(temp_output_dir):
    """Create a temporary clips directory."""
    clips = temp_output_dir / "clips"
    clips.mkdir()
    return clips


@pytest.fixture
def temp_audio_dir(temp_output_dir):
    """Create a temporary audio directory."""
    audio = temp_output_dir / "audio"
    audio.mkdir()
    return audio


# =============================================================================
# MOCK GEMINI API
# =============================================================================

@pytest.fixture
def mock_gemini_api(monkeypatch):
    """Mock all Gemini API calls."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text='[{"text": "Test scene", "emotion": "neutral", "tension": 0.5}]'
    )
    
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    monkeypatch.setattr("google.generativeai", mock_genai, raising=False)
    return mock_genai


# =============================================================================
# MOCK HTTP REQUESTS
# =============================================================================

@pytest.fixture
def mock_http_requests(monkeypatch):
    """Mock HTTP requests (requests.get, requests.head)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Length": "1024"}
    mock_response.content = b"mock content"
    mock_response.iter_content = lambda chunk_size: [b"mock content"]
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {"videos": []}
    
    with patch("requests.get", return_value=mock_response):
        with patch("requests.head", return_value=mock_response):
            yield mock_response


# =============================================================================
# MOCK TTS RESULT
# =============================================================================

@pytest.fixture
def mock_tts_result():
    """Create a mock TTS result."""
    from uvg_core.tts_engine import TTSResult, WordTiming
    
    return TTSResult(
        success=True,
        text="This is a test sentence for TTS.",
        audio_path="/tmp/test_audio.wav",
        duration_ms=3000,
        word_timings=[
            WordTiming(word="This", start_ms=0, end_ms=200),
            WordTiming(word="is", start_ms=200, end_ms=350),
            WordTiming(word="a", start_ms=350, end_ms=450),
            WordTiming(word="test", start_ms=450, end_ms=700),
            WordTiming(word="sentence", start_ms=700, end_ms=1100),
            WordTiming(word="for", start_ms=1100, end_ms=1300),
            WordTiming(word="TTS.", start_ms=1300, end_ms=1600),
        ],
        voice_style="calm",
    )


@pytest.fixture
def mock_word_timings(mock_tts_result):
    """Extract word timings from mock TTS result."""
    return mock_tts_result.word_timings


# =============================================================================
# MOCK SCRIPT
# =============================================================================

@pytest.fixture
def mock_script():
    """Create a mock script for testing."""
    from uvg_core.script_generator import Script, Scene
    
    script = Script(
        title="Test Video",
        style="cinematic",
        source="mock",
    )
    
    script.scenes = [
        Scene(index=0, text="Welcome to the journey.", duration=4.0, emotion="hope", tension=0.3),
        Scene(index=1, text="Every step matters.", duration=4.0, emotion="motivational", tension=0.6),
        Scene(index=2, text="The future is bright.", duration=4.0, emotion="happy", tension=0.4),
    ]
    script.total_duration = sum(s.duration for s in script.scenes)
    
    return script


# =============================================================================
# SKIP INTEGRATION
# =============================================================================

@pytest.fixture
def skip_unless_integration():
    """Skip test unless integration tests are enabled."""
    if os.getenv("UVG_RUN_INTEGRATION", "").lower() != "true":
        pytest.skip("Integration tests disabled. Set UVG_RUN_INTEGRATION=true to run.")


# =============================================================================
# SAMPLE FILES
# =============================================================================

@pytest.fixture
def sample_video_path(temp_output_dir):
    """Create a placeholder video file path."""
    video = temp_output_dir / "sample.mp4"
    # Create empty file as placeholder
    video.touch()
    return video


@pytest.fixture
def sample_audio_path(temp_audio_dir):
    """Create a placeholder audio file path."""
    audio = temp_audio_dir / "sample.wav"
    audio.touch()
    return audio
