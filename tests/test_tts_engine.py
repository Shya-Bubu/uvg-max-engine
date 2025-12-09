"""
Unit tests for tts_engine module.
"""

import pytest
from pathlib import Path


class TestWordTiming:
    """Test WordTiming dataclass."""
    
    def test_word_timing_creation(self):
        from uvg_core.tts_engine import WordTiming
        
        timing = WordTiming(word="hello", start_ms=0, end_ms=500)
        
        assert timing.word == "hello"
        assert timing.start_ms == 0
        assert timing.end_ms == 500
    
    def test_word_timing_to_dict(self):
        from uvg_core.tts_engine import WordTiming
        
        timing = WordTiming(word="test", start_ms=100, end_ms=300)
        data = timing.to_dict()
        
        assert data["word"] == "test"
        assert data["start_ms"] == 100
        assert data["end_ms"] == 300


class TestTTSResult:
    """Test TTSResult dataclass."""
    
    def test_tts_result_creation(self):
        from uvg_core.tts_engine import TTSResult, WordTiming
        
        result = TTSResult(
            success=True,
            text="Hello world",
            audio_path="/tmp/test.wav",
            duration_ms=2000,
            word_timings=[
                WordTiming(word="Hello", start_ms=0, end_ms=500),
                WordTiming(word="world", start_ms=600, end_ms=1200),
            ]
        )
        
        assert result.success is True
        assert len(result.word_timings) == 2
        assert result.duration_ms == 2000


class TestMockTTSAdapter:
    """Test MockTTSAdapter."""
    
    def test_mock_adapter_available(self, temp_audio_dir):
        from uvg_core.tts_engine import MockTTSAdapter
        
        adapter = MockTTSAdapter(temp_audio_dir)
        assert adapter.is_available() is True
    
    def test_mock_adapter_synthesize(self, temp_audio_dir):
        from uvg_core.tts_engine import MockTTSAdapter
        
        adapter = MockTTSAdapter(temp_audio_dir)
        output_path = str(temp_audio_dir / "test.wav")
        
        result = adapter.synthesize(
            text="This is a test sentence.",
            voice_style="calm",
            output_path=output_path
        )
        
        assert result.success is True
        assert len(result.word_timings) > 0
        assert result.duration_ms > 0
        assert Path(output_path).exists()


class TestGeminiTTSAdapter:
    """Test GeminiTTSAdapter (MOCK)."""
    
    def test_gemini_adapter_without_key(self, temp_audio_dir):
        from uvg_core.tts_engine import GeminiTTSAdapter
        
        adapter = GeminiTTSAdapter(
            api_key="",
            model_name="gemini-2.5-flash-tts",
            output_dir=temp_audio_dir
        )
        
        # Without key, should not be available
        assert adapter.is_available() is False
    
    def test_gemini_adapter_with_key_uses_mock(self, temp_audio_dir):
        from uvg_core.tts_engine import GeminiTTSAdapter
        
        adapter = GeminiTTSAdapter(
            api_key="test-key",
            model_name="gemini-2.5-flash-tts",
            output_dir=temp_audio_dir
        )
        
        output_path = str(temp_audio_dir / "gemini_test.wav")
        result = adapter.synthesize("Test text", "calm", output_path)
        
        # Should succeed (using mock fallback)
        assert result.success is True


class TestTTSEngine:
    """Test TTSEngine facade."""
    
    def test_engine_defaults_to_mock(self, temp_audio_dir, monkeypatch):
        """Test that engine defaults to mock adapter."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        
        from uvg_core.tts_engine import TTSEngine
        
        engine = TTSEngine(output_dir=temp_audio_dir)
        
        assert engine.mock_mode is True
        assert engine._adapter is not None
    
    def test_engine_adapter_selection_mock(self, temp_audio_dir, monkeypatch):
        """Test adapter selection for mock mode."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        monkeypatch.setenv("UVG_TTS_PROVIDER", "mock")
        
        from uvg_core.tts_engine import TTSEngine, MockTTSAdapter
        
        engine = TTSEngine(output_dir=temp_audio_dir, mock_mode=True)
        
        assert isinstance(engine._adapter, MockTTSAdapter)
