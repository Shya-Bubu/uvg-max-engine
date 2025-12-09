# tests/test_94_blueprint.py
"""
Tests for 94/100 Blueprint modules.
"""

import pytest
from pathlib import Path


# =============================================================================
# SCRIPT LOADER TESTS
# =============================================================================

class TestScriptLoader:
    """Tests for script_loader module."""
    
    def test_load_from_dict(self):
        """Test loading script from dictionary."""
        from uvg_core.script_loader import ScriptLoader, ScriptData
        
        loader = ScriptLoader()
        data = {
            "title": "Test Script",
            "scenes": [
                {"text": "Hello world", "duration": 3.0},
                {"text": "Second scene", "duration": 4.0},
            ]
        }
        
        script = loader.load_from_dict(data)
        assert isinstance(script, ScriptData)
        assert script.title == "Test Script"
        assert len(script.scenes) == 2
    
    def test_validate_script(self):
        """Test script validation."""
        from uvg_core.script_loader import ScriptLoader, ScriptData, SceneData
        
        loader = ScriptLoader()
        script = ScriptData(
            title="Test",
            scenes=[SceneData(index=0, text="Hello", duration=5.0)]
        )
        
        result = loader.validate(script)
        assert result.valid
        assert len(result.errors) == 0
    
    def test_cliche_detection(self):
        """Test anti-cliché filter."""
        from uvg_core.script_loader import ScriptLoader, ScriptData, SceneData
        
        loader = ScriptLoader(enable_cliche_filter=True)
        script = ScriptData(
            title="Test",
            scenes=[SceneData(index=0, text="Believe in yourself", duration=5.0)]
        )
        
        result = loader.validate(script)
        assert len(result.warnings) > 0
        assert any("cliché" in w for w in result.warnings)


# =============================================================================
# WHISPER TIMING TESTS
# =============================================================================

class TestWhisperTiming:
    """Tests for whisper_timing module."""
    
    def test_whisper_word_dataclass(self):
        """Test WhisperWord dataclass."""
        from uvg_core.whisper_timing import WhisperWord
        
        word = WhisperWord(word="test", start_ms=100, end_ms=200)
        assert word.word == "test"
        assert word.start_ms == 100
        assert word.end_ms == 200
    
    def test_is_whisper_available(self):
        """Test availability check."""
        from uvg_core.whisper_timing import is_whisper_available
        
        # Should return True or False without error
        result = is_whisper_available()
        assert isinstance(result, bool)
    
    def test_extractor_fallback(self):
        """Test extractor returns valid result without Whisper."""
        from uvg_core.whisper_timing import WhisperTimingExtractor
        
        extractor = WhisperTimingExtractor()
        # Test with non-existent file
        result = extractor.extract_timings("nonexistent.wav")
        assert not result.success or result.model_used == "fallback"


# =============================================================================
# AUDIO MASTERING TESTS
# =============================================================================

class TestAudioMastering:
    """Tests for audio_mastering module."""
    
    def test_engine_creation(self):
        """Test engine initialization."""
        from uvg_core.audio_mastering import AudioMasteringEngine
        
        engine = AudioMasteringEngine()
        assert engine.LUFS_TARGET == -14.0
    
    def test_master_voice_invalid(self):
        """Test mastering with invalid file."""
        from uvg_core.audio_mastering import AudioMasteringEngine
        
        engine = AudioMasteringEngine()
        result = engine.master_voice("nonexistent.wav")
        assert not result.success
    
    def test_ducking_curve_empty_timings(self):
        """Test ducking with no word timings."""
        from uvg_core.audio_mastering import AudioMasteringEngine
        import tempfile
        
        engine = AudioMasteringEngine()
        # With empty timings, should handle gracefully
        result = engine.create_ducking_curve([], "nonexistent.mp3")
        # Should fail due to missing music file
        assert not result.success


# =============================================================================
# VISUAL OVERLAYS TESTS
# =============================================================================

class TestVisualOverlays:
    """Tests for visual_overlays module."""
    
    def test_presets_exist(self):
        """Test overlay presets are defined."""
        from uvg_core.visual_overlays import OVERLAY_PRESETS
        
        assert "cinematic" in OVERLAY_PRESETS
        assert "documentary" in OVERLAY_PRESETS
        assert "vintage" in OVERLAY_PRESETS
    
    def test_apply_grain_invalid(self):
        """Test grain with invalid file."""
        from uvg_core.visual_overlays import VisualOverlayEngine
        
        engine = VisualOverlayEngine()
        result = engine.apply_grain("nonexistent.mp4")
        assert not result.success
    
    def test_get_available_presets(self):
        """Test listing presets."""
        from uvg_core.visual_overlays import get_available_presets
        
        presets = get_available_presets()
        assert len(presets) >= 3


# =============================================================================
# COLOR MATCHING TESTS
# =============================================================================

class TestColorMatching:
    """Tests for color_matching module."""
    
    def test_extract_palette_fallback(self):
        """Test palette extraction fallback."""
        from uvg_core.color_matching import ColorMatchingEngine
        
        engine = ColorMatchingEngine()
        palette = engine.extract_palette("nonexistent.jpg")
        
        # Should return fallback palette
        assert len(palette.colors) == 5
    
    def test_match_histogram_invalid(self):
        """Test histogram match with invalid files."""
        from uvg_core.color_matching import ColorMatchingEngine
        
        engine = ColorMatchingEngine()
        result = engine.match_histogram("nonexistent1.mp4", "nonexistent2.mp4")
        assert not result.success


# =============================================================================
# MOTION EFFECTS TESTS
# =============================================================================

class TestMotionEffects:
    """Tests for motion_effects module."""
    
    def test_shake_invalid(self):
        """Test shake with invalid file."""
        from uvg_core.motion_effects import MotionEffectsEngine
        
        engine = MotionEffectsEngine()
        result = engine.add_camera_shake("nonexistent.mp4")
        assert not result.success
    
    def test_ken_burns_invalid(self):
        """Test Ken Burns with invalid file."""
        from uvg_core.motion_effects import MotionEffectsEngine
        
        engine = MotionEffectsEngine()
        result = engine.add_ken_burns("nonexistent.mp4")
        assert not result.success
    
    def test_parallax_not_implemented(self):
        """Test parallax raises NotImplementedError."""
        from uvg_core.motion_effects import MotionEffectsEngine
        
        engine = MotionEffectsEngine()
        with pytest.raises(NotImplementedError):
            engine.parallax_effect("test.mp4")


# =============================================================================
# CAPTION RENDERER TESTS
# =============================================================================

class TestCaptionRenderer:
    """Tests for caption_renderer module."""
    
    def test_styles_exist(self):
        """Test caption styles are defined."""
        from uvg_core.caption_renderer import CAPTION_STYLES
        
        assert "youtube" in CAPTION_STYLES
        assert "tiktok" in CAPTION_STYLES
        assert "cinematic" in CAPTION_STYLES
    
    def test_render_ass_creates_file(self):
        """Test ASS rendering creates file."""
        from uvg_core.caption_renderer import CaptionRenderer
        import tempfile
        
        renderer = CaptionRenderer()
        word_timings = [
            {"word": "Hello", "start_ms": 100, "end_ms": 300},
            {"word": "world", "start_ms": 350, "end_ms": 550},
        ]
        
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            result = renderer.render_ass(word_timings, "youtube", f.name)
            assert result.success
            assert result.caption_count > 0
    
    def test_manim_not_implemented(self):
        """Test Manim raises NotImplementedError."""
        from uvg_core.caption_renderer import CaptionRenderer
        
        renderer = CaptionRenderer()
        with pytest.raises(NotImplementedError):
            renderer.render_manim([{"word": "test"}], "youtube")


# =============================================================================
# SFX ENGINE TESTS
# =============================================================================

class TestSFXEngine:
    """Tests for sfx_engine module."""
    
    def test_triggers_defined(self):
        """Test SFX triggers are defined."""
        from uvg_core.sfx_engine import SFX_TRIGGERS
        
        assert "transition:whip_pan" in SFX_TRIGGERS
        assert "emotion:tense" in SFX_TRIGGERS
    
    def test_get_sfx_unknown_trigger(self):
        """Test unknown trigger returns None."""
        from uvg_core.sfx_engine import SFXEngine
        
        engine = SFXEngine()
        result = engine.get_sfx("unknown:trigger")
        assert result is None


# =============================================================================
# MULTI CLIP COMPOSER TESTS
# =============================================================================

class TestMultiClipComposer:
    """Tests for multi_clip_composer module."""
    
    def test_blend_modes_defined(self):
        """Test blend modes are defined."""
        from uvg_core.multi_clip_composer import BLEND_MODES
        
        assert "overlay" in BLEND_MODES
        assert "screen" in BLEND_MODES
    
    def test_compose_invalid(self):
        """Test compose with invalid files."""
        from uvg_core.multi_clip_composer import MultiClipComposer
        
        composer = MultiClipComposer()
        result = composer.compose("nonexistent.mp4")
        assert not result.success


# =============================================================================
# EXPORT OPTIMIZER TESTS
# =============================================================================

class TestExportOptimizer:
    """Tests for export_optimizer module."""
    
    def test_platforms_defined(self):
        """Test platform presets are defined."""
        from uvg_core.export_optimizer import PLATFORM_PRESETS
        
        assert "youtube" in PLATFORM_PRESETS
        assert "tiktok" in PLATFORM_PRESETS
        assert "instagram_reels" in PLATFORM_PRESETS
    
    def test_get_available_platforms(self):
        """Test listing platforms."""
        from uvg_core.export_optimizer import get_available_platforms
        
        platforms = get_available_platforms()
        assert "youtube" in platforms
        assert len(platforms) >= 5
    
    def test_export_invalid(self):
        """Test export with invalid file."""
        from uvg_core.export_optimizer import ExportOptimizer
        
        optimizer = ExportOptimizer()
        result = optimizer.export("nonexistent.mp4")
        assert not result.success


# =============================================================================
# ELEVENLABS ADAPTER TESTS
# =============================================================================

class TestElevenLabsAdapter:
    """Tests for ElevenLabsAdapter hook."""
    
    def test_adapter_not_available(self):
        """Test ElevenLabs adapter returns not available."""
        from uvg_core.tts_engine import ElevenLabsAdapter
        
        adapter = ElevenLabsAdapter()
        assert not adapter.is_available()
    
    def test_synthesize_raises(self):
        """Test synthesize raises NotImplementedError."""
        from uvg_core.tts_engine import ElevenLabsAdapter
        
        adapter = ElevenLabsAdapter()
        with pytest.raises(NotImplementedError):
            adapter.synthesize("test", "calm", "output.wav")
    
    def test_clone_voice_raises(self):
        """Test clone_voice raises NotImplementedError."""
        from uvg_core.tts_engine import ElevenLabsAdapter
        
        adapter = ElevenLabsAdapter()
        with pytest.raises(NotImplementedError):
            adapter.clone_voice([])
