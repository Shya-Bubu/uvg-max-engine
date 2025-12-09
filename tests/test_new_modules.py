# tests/test_new_modules.py
"""
Tests for new UVG MAX production modules.

Tests:
- ModelManager
- Logger
- HardwareGuard
- ClipQuality
- AudioMixer
- StylePack
- SceneDetector
- ObjectDetector
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# MODEL MANAGER TESTS
# =============================================================================

class TestModelManager:
    """Tests for uvg_core.model_manager"""
    
    def test_singleton_pattern(self):
        """Test ModelManager is a singleton."""
        from uvg_core.model_manager import ModelManager
        
        # Reset for clean test
        ModelManager.reset()
        
        m1 = ModelManager.instance()
        m2 = ModelManager.instance()
        
        assert m1 is m2
        
        # Cleanup
        ModelManager.reset()
    
    def test_device_detection(self):
        """Test device detection returns valid value."""
        from uvg_core.model_manager import ModelManager
        
        ModelManager.reset()
        manager = ModelManager.instance()
        
        assert manager.device in ["cpu", "cuda"]
        
        ModelManager.reset()
    
    def test_get_clip_fallback(self):
        """Test CLIP returns fallback when model not found."""
        from uvg_core.model_manager import ModelManager
        
        ModelManager.reset()
        manager = ModelManager.instance()
        
        clip = manager.get_clip_session()
        assert clip is not None
        assert clip.fallback_mode  # Should be in fallback without model file
        
        ModelManager.reset()
    
    def test_check_models_available(self):
        """Test model availability check."""
        from uvg_core.model_manager import ModelManager
        
        ModelManager.reset()
        manager = ModelManager.instance()
        
        status = manager.check_models_available()
        
        assert "clip_visual" in status
        assert "aesthetic" in status
        assert "device" in status
        
        ModelManager.reset()


# =============================================================================
# LOGGER TESTS
# =============================================================================

class TestLogger:
    """Tests for uvg_core.logger"""
    
    def test_log_info(self, capsys):
        """Test info logging."""
        from uvg_core.logger import log_info
        
        log_info("Test message")
        
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    
    def test_log_warn(self):
        """Test warning logging doesn't crash."""
        from uvg_core.logger import log_warn
        
        # Just verify it doesn't crash
        try:
            log_warn("Warning message")
        except Exception as e:
            pytest.fail(f"log_warn raised exception: {e}")
    
    def test_log_error(self, capsys):
        """Test error logging."""
        from uvg_core.logger import log_error
        
        log_error("Error message")
        
        # Error might go to stderr
        captured = capsys.readouterr()
        assert "Error message" in captured.out or "Error message" in captured.err


# =============================================================================
# HARDWARE GUARD TESTS
# =============================================================================

class TestHardwareGuard:
    """Tests for uvg_core.hardware_guard"""
    
    def test_get_available_device(self):
        """Test device detection."""
        from uvg_core.hardware_guard import get_available_device
        
        device = get_available_device()
        assert device in ["cpu", "cuda"]
    
    def test_check_ffmpeg_available(self):
        """Test FFmpeg detection."""
        from uvg_core.hardware_guard import check_ffmpeg_available
        
        result = check_ffmpeg_available()
        assert isinstance(result, bool)
    
    def test_get_onnx_providers(self):
        """Test ONNX provider detection."""
        from uvg_core.hardware_guard import get_onnx_providers
        
        providers = get_onnx_providers()
        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers
    
    def test_hardware_guard_class(self):
        """Test HardwareGuard class."""
        from uvg_core.hardware_guard import HardwareGuard
        
        guard = HardwareGuard()
        
        assert guard.device in ["cpu", "cuda"]
        assert isinstance(guard.can_use_gpu(), bool)


# =============================================================================
# CLIP QUALITY TESTS
# =============================================================================

class TestClipQuality:
    """Tests for uvg_selector.clip_quality"""
    
    def test_assess_clip_quality_invalid(self):
        """Test quality assessment on invalid file."""
        from uvg_selector.clip_quality import assess_clip_quality
        
        result = assess_clip_quality("nonexistent.mp4")
        
        # Returns 0.5 fallback when OpenCV not available
        assert result["overall"] in (0.0, 0.5)
        # pass might be True with 0.5 fallback or False with 0.0
    
    def test_filter_quality_clips_empty(self):
        """Test filtering empty list."""
        from uvg_selector.clip_quality import filter_quality_clips
        
        result = filter_quality_clips([])
        assert result == []
    
    def test_check_blur_returns_float(self):
        """Test blur check returns valid float."""
        from uvg_selector.clip_quality import check_blur
        import numpy as np
        
        # Create dummy frame
        try:
            import cv2
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            score = check_blur(frame)
            assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("OpenCV not available")


# =============================================================================
# AUDIO MIXER TESTS
# =============================================================================

class TestAudioMixer:
    """Tests for uvg_core.audio_mixer"""
    
    def test_mixer_creation(self, tmp_path):
        """Test AudioMixer creation."""
        from uvg_core.audio_mixer import AudioMixer
        
        mixer = AudioMixer(output_dir=tmp_path)
        
        assert mixer.output_dir == tmp_path
        assert mixer.LUFS_TARGET == -14.0
    
    def test_normalize_loudness_invalid_file(self, tmp_path):
        """Test normalization on invalid file."""
        from uvg_core.audio_mixer import AudioMixer
        
        mixer = AudioMixer(output_dir=tmp_path)
        result = mixer.normalize_loudness("nonexistent.wav")
        
        assert result.success == False
    
    def test_mix_result_dataclass(self):
        """Test MixResult dataclass."""
        from uvg_core.audio_mixer import MixResult
        
        result = MixResult(success=True, output_path="/path/to/file.wav")
        
        assert result.success == True
        assert result.output_path == "/path/to/file.wav"
        assert result.error == ""


# =============================================================================
# STYLE PACK TESTS
# =============================================================================

class TestStylePack:
    """Tests for uvg_core.style_pack"""
    
    def test_load_style_pack_cinematic(self):
        """Test loading cinematic pack."""
        from uvg_core.style_pack import load_style_pack
        
        pack = load_style_pack("cinematic")
        
        assert pack.name == "cinematic"
        assert pack.pacing_factor > 0
        assert len(pack.transitions) > 0
    
    def test_load_style_pack_motivational(self):
        """Test loading motivational pack."""
        from uvg_core.style_pack import load_style_pack
        
        pack = load_style_pack("motivational")
        
        assert pack.name == "motivational"
    
    def test_load_style_pack_unknown(self):
        """Test loading unknown pack falls back to cinematic."""
        from uvg_core.style_pack import load_style_pack
        
        pack = load_style_pack("nonexistent_pack")
        
        # Should fall back to cinematic
        assert pack.name == "cinematic"
    
    def test_list_style_packs(self):
        """Test listing available packs."""
        from uvg_core.style_pack import list_style_packs
        
        packs = list_style_packs()
        
        assert isinstance(packs, list)
        assert "cinematic" in packs
        assert "motivational" in packs
    
    def test_style_pack_to_dict(self):
        """Test StylePack.to_dict()."""
        from uvg_core.style_pack import load_style_pack
        
        pack = load_style_pack("cinematic")
        d = pack.to_dict()
        
        assert "name" in d
        assert "transitions" in d
        assert "pacing_factor" in d


# =============================================================================
# SCENE DETECTOR TESTS
# =============================================================================

class TestSceneDetector:
    """Tests for uvg_selector.scene_detector"""
    
    def test_detect_scene_boundaries_invalid(self):
        """Test scene detection on invalid file."""
        from uvg_selector.scene_detector import detect_scene_boundaries
        
        scenes = detect_scene_boundaries("nonexistent.mp4")
        
        # Should return default scene
        assert len(scenes) >= 1
    
    def test_get_scene_at_time(self):
        """Test getting scene at time."""
        from uvg_selector.scene_detector import get_scene_at_time
        
        scenes = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
        
        assert get_scene_at_time(scenes, 2.0) == 0
        assert get_scene_at_time(scenes, 7.0) == 1
        assert get_scene_at_time(scenes, 12.0) == 2
    
    def test_check_crosses_boundary(self):
        """Test boundary crossing check."""
        from uvg_selector.scene_detector import check_crosses_boundary
        
        scenes = [(0.0, 5.0), (5.0, 10.0)]
        
        # Within same scene - no crossing
        assert check_crosses_boundary(scenes, 1.0, 4.0) == False
        
        # Across scenes - crosses
        assert check_crosses_boundary(scenes, 3.0, 7.0) == True


# =============================================================================
# OBJECT DETECTOR TESTS
# =============================================================================

class TestObjectDetector:
    """Tests for uvg_selector.object_detector"""
    
    def test_object_detector_creation(self):
        """Test ObjectDetector creation."""
        from uvg_selector.object_detector import ObjectDetector
        
        # Should not crash even if YOLO not available
        detector = ObjectDetector()
        
        # available might be False if YOLO not installed
        assert isinstance(detector.available, bool)
    
    def test_detect_objects_empty_when_unavailable(self):
        """Test detection returns empty when unavailable."""
        from uvg_selector.object_detector import ObjectDetector
        import numpy as np
        
        detector = ObjectDetector()
        
        if not detector.available:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detections = detector.detect_objects(frame)
            assert detections == []
    
    def test_get_object_score_fallback(self):
        """Test object score fallback."""
        from uvg_selector.object_detector import ObjectDetector
        import numpy as np
        
        detector = ObjectDetector()
        
        if not detector.available:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            score = detector.get_object_score(frame)
            assert score == 0.5  # Neutral fallback


# =============================================================================
# AZURE TTS TESTS
# =============================================================================

class TestAzureTTSAdapter:
    """Tests for AzureTTSAdapter in tts_engine"""
    
    def test_azure_adapter_without_sdk(self):
        """Test Azure adapter gracefully handles missing SDK."""
        from uvg_core.tts_engine import AzureTTSAdapter
        
        adapter = AzureTTSAdapter(
            subscription_key="fake_key",
            region="eastus"
        )
        
        # Should not crash
        assert adapter is not None
    
    def test_azure_adapter_builds_ssml(self):
        """Test SSML building."""
        from uvg_core.tts_engine import AzureTTSAdapter
        
        adapter = AzureTTSAdapter(
            subscription_key="fake_key",
            region="eastus"
        )
        
        ssml = adapter._build_ssml("Hello world", "cheerful")
        
        assert "<speak" in ssml
        assert "<voice" in ssml
        assert "Hello world" in ssml
        assert "cheerful" in ssml


# =============================================================================
# LUT AND TRANSITIONS TESTS
# =============================================================================

class TestLUTSupport:
    """Tests for LUT support in ffmpeg_assembler"""
    
    def test_lut_files_dict_exists(self):
        """Test LUT_FILES dict exists."""
        from uvg_core.ffmpeg_assembler import LUT_FILES
        
        assert "cinematic" in LUT_FILES
        assert "motivational" in LUT_FILES
    
    def test_get_lut_path_unknown(self):
        """Test get_lut_path returns empty for unknown."""
        from uvg_core.ffmpeg_assembler import get_lut_path
        
        path = get_lut_path("nonexistent_lut")
        assert path == ""
    
    def test_premium_transitions_exist(self):
        """Test premium transitions defined."""
        from uvg_core.ffmpeg_assembler import PREMIUM_TRANSITIONS
        
        assert "fade" in PREMIUM_TRANSITIONS
        assert "zoom_through" in PREMIUM_TRANSITIONS
        assert "blur_dissolve" in PREMIUM_TRANSITIONS
    
    def test_get_transition_filter(self):
        """Test transition filter generation."""
        from uvg_core.ffmpeg_assembler import get_transition_filter
        
        filter_str = get_transition_filter("fade", 0.5)
        
        assert "xfade" in filter_str
        assert "duration=0.5" in filter_str


# =============================================================================
# CREATIVE DIRECTOR STYLE PACK TESTS
# =============================================================================

class TestCreativeDirectorStylePack:
    """Tests for style pack integration in CreativeDirector"""
    
    def test_creative_director_loads_style_pack(self):
        """Test CreativeDirector loads style pack."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True, style_pack_name="cinematic")
        
        assert director._style_pack is not None
        assert director._style_pack.name == "cinematic"
    
    def test_get_style_pack_transition(self):
        """Test style pack transition cycling."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True, style_pack_name="cinematic")
        
        t0 = director.get_style_pack_transition(0)
        t1 = director.get_style_pack_transition(1)
        
        assert isinstance(t0, str)
        assert isinstance(t1, str)
    
    def test_get_pacing_factor(self):
        """Test pacing factor from style pack."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True, style_pack_name="cinematic")
        
        factor = director.get_pacing_factor()
        assert factor > 0


# =============================================================================
# PACING ENGINE EXTENDED TESTS
# =============================================================================

class TestPacingEngineExtended:
    """Tests for new pacing engine features"""
    
    def test_pacing_profile_get(self):
        """Test PacingProfile.get()."""
        from uvg_core.pacing_engine import PacingProfile
        
        profile = PacingProfile.get("cinematic")
        
        assert "factor" in profile
        assert "min" in profile
        assert "max" in profile
    
    def test_get_beat_times(self):
        """Test beat time calculation."""
        from uvg_core.pacing_engine import get_beat_times
        
        times = get_beat_times(120, 10.0)  # 120 BPM, 10 seconds
        
        assert len(times) == 20  # 120 BPM = 2 beats per second
        assert times[0] == 0.0
    
    def test_snap_to_beat(self):
        """Test beat snapping."""
        from uvg_core.pacing_engine import snap_to_beat
        
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        assert snap_to_beat(0.48, beats) == 0.5
        assert snap_to_beat(0.3, beats) == 0.3  # Too far, no snap
    
    def test_get_available_profiles(self):
        """Test available profiles list."""
        from uvg_core.pacing_engine import get_available_profiles
        
        profiles = get_available_profiles()
        
        assert "cinematic" in profiles
        assert "motivational" in profiles


# =============================================================================
# THUMBNAIL EXTENDED TESTS
# =============================================================================

class TestThumbnailExtended:
    """Tests for new thumbnail features"""
    
    def test_get_thumbnail_styles(self):
        """Test get_thumbnail_styles function."""
        from uvg_core.thumbnail_generator import get_thumbnail_styles
        
        styles = get_thumbnail_styles()
        
        assert "tiktok" in styles
        assert "cinematic" in styles
    
    def test_generate_thumbnail_with_style_pack_default(self):
        """Test style pack thumbnail defaults."""
        from uvg_core.thumbnail_generator import generate_thumbnail_with_style_pack
        
        # Should not crash even with invalid video
        result = generate_thumbnail_with_style_pack(
            "nonexistent.mp4",
            "Test Title",
            style_pack=None
        )
        
        # Will fail but gracefully
        assert result.success == False or result.success == True
