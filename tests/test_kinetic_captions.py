"""
Unit tests for kinetic_captions module.
"""

import pytest


class TestCaptionLayer:
    """Test CaptionLayer dataclass."""
    
    def test_caption_layer_creation(self):
        from uvg_core.kinetic_captions import CaptionLayer
        
        layer = CaptionLayer(
            text="Hello World",
            start_ms=1000,
            end_ms=3000,
            animation="fade_in"
        )
        
        assert layer.text == "Hello World"
        assert layer.start_ms == 1000
        assert layer.end_ms == 3000
        assert layer.animation == "fade_in"
    
    def test_caption_layer_timing_properties(self):
        from uvg_core.kinetic_captions import CaptionLayer
        
        layer = CaptionLayer(text="Test", start_ms=1000, end_ms=2500)
        
        assert layer.start_sec == 1.0
        assert layer.end_sec == 2.5
        assert layer.duration_sec == 1.5
    
    def test_caption_layer_to_dict(self):
        from uvg_core.kinetic_captions import CaptionLayer
        
        layer = CaptionLayer(text="Test", start_ms=0, end_ms=1000)
        data = layer.to_dict()
        
        assert "text" in data
        assert "animation" in data
        assert "font_size" in data


class TestAnimationPresets:
    """Test animation preset definitions."""
    
    def test_presets_exist(self):
        from uvg_core.kinetic_captions import ANIMATION_PRESETS
        
        assert "slide_left" in ANIMATION_PRESETS
        assert "slide_right" in ANIMATION_PRESETS
        assert "fade_in" in ANIMATION_PRESETS
        assert "pop" in ANIMATION_PRESETS
        assert "bounce" in ANIMATION_PRESETS
    
    def test_preset_has_required_keys(self):
        from uvg_core.kinetic_captions import ANIMATION_PRESETS
        
        for name, preset in ANIMATION_PRESETS.items():
            assert "x_expr" in preset, f"{name} missing x_expr"
            assert "y_expr" in preset, f"{name} missing y_expr"
            assert "alpha_expr" in preset, f"{name} missing alpha_expr"


class TestCaptionStyles:
    """Test caption style definitions."""
    
    def test_styles_exist(self):
        from uvg_core.kinetic_captions import CAPTION_STYLES
        
        assert "tiktok" in CAPTION_STYLES
        assert "youtube" in CAPTION_STYLES
        assert "cinematic" in CAPTION_STYLES
    
    def test_style_has_required_keys(self):
        from uvg_core.kinetic_captions import CAPTION_STYLES
        
        required_keys = ["font_size", "font_family", "color", "animation"]
        
        for style_name, style in CAPTION_STYLES.items():
            for key in required_keys:
                assert key in style, f"{style_name} missing {key}"


class TestKineticCaptions:
    """Test KineticCaptions class."""
    
    def test_generate_layers_from_timings(self, mock_word_timings):
        from uvg_core.kinetic_captions import KineticCaptions
        
        engine = KineticCaptions(mock_mode=True)
        layers = engine.generate_layers(mock_word_timings, style="youtube")
        
        assert len(layers) > 0
        assert all(isinstance(l.text, str) for l in layers)
    
    def test_generate_layers_empty_input(self):
        from uvg_core.kinetic_captions import KineticCaptions
        
        engine = KineticCaptions(mock_mode=True)
        layers = engine.generate_layers([])
        
        assert layers == []
    
    def test_to_ffmpeg_filter(self, mock_word_timings):
        from uvg_core.kinetic_captions import KineticCaptions
        
        engine = KineticCaptions(mock_mode=True)
        layers = engine.generate_layers(mock_word_timings)
        
        filter_str = engine.to_ffmpeg_filter(layers)
        
        assert isinstance(filter_str, str)
        assert "drawtext" in filter_str or filter_str == "null"
    
    def test_generate_mock_layers(self):
        from uvg_core.kinetic_captions import KineticCaptions
        
        engine = KineticCaptions(mock_mode=True)
        layers = engine.generate_mock_layers(
            text="This is a test sentence for captions.",
            duration_ms=5000
        )
        
        assert len(layers) > 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_generate_kinetic_captions_function(self, mock_word_timings):
        from uvg_core.kinetic_captions import generate_kinetic_captions
        
        layers = generate_kinetic_captions(mock_word_timings, style="tiktok")
        
        assert len(layers) > 0
    
    def test_captions_to_ffmpeg_function(self, mock_word_timings):
        from uvg_core.kinetic_captions import generate_kinetic_captions, captions_to_ffmpeg
        
        layers = generate_kinetic_captions(mock_word_timings)
        filter_str = captions_to_ffmpeg(layers)
        
        assert isinstance(filter_str, str)
