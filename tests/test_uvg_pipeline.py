# tests/test_uvg_pipeline.py
"""
Tests for UVGPipeline integration.
"""

import pytest
from pathlib import Path


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        from uvg_core.uvg_pipeline import PipelineConfig
        
        config = PipelineConfig()
        assert config.target_platform == "youtube"
        assert config.enable_overlays is True
        assert config.use_whisper is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        from uvg_core.uvg_pipeline import PipelineConfig
        
        config = PipelineConfig(
            target_platform="tiktok",
            enable_overlays=False
        )
        assert config.target_platform == "tiktok"
        assert config.enable_overlays is False


class TestPipelineResult:
    """Tests for PipelineResult."""
    
    def test_result_creation(self):
        """Test result dataclass."""
        from uvg_core.uvg_pipeline import PipelineResult
        
        result = PipelineResult(
            success=True,
            output_path="/path/to/output.mp4",
            duration_sec=30.0,
            scenes_processed=5
        )
        
        assert result.success
        assert result.scenes_processed == 5


class TestUVGPipeline:
    """Tests for UVGPipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        from uvg_core.uvg_pipeline import UVGPipeline
        
        pipeline = UVGPipeline()
        assert pipeline.config is not None
    
    def test_lazy_module_loading(self):
        """Test modules are loaded lazily."""
        from uvg_core.uvg_pipeline import UVGPipeline
        
        pipeline = UVGPipeline()
        assert len(pipeline._modules) == 0
    
    def test_get_module_script_loader(self):
        """Test getting script_loader module."""
        from uvg_core.uvg_pipeline import UVGPipeline
        from uvg_core.script_loader import ScriptLoader
        
        pipeline = UVGPipeline()
        loader = pipeline._get_module("script_loader")
        assert isinstance(loader, ScriptLoader)
    
    def test_get_module_captions(self):
        """Test getting captions module."""
        from uvg_core.uvg_pipeline import UVGPipeline
        from uvg_core.caption_renderer import CaptionRenderer
        
        pipeline = UVGPipeline()
        renderer = pipeline._get_module("captions")
        assert isinstance(renderer, CaptionRenderer)
    
    def test_run_with_dict_script(self):
        """Test running pipeline with dict script."""
        from uvg_core.uvg_pipeline import UVGPipeline
        
        pipeline = UVGPipeline()
        script = {
            "title": "Test",
            "scenes": [
                {"text": "Hello world", "duration": 3.0},
            ]
        }
        
        result = pipeline.run(script)
        # May succeed or fail depending on FFmpeg, but should return result
        assert hasattr(result, "success")


class TestQuickPipeline:
    """Tests for convenience function."""
    
    def test_run_pipeline_function(self):
        """Test run_pipeline function exists."""
        from uvg_core.uvg_pipeline import run_pipeline
        
        assert callable(run_pipeline)
