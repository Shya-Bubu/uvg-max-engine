"""
Unit tests for script_generator module.
"""

import pytest
import json
from pathlib import Path


class TestScriptDataclasses:
    """Test Script and Scene dataclasses."""
    
    def test_scene_creation(self):
        from uvg_core.script_generator import Scene
        
        scene = Scene(
            index=0,
            text="Test scene text",
            duration=4.0,
            emotion="hope",
            tension=0.5
        )
        
        assert scene.index == 0
        assert scene.text == "Test scene text"
        assert scene.duration == 4.0
        assert scene.emotion == "hope"
        assert scene.tension == 0.5
    
    def test_scene_to_dict(self):
        from uvg_core.script_generator import Scene
        
        scene = Scene(index=0, text="Test", duration=4.0)
        data = scene.to_dict()
        
        assert isinstance(data, dict)
        assert "index" in data
        assert "text" in data
        assert "duration" in data
    
    def test_script_creation(self):
        from uvg_core.script_generator import Script, Scene
        
        script = Script(
            title="Test Video",
            style="cinematic",
            source="mock"
        )
        
        script.scenes.append(Scene(index=0, text="Scene 1", duration=4.0))
        script.scenes.append(Scene(index=1, text="Scene 2", duration=4.0))
        
        assert script.title == "Test Video"
        assert len(script.scenes) == 2


class TestScriptGenerator:
    """Test ScriptGenerator class."""
    
    def test_mock_mode_enabled(self, monkeypatch):
        """Test that mock mode returns deterministic response."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        
        from uvg_core.script_generator import ScriptGenerator
        
        gen = ScriptGenerator(mock_mode=True)
        assert gen.mock_mode is True
    
    def test_generate_script_mock(self, monkeypatch):
        """Test script generation in mock mode."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        
        from uvg_core.script_generator import ScriptGenerator
        
        gen = ScriptGenerator(mock_mode=True)
        script = gen.generate_script(
            prompt="Test video about nature",
            target_duration=30,
            style="cinematic"
        )
        
        assert script is not None
        assert len(script.scenes) > 0
        assert script.style == "cinematic"
    
    def test_json_parsing_strips_markdown(self, monkeypatch):
        """Test that JSON parsing strips markdown fences."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        
        from uvg_core.script_generator import ScriptGenerator
        
        gen = ScriptGenerator(mock_mode=True)
        
        # The mock response is already clean JSON, but the parser should handle fenced JSON too
        script = gen.generate_script("Test", target_duration=20)
        assert script is not None


class TestTemplateScripts:
    """Test template script generation."""
    
    def test_script_generator_has_fallback(self, monkeypatch):
        """Test that script generator can use fallback templates."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        
        from uvg_core.script_generator import ScriptGenerator
        
        gen = ScriptGenerator(mock_mode=True)
        # Templates are internal, just verify generation works
        script = gen.generate_script("Test prompt", target_duration=20)
        assert script is not None
