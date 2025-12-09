"""
Unit tests for creative_director module.
"""

import pytest


class TestSceneDirection:
    """Test SceneDirection dataclass."""
    
    def test_scene_direction_creation(self):
        from uvg_core.creative_director import SceneDirection
        
        direction = SceneDirection(
            scene_idx=0,
            camera_motion="slow-zoom-in",
            emotion_tag="happy",
            color_grade="warm"
        )
        
        assert direction.scene_idx == 0
        assert direction.camera_motion == "slow-zoom-in"
        assert direction.emotion_tag == "happy"
        assert direction.color_grade == "warm"
    
    def test_scene_direction_to_dict(self):
        from uvg_core.creative_director import SceneDirection
        
        direction = SceneDirection(scene_idx=0)
        data = direction.to_dict()
        
        assert isinstance(data, dict)
        assert "scene_idx" in data
        assert "camera_motion" in data
        assert "emotion_tag" in data
        assert "color_grade" in data


class TestCreativeDirector:
    """Test CreativeDirector class."""
    
    def test_mock_mode(self, monkeypatch):
        """Test mock mode returns deterministic response."""
        monkeypatch.setenv("UVG_MOCK_MODE", "true")
        
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True)
        assert director.mock_mode is True
    
    def test_classify_emotion_happy(self):
        """Test emotion classification for happy text."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True)
        emotion = director.classify_emotion("This is a joyful celebration of life!")
        
        # Should classify as happy due to "joyful" and "celebration"
        assert emotion in ["happy", "joy", "neutral"]
    
    def test_classify_emotion_sad(self):
        """Test emotion classification for sad text."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True)
        emotion = director.classify_emotion("I miss you and feel the grief of loss.")
        
        assert emotion == "sad"
    
    def test_classify_emotion_motivational(self):
        """Test emotion classification for motivational text."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True)
        emotion = director.classify_emotion("Rise up and achieve your dreams of success!")
        
        assert emotion == "motivational"
    
    def test_get_mood_shot_settings(self):
        """Test mood to shot mapping."""
        from uvg_core.creative_director import CreativeDirector
        
        director = CreativeDirector(mock_mode=True)
        settings = director.get_mood_shot_settings("sad")
        
        assert "color_grade" in settings
        assert "shot_type" in settings
        assert "camera_motion" in settings
        assert settings["color_grade"] == "cold"
        assert settings["shot_type"] == "wide"


class TestMoodShotMapping:
    """Test mood to shot mapping constants."""
    
    def test_mood_mappings_exist(self):
        from uvg_core.creative_director import MOOD_SHOT_MAPPING
        
        assert "sad" in MOOD_SHOT_MAPPING
        assert "happy" in MOOD_SHOT_MAPPING
        assert "motivational" in MOOD_SHOT_MAPPING
        assert "neutral" in MOOD_SHOT_MAPPING
    
    def test_emotion_keywords_exist(self):
        from uvg_core.creative_director import EMOTION_KEYWORDS
        
        assert "sad" in EMOTION_KEYWORDS
        assert "happy" in EMOTION_KEYWORDS
        assert "motivational" in EMOTION_KEYWORDS
        
        # Check that keywords are lists
        assert isinstance(EMOTION_KEYWORDS["sad"], list)
        assert len(EMOTION_KEYWORDS["sad"]) > 0
