# uvg_core/scene_emotion.py
"""
Scene Emotion System for UVG MAX.

Central emotion controller that influences:
- SFX intensity and selection
- VFX bloom/contrast/shake
- Music intensity curve
- Color grading adjustments

This creates the "10× more cinematic" experience.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SceneEmotion(str, Enum):
    """Available scene emotions."""
    CALM = "calm"
    EXCITING = "exciting"
    DRAMATIC = "dramatic"
    TENSE = "tense"
    JOYFUL = "joyful"
    SAD = "sad"
    MYSTERIOUS = "mysterious"
    NEUTRAL = "neutral"


@dataclass
class EmotionConfig:
    """Configuration for a scene emotion."""
    name: str
    
    # SFX parameters
    sfx_intensity: float  # 0-1, volume/density of SFX
    sfx_types: List[str]  # Preferred SFX categories
    
    # VFX parameters
    vfx_bloom: float      # 0-1, glow effect strength
    vfx_contrast: float   # 0.8-1.5, contrast multiplier
    vfx_shake: float      # 0-0.5, camera shake intensity
    vfx_saturation: float # 0.8-1.3, color saturation
    vfx_vignette: float   # 0-0.5, vignette strength
    
    # Music parameters
    music_intensity: float    # 0-1, volume curve modifier
    music_tempo_bias: float   # 0.8-1.2, preferred tempo
    
    # Motion parameters
    motion_speed: float   # 0.5-1.5, Ken Burns speed
    
    # Color grading
    color_temperature: str  # warm, neutral, cool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sfx_intensity": self.sfx_intensity,
            "sfx_types": self.sfx_types,
            "vfx_bloom": self.vfx_bloom,
            "vfx_contrast": self.vfx_contrast,
            "vfx_shake": self.vfx_shake,
            "vfx_saturation": self.vfx_saturation,
            "vfx_vignette": self.vfx_vignette,
            "music_intensity": self.music_intensity,
            "music_tempo_bias": self.music_tempo_bias,
            "motion_speed": self.motion_speed,
            "color_temperature": self.color_temperature,
        }


# =============================================================================
# SCENE EMOTION CONFIGURATIONS
# =============================================================================

SCENE_EMOTIONS: Dict[str, EmotionConfig] = {
    "calm": EmotionConfig(
        name="Calm",
        sfx_intensity=0.3,
        sfx_types=["ambient", "nature", "subtle"],
        vfx_bloom=0.2,
        vfx_contrast=1.0,
        vfx_shake=0.0,
        vfx_saturation=1.0,
        vfx_vignette=0.1,
        music_intensity=0.4,
        music_tempo_bias=0.85,
        motion_speed=0.7,
        color_temperature="warm"
    ),
    "exciting": EmotionConfig(
        name="Exciting",
        sfx_intensity=0.8,
        sfx_types=["whoosh", "impact", "energy"],
        vfx_bloom=0.5,
        vfx_contrast=1.2,
        vfx_shake=0.15,
        vfx_saturation=1.2,
        vfx_vignette=0.15,
        music_intensity=0.9,
        music_tempo_bias=1.15,
        motion_speed=1.3,
        color_temperature="neutral"
    ),
    "dramatic": EmotionConfig(
        name="Dramatic",
        sfx_intensity=0.9,
        sfx_types=["epic", "tension", "impact_deep"],
        vfx_bloom=0.7,
        vfx_contrast=1.4,
        vfx_shake=0.1,
        vfx_saturation=1.1,
        vfx_vignette=0.3,
        music_intensity=1.0,
        music_tempo_bias=0.9,
        motion_speed=0.8,
        color_temperature="cool"
    ),
    "tense": EmotionConfig(
        name="Tense",
        sfx_intensity=0.7,
        sfx_types=["tension_riser", "heartbeat", "suspense"],
        vfx_bloom=0.1,
        vfx_contrast=1.3,
        vfx_shake=0.2,
        vfx_saturation=0.9,
        vfx_vignette=0.4,
        music_intensity=0.7,
        music_tempo_bias=1.0,
        motion_speed=0.9,
        color_temperature="cool"
    ),
    "joyful": EmotionConfig(
        name="Joyful",
        sfx_intensity=0.6,
        sfx_types=["bright_ding", "pop_soft", "sparkle"],
        vfx_bloom=0.6,
        vfx_contrast=1.1,
        vfx_shake=0.0,
        vfx_saturation=1.25,
        vfx_vignette=0.05,
        music_intensity=0.8,
        music_tempo_bias=1.1,
        motion_speed=1.1,
        color_temperature="warm"
    ),
    "sad": EmotionConfig(
        name="Sad",
        sfx_intensity=0.3,
        sfx_types=["sad_tone", "rain", "subtle"],
        vfx_bloom=0.3,
        vfx_contrast=0.95,
        vfx_shake=0.0,
        vfx_saturation=0.85,
        vfx_vignette=0.35,
        music_intensity=0.5,
        music_tempo_bias=0.8,
        motion_speed=0.6,
        color_temperature="cool"
    ),
    "mysterious": EmotionConfig(
        name="Mysterious",
        sfx_intensity=0.5,
        sfx_types=["ambient_dark", "tension_subtle", "whisper"],
        vfx_bloom=0.15,
        vfx_contrast=1.2,
        vfx_shake=0.05,
        vfx_saturation=0.9,
        vfx_vignette=0.4,
        music_intensity=0.6,
        music_tempo_bias=0.9,
        motion_speed=0.75,
        color_temperature="cool"
    ),
    "neutral": EmotionConfig(
        name="Neutral",
        sfx_intensity=0.4,
        sfx_types=["subtle", "transition"],
        vfx_bloom=0.2,
        vfx_contrast=1.0,
        vfx_shake=0.0,
        vfx_saturation=1.0,
        vfx_vignette=0.1,
        music_intensity=0.5,
        music_tempo_bias=1.0,
        motion_speed=1.0,
        color_temperature="neutral"
    ),
}


# =============================================================================
# SCENE EMOTION CONTROLLER
# =============================================================================

class SceneEmotionController:
    """
    Central controller for scene-level emotion effects.
    
    Usage:
        controller = SceneEmotionController()
        config = controller.get_emotion_config("dramatic")
        
        # Apply to SFX
        sfx_volume = config.sfx_intensity * base_volume
        
        # Apply to VFX
        bloom_strength = config.vfx_bloom
        
        # Apply to music
        music_volume = config.music_intensity * base_music
    """
    
    def __init__(self):
        self.emotions = SCENE_EMOTIONS
    
    def get_emotion_config(self, emotion: str) -> EmotionConfig:
        """
        Get configuration for an emotion.
        
        Args:
            emotion: Emotion name (calm, exciting, dramatic, etc.)
            
        Returns:
            EmotionConfig object
        """
        emotion_lower = emotion.lower().strip()
        
        if emotion_lower in self.emotions:
            return self.emotions[emotion_lower]
        
        logger.warning(f"Unknown emotion '{emotion}', using neutral")
        return self.emotions["neutral"]
    
    def get_sfx_params(self, emotion: str) -> Dict[str, Any]:
        """Get SFX parameters for emotion."""
        config = self.get_emotion_config(emotion)
        return {
            "intensity": config.sfx_intensity,
            "types": config.sfx_types,
            "volume_db": -12 + (config.sfx_intensity * 6),  # -12 to -6 dB
        }
    
    def get_vfx_params(self, emotion: str) -> Dict[str, Any]:
        """Get VFX parameters for emotion."""
        config = self.get_emotion_config(emotion)
        return {
            "bloom": config.vfx_bloom,
            "contrast": config.vfx_contrast,
            "shake": config.vfx_shake,
            "saturation": config.vfx_saturation,
            "vignette": config.vfx_vignette,
        }
    
    def get_music_params(self, emotion: str) -> Dict[str, Any]:
        """Get music parameters for emotion."""
        config = self.get_emotion_config(emotion)
        return {
            "intensity": config.music_intensity,
            "tempo_bias": config.music_tempo_bias,
            "volume_multiplier": config.music_intensity,
        }
    
    def get_motion_params(self, emotion: str) -> Dict[str, Any]:
        """Get motion effect parameters for emotion."""
        config = self.get_emotion_config(emotion)
        return {
            "speed": config.motion_speed,
            "shake": config.vfx_shake,
        }
    
    def get_color_params(self, emotion: str) -> Dict[str, Any]:
        """Get color grading parameters for emotion."""
        config = self.get_emotion_config(emotion)
        
        # Temperature to FFmpeg colortemperature filter value
        temp_values = {
            "warm": 6500,    # Warmer
            "neutral": 5500,  # Neutral daylight
            "cool": 4500,    # Cooler
        }
        
        return {
            "temperature": temp_values.get(config.color_temperature, 5500),
            "saturation": config.vfx_saturation,
            "contrast": config.vfx_contrast,
        }
    
    def get_transition_for_emotion_change(
        self,
        from_emotion: str,
        to_emotion: str
    ) -> str:
        """
        Get recommended transition based on emotion change.
        
        Args:
            from_emotion: Starting emotion
            to_emotion: Ending emotion
            
        Returns:
            Transition type
        """
        from_config = self.get_emotion_config(from_emotion)
        to_config = self.get_emotion_config(to_emotion)
        
        intensity_change = to_config.music_intensity - from_config.music_intensity
        
        if intensity_change > 0.3:
            # Big energy increase
            return "whip_pan"
        elif intensity_change < -0.3:
            # Big energy decrease
            return "film_dissolve"
        elif to_config.name in ["Dramatic", "Tense"]:
            return "cross_zoom"
        elif to_config.name in ["Calm", "Sad"]:
            return "fade"
        else:
            return "fade"
    
    def list_emotions(self) -> List[str]:
        """List all available emotions."""
        return list(self.emotions.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_controller: Optional[SceneEmotionController] = None


def get_controller() -> SceneEmotionController:
    """Get global scene emotion controller."""
    global _controller
    if _controller is None:
        _controller = SceneEmotionController()
    return _controller


def get_emotion_config(emotion: str) -> EmotionConfig:
    """Get emotion configuration."""
    return get_controller().get_emotion_config(emotion)


def get_sfx_for_emotion(emotion: str) -> Dict[str, Any]:
    """Get SFX parameters for emotion."""
    return get_controller().get_sfx_params(emotion)


def get_vfx_for_emotion(emotion: str) -> Dict[str, Any]:
    """Get VFX parameters for emotion."""
    return get_controller().get_vfx_params(emotion)


def get_music_for_emotion(emotion: str) -> Dict[str, Any]:
    """Get music parameters for emotion."""
    return get_controller().get_music_params(emotion)


def list_emotions() -> List[str]:
    """List all available emotions."""
    return get_controller().list_emotions()
