# uvg_core/voice_presets.py
"""
Voice Presets for Fish-Speech S1.

DECISION: Fish-Speech S1 is the SOLE TTS engine for UVG MAX.
- Eliminates Azure TTS dependency
- Eliminates ElevenLabs API costs  
- 50+ emotion markers, #1 TTS Arena naturalness
- $0 cost, fully local

Word timestamps are auto-generated via Whisper - users never enter them manually.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class VoiceStyle(str, Enum):
    """Available voice style presets."""
    DOCUMENTARY = "documentary"
    MOTIVATIONAL = "motivational"
    CINEMATIC = "cinematic"
    TIKTOK_FAST = "tiktok_fast"
    CALM_NARRATIVE = "calm_narrative"
    ENERGETIC_HYPE = "energetic_hype"
    NEUTRAL = "neutral"


@dataclass
class VoicePreset:
    """
    Fish-Speech S1 voice preset configuration.
    
    Optimal parameter ranges:
    - temperature: 0.4-0.5 (creativity vs consistency)
    - top_p: 0.85-0.9 (diversity control)
    - repetition_penalty: 1.1-1.3 (prevent loops)
    - speed: 0.8-1.3 (playback rate)
    """
    name: str
    emotion: str  # Fish-Speech emotion marker
    temperature: float
    top_p: float
    speed: float
    repetition_penalty: float
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "emotion": self.emotion,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "speed": self.speed,
            "repetition_penalty": self.repetition_penalty,
            "description": self.description,
        }
    
    def to_fish_speech_params(self) -> Dict[str, Any]:
        """Get parameters for Fish-Speech S1 API call."""
        return {
            "emotion": self.emotion,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "speed": self.speed,
        }


# =============================================================================
# FISH-SPEECH S1 PRESETS
# =============================================================================

FISH_SPEECH_PRESETS: Dict[str, VoicePreset] = {
    "documentary": VoicePreset(
        name="Documentary",
        emotion="(calm)",
        temperature=0.4,
        top_p=0.85,
        speed=1.0,
        repetition_penalty=1.2,
        description="Professional, informative narration style"
    ),
    "motivational": VoicePreset(
        name="Motivational",
        emotion="(excited)",
        temperature=0.5,
        top_p=0.9,
        speed=1.15,
        repetition_penalty=1.2,
        description="Uplifting, inspiring delivery"
    ),
    "cinematic": VoicePreset(
        name="Cinematic",
        emotion="(dramatic)",
        temperature=0.5,
        top_p=0.85,
        speed=0.95,
        repetition_penalty=1.3,
        description="Epic, theatrical narration"
    ),
    "tiktok_fast": VoicePreset(
        name="TikTok Fast",
        emotion="(excited, in a hurry tone)",
        temperature=0.45,
        top_p=0.9,
        speed=1.3,
        repetition_penalty=1.1,
        description="Quick-paced, attention-grabbing style"
    ),
    "calm_narrative": VoicePreset(
        name="Calm Narrative",
        emotion="(peaceful, storytelling)",
        temperature=0.35,
        top_p=0.8,
        speed=0.9,
        repetition_penalty=1.25,
        description="Soothing, bedtime story style"
    ),
    "energetic_hype": VoicePreset(
        name="Energetic Hype",
        emotion="(hyped, urgent)",
        temperature=0.55,
        top_p=0.95,
        speed=1.4,
        repetition_penalty=1.05,
        description="High-energy, hype-building delivery"
    ),
    "neutral": VoicePreset(
        name="Neutral",
        emotion="(neutral)",
        temperature=0.4,
        top_p=0.85,
        speed=1.0,
        repetition_penalty=1.15,
        description="Clean, neutral narration"
    ),
}


# =============================================================================
# FISH-SPEECH EMOTION MARKERS (50+)
# =============================================================================

FISH_SPEECH_EMOTIONS: List[str] = [
    # Basic emotions
    "(neutral)", "(calm)", "(happy)", "(sad)", "(angry)", "(fearful)",
    "(surprised)", "(disgusted)",
    
    # Intensity modifiers
    "(excited)", "(very excited)", "(slightly excited)",
    "(whispered)", "(loud)", "(very loud)",
    
    # Speaking styles
    "(dramatic)", "(storytelling)", "(professional)", "(casual)",
    "(formal)", "(informal)", "(friendly)", "(serious)",
    
    # Mood combinations
    "(peaceful, storytelling)", "(excited, in a hurry tone)",
    "(hyped, urgent)", "(calm, professional)",
    "(sad, reflective)", "(happy, energetic)",
    
    # Performance styles
    "(narration)", "(conversational)", "(announcement)",
    "(documentary)", "(news anchor)", "(podcast host)",
    
    # Emotional depth
    "(empathetic)", "(sympathetic)", "(enthusiastic)",
    "(confident)", "(hesitant)", "(thoughtful)",
    "(melancholic)", "(hopeful)", "(determined)",
    
    # Special effects
    "(sarcastic)", "(ironic)", "(playful)", "(mysterious)",
    "(intense)", "(relaxed)", "(contemplative)",
]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_preset(style: str) -> VoicePreset:
    """
    Get voice preset by name.
    
    Args:
        style: Preset name (documentary, motivational, etc.)
        
    Returns:
        VoicePreset object (falls back to neutral if not found)
    """
    style_lower = style.lower().replace("-", "_").replace(" ", "_")
    
    if style_lower in FISH_SPEECH_PRESETS:
        return FISH_SPEECH_PRESETS[style_lower]
    
    # Fallback to neutral
    return FISH_SPEECH_PRESETS["neutral"]


def list_presets() -> List[str]:
    """List all available voice presets."""
    return list(FISH_SPEECH_PRESETS.keys())


def list_emotions() -> List[str]:
    """List all available Fish-Speech emotion markers."""
    return FISH_SPEECH_EMOTIONS.copy()


def get_preset_info(style: str) -> Dict[str, Any]:
    """Get detailed info about a preset."""
    preset = get_preset(style)
    return preset.to_dict()


def merge_preset_with_overrides(
    style: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get preset with optional parameter overrides.
    
    Args:
        style: Base preset name
        overrides: Optional dict of parameters to override
        
    Returns:
        Merged preset dict ready for Fish-Speech S1
    """
    preset = get_preset(style).to_fish_speech_params()
    
    if overrides:
        # Only allow valid overrides
        valid_keys = {"emotion", "temperature", "top_p", "speed", "repetition_penalty"}
        for key, value in overrides.items():
            if key in valid_keys:
                preset[key] = value
    
    return preset


def validate_emotion(emotion: str) -> bool:
    """Check if emotion marker is valid."""
    return emotion in FISH_SPEECH_EMOTIONS or emotion.startswith("(")


def get_style_for_platform(platform: str) -> str:
    """
    Get recommended voice style for target platform.
    
    Args:
        platform: youtube, tiktok, instagram, linkedin, etc.
        
    Returns:
        Recommended preset name
    """
    platform_styles = {
        "youtube": "documentary",
        "youtube_shorts": "energetic_hype",
        "tiktok": "tiktok_fast",
        "instagram": "tiktok_fast",
        "instagram_reels": "energetic_hype",
        "linkedin": "documentary",
        "twitter": "tiktok_fast",
        "podcast": "calm_narrative",
        "cinematic": "cinematic",
        "documentary": "documentary",
        "motivational": "motivational",
    }
    
    return platform_styles.get(platform.lower(), "neutral")
