# uvg_core/style_pack/base.py
"""
Style Pack Base Module.

Loads and manages style pack configurations.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Try to import yaml
try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    logger.warning("PyYAML not installed - style packs will use defaults")

PACKS_DIR = Path(__file__).parent


@dataclass
class StylePack:
    """
    Style pack configuration.
    
    Contains all settings for a consistent visual style.
    """
    name: str
    display_name: str = ""
    
    # Color grading
    lut_path: str = ""
    color_palette: str = "warm"
    
    # Transitions
    transitions: List[str] = field(default_factory=lambda: ["fade"])
    transition_duration: float = 0.5
    
    # Captions
    caption_style: str = "youtube"
    caption_animation: str = "fade_slide"
    font_family: str = "Arial"
    font_size: int = 48
    
    # Pacing
    pacing_factor: float = 1.0
    motion_curve: str = "slow_zoom"
    
    # Camera
    camera_motions: List[str] = field(default_factory=lambda: ["slow-zoom-in"])
    
    # Thumbnail
    thumbnail_style: str = "tiktok"
    
    # Audio
    music_volume: float = 0.15
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "lut_path": self.lut_path,
            "color_palette": self.color_palette,
            "transitions": self.transitions,
            "transition_duration": self.transition_duration,
            "caption_style": self.caption_style,
            "caption_animation": self.caption_animation,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "pacing_factor": self.pacing_factor,
            "motion_curve": self.motion_curve,
            "camera_motions": self.camera_motions,
            "thumbnail_style": self.thumbnail_style,
            "music_volume": self.music_volume,
        }


# Default style packs (built-in without files)
DEFAULT_PACKS = {
    "cinematic": StylePack(
        name="cinematic",
        display_name="Cinematic",
        lut_path="luts/kodak_2395.cube",
        color_palette="cool",
        transitions=["blur_dissolve", "zoom_through", "fade"],
        transition_duration=0.8,
        caption_style="elegant",
        caption_animation="fade_slide",
        font_family="Georgia",
        font_size=44,
        pacing_factor=1.2,
        motion_curve="slow_zoom",
        camera_motions=["slow-zoom-in", "slow-zoom-out", "pan-left"],
        thumbnail_style="cinematic",
        music_volume=0.12,
    ),
    "motivational": StylePack(
        name="motivational",
        display_name="Motivational",
        lut_path="luts/orange_teal.cube",
        color_palette="warm",
        transitions=["zoom_through", "whip_pan", "fade"],
        transition_duration=0.5,
        caption_style="bold",
        caption_animation="pop",
        font_family="Impact",
        font_size=54,
        pacing_factor=0.9,
        motion_curve="dynamic",
        camera_motions=["zoom-in", "tilt-up", "pan-right"],
        thumbnail_style="motivational",
        music_volume=0.18,
    ),
    "corporate": StylePack(
        name="corporate",
        display_name="Corporate",
        lut_path="luts/neutral.cube",
        color_palette="neutral",
        transitions=["fade", "dissolve"],
        transition_duration=0.6,
        caption_style="clean",
        caption_animation="fade_in",
        font_family="Arial",
        font_size=42,
        pacing_factor=1.1,
        motion_curve="subtle",
        camera_motions=["static", "slow-zoom-in"],
        thumbnail_style="professional",
        music_volume=0.10,
    ),
    "neon": StylePack(
        name="neon",
        display_name="Neon/Gaming",
        lut_path="luts/vibrant.cube",
        color_palette="vibrant",
        transitions=["whip_pan", "spin_fade", "zoom_through"],
        transition_duration=0.4,
        caption_style="gaming",
        caption_animation="bounce",
        font_family="Bebas Neue",
        font_size=58,
        pacing_factor=0.8,
        motion_curve="fast",
        camera_motions=["zoom-in", "shake", "whip-pan"],
        thumbnail_style="gaming",
        music_volume=0.20,
    ),
    "documentary": StylePack(
        name="documentary",
        display_name="Documentary",
        lut_path="luts/nature_green.cube",
        color_palette="natural",
        transitions=["fade", "dissolve"],
        transition_duration=1.0,
        caption_style="minimal",
        caption_animation="fade_in",
        font_family="Helvetica",
        font_size=40,
        pacing_factor=1.3,
        motion_curve="slow_zoom",
        camera_motions=["slow-zoom-in", "static", "pan-left"],
        thumbnail_style="documentary",
        music_volume=0.08,
    ),
}


def load_style_pack(name: str) -> StylePack:
    """
    Load a style pack by name.
    
    Args:
        name: Pack name (e.g., "cinematic", "motivational")
        
    Returns:
        StylePack instance
        
    Raises:
        ValueError if pack not found
    """
    # Check for file-based pack first
    pack_dir = PACKS_DIR / name
    config_path = pack_dir / "config.yaml"
    
    if config_path.exists() and HAVE_YAML:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check for LUT file
            lut_path = pack_dir / "lut.cube"
            if lut_path.exists():
                config["lut_path"] = str(lut_path)
            
            # Create pack from config
            pack = StylePack(
                name=name,
                display_name=config.get("display_name", name.title()),
                lut_path=config.get("lut_path", ""),
                color_palette=config.get("color_palette", "warm"),
                transitions=config.get("transitions", ["fade"]),
                transition_duration=config.get("transition_duration", 0.5),
                caption_style=config.get("caption_style", "youtube"),
                caption_animation=config.get("caption_animation", "fade_slide"),
                font_family=config.get("font_family", "Arial"),
                font_size=config.get("font_size", 48),
                pacing_factor=config.get("pacing_factor", 1.0),
                motion_curve=config.get("motion_curve", "slow_zoom"),
                camera_motions=config.get("camera_motions", ["slow-zoom-in"]),
                thumbnail_style=config.get("thumbnail_style", "tiktok"),
                music_volume=config.get("music_volume", 0.15),
            )
            logger.debug(f"Loaded style pack from file: {name}")
            return pack
            
        except Exception as e:
            logger.warning(f"Failed to load pack {name} from file: {e}")
    
    # Fall back to built-in packs
    if name in DEFAULT_PACKS:
        logger.debug(f"Using built-in style pack: {name}")
        return DEFAULT_PACKS[name]
    
    # Default fallback
    logger.warning(f"Style pack '{name}' not found, using cinematic")
    return DEFAULT_PACKS["cinematic"]


def list_style_packs() -> List[str]:
    """
    List all available style packs.
    
    Returns:
        List of pack names
    """
    packs = set(DEFAULT_PACKS.keys())
    
    # Add file-based packs
    for item in PACKS_DIR.iterdir():
        if item.is_dir() and (item / "config.yaml").exists():
            packs.add(item.name)
    
    return sorted(list(packs))


def get_default_pack() -> StylePack:
    """Get the default style pack (cinematic)."""
    return DEFAULT_PACKS["cinematic"]


def get_pack_for_emotion(emotion: str) -> StylePack:
    """
    Get recommended pack for an emotion.
    
    Args:
        emotion: Emotion like "happy", "sad", "inspiring"
        
    Returns:
        Recommended StylePack
    """
    emotion_map = {
        "happy": "motivational",
        "exciting": "neon",
        "inspiring": "motivational",
        "motivational": "motivational",
        "sad": "cinematic",
        "melancholic": "cinematic",
        "calm": "documentary",
        "peaceful": "documentary",
        "professional": "corporate",
        "business": "corporate",
        "energetic": "neon",
        "gaming": "neon",
    }
    
    pack_name = emotion_map.get(emotion.lower(), "cinematic")
    return load_style_pack(pack_name)
