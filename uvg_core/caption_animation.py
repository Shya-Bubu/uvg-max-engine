"""
UVG MAX Caption Animation Module

Advanced caption animations with word-by-word effects.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AnimationType(Enum):
    """Caption animation types."""
    FADE_SLIDE = "fade_slide"
    POP_BOUNCE = "pop_bounce"
    TYPEWRITER = "typewriter"
    SCALE_BOUNCE = "scale_bounce"
    FADE = "fade"
    WAVE = "wave"
    GLITCH = "glitch"


@dataclass
class WordAnimation:
    """Animation for a single word."""
    word: str
    start_ms: int
    end_ms: int
    animation: AnimationType
    color: str = "#FFFFFF"
    highlight: bool = False


@dataclass
class CaptionFrame:
    """Single frame of caption animation."""
    time_ms: int
    text: str
    x: int
    y: int
    scale: float = 1.0
    opacity: float = 1.0
    rotation: float = 0.0


class CaptionAnimation:
    """
    Advanced caption animations.
    
    Features:
    - Word-by-word highlights
    - Bounce/pop entrance
    - Typewriter effect
    - Color emphasis on keywords
    """
    
    ANIMATION_CONFIGS = {
        AnimationType.FADE_SLIDE: {
            "entrance_ms": 200,
            "exit_ms": 150,
            "slide_distance": 20,
        },
        AnimationType.POP_BOUNCE: {
            "entrance_ms": 150,
            "bounce_scale": 1.2,
            "settle_ms": 100,
        },
        AnimationType.TYPEWRITER: {
            "char_delay_ms": 50,
        },
        AnimationType.SCALE_BOUNCE: {
            "entrance_ms": 200,
            "max_scale": 1.3,
            "settle_scale": 1.0,
        },
        AnimationType.FADE: {
            "fade_duration_ms": 300,
        },
        AnimationType.WAVE: {
            "wave_amplitude": 10,
            "wave_frequency": 2,
        },
        AnimationType.GLITCH: {
            "glitch_probability": 0.3,
            "offset_range": 5,
        },
    }
    
    def __init__(self, 
                 default_animation: AnimationType = AnimationType.FADE_SLIDE,
                 highlight_color: str = "#FFFF00"):
        """
        Initialize caption animation.
        
        Args:
            default_animation: Default animation type
            highlight_color: Color for word highlights
        """
        self.default_animation = default_animation
        self.highlight_color = highlight_color
    
    def generate_word_animations(self,
                                   words: List[Dict[str, Any]],
                                   animation: AnimationType = None) -> List[WordAnimation]:
        """
        Generate animations for words.
        
        Args:
            words: List of word dicts with start_ms, end_ms, word
            animation: Animation type
            
        Returns:
            List of WordAnimation
        """
        animation = animation or self.default_animation
        
        animations = []
        for w in words:
            animations.append(WordAnimation(
                word=w.get("word", ""),
                start_ms=w.get("start_ms", 0),
                end_ms=w.get("end_ms", 0),
                animation=animation,
            ))
        
        return animations
    
    def get_ass_animation_tags(self,
                                animation: AnimationType,
                                is_entrance: bool = True) -> str:
        """
        Get ASS subtitle animation tags.
        
        Args:
            animation: Animation type
            is_entrance: True for entrance, False for exit
            
        Returns:
            ASS override tags string
        """
        config = self.ANIMATION_CONFIGS[animation]
        
        if animation == AnimationType.FADE_SLIDE:
            if is_entrance:
                return "{\\fad(200,0)\\move(0,-20,0,0)}"
            return "{\\fad(0,150)}"
        
        elif animation == AnimationType.POP_BOUNCE:
            if is_entrance:
                return "{\\t(0,100,\\fscx120\\fscy120)\\t(100,200,\\fscx100\\fscy100)}"
            return "{\\fad(0,100)}"
        
        elif animation == AnimationType.SCALE_BOUNCE:
            if is_entrance:
                return "{\\fscx0\\fscy0\\t(0,150,\\fscx110\\fscy110)\\t(150,250,\\fscx100\\fscy100)}"
            return "{\\t(0,100,\\alpha&HFF&)}"
        
        elif animation == AnimationType.FADE:
            return "{\\fad(300,300)}"
        
        return ""
    
    def generate_karaoke_effect(self,
                                  words: List[Dict[str, Any]],
                                  fill_color: str = "FFFFFF",
                                  highlight_color: str = "00FFFF") -> str:
        """
        Generate ASS karaoke tags for word highlighting.
        
        Args:
            words: Word timings
            fill_color: Default text color
            highlight_color: Highlight color
            
        Returns:
            ASS-formatted text with karaoke
        """
        parts = []
        
        for w in words:
            duration_cs = (w.get("end_ms", 0) - w.get("start_ms", 0)) // 10
            word = w.get("word", "")
            
            # Karaoke highlight effect
            parts.append(f"{{\\kf{duration_cs}}}{word} ")
        
        return "".join(parts).strip()
    
    def create_drawtext_animation(self,
                                    text: str,
                                    animation: AnimationType,
                                    position: tuple = (540, 1400),
                                    duration_ms: int = 3000) -> Dict[str, Any]:
        """
        Create FFmpeg drawtext animation parameters.
        
        Args:
            text: Text to animate
            animation: Animation type
            position: (x, y) position
            duration_ms: Total duration
            
        Returns:
            Dict with FFmpeg filter parameters
        """
        x, y = position
        config = self.ANIMATION_CONFIGS[animation]
        
        if animation == AnimationType.FADE_SLIDE:
            entrance = config["entrance_ms"] / 1000
            slide = config["slide_distance"]
            
            return {
                "text": text,
                "x": x,
                "y_expr": f"{y}+{slide}*(1-min(t/{entrance},1))",
                "alpha_expr": f"if(lt(t,{entrance}),t/{entrance},1)",
            }
        
        elif animation == AnimationType.POP_BOUNCE:
            entrance = config["entrance_ms"] / 1000
            bounce = config["bounce_scale"]
            
            return {
                "text": text,
                "x": x,
                "y": y,
                "fontsize_expr": f"fontsize*(if(lt(t,{entrance}),1+({bounce}-1)*sin(t/{entrance}*3.14159),1))",
            }
        
        return {
            "text": text,
            "x": x,
            "y": y,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def animate_captions(words: List[Dict], 
                     style: str = "fade_slide") -> List[WordAnimation]:
    """Generate word animations."""
    engine = CaptionAnimation(AnimationType(style))
    return engine.generate_word_animations(words)
