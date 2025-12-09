"""
UVG MAX Kinetic Captions Module

Motion captions with animation presets for premium video production.
Generates FFmpeg filter chains for animated text overlays.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CaptionLayer:
    """A single animated caption layer."""
    text: str
    start_ms: int
    end_ms: int
    animation: str = "fade_in"  # slide_left, slide_right, slide_up, fade_in, pop, bounce, word_reveal
    position: Tuple[int, int] = (0, 0)  # (x, y) - 0,0 means centered
    font_size: int = 48
    font_family: str = "Arial"
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "animation": self.animation,
            "position": self.position,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "color": self.color,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
        }
    
    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0
    
    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0
    
    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


# =============================================================================
# ANIMATION PRESETS
# =============================================================================

ANIMATION_PRESETS = {
    "slide_left": {
        "description": "Text slides in from the right",
        "x_expr": "if(lt(t-{start},0.3),w-(w+tw)*((t-{start})/0.3),(w-tw)/2)",
        "y_expr": "{y}",
        "alpha_expr": "1",
    },
    "slide_right": {
        "description": "Text slides in from the left",
        "x_expr": "if(lt(t-{start},0.3),-tw+(w+tw)*((t-{start})/0.3),(w-tw)/2)",
        "y_expr": "{y}",
        "alpha_expr": "1",
    },
    "slide_up": {
        "description": "Text slides up from bottom",
        "x_expr": "(w-tw)/2",
        "y_expr": "if(lt(t-{start},0.3),h-(h-{y})*((t-{start})/0.3),{y})",
        "alpha_expr": "1",
    },
    "fade_in": {
        "description": "Text fades in",
        "x_expr": "(w-tw)/2",
        "y_expr": "{y}",
        "alpha_expr": "if(lt(t-{start},0.3),(t-{start})/0.3,1)",
    },
    "pop": {
        "description": "Text pops in with scale effect",
        "x_expr": "(w-tw)/2",
        "y_expr": "{y}",
        "alpha_expr": "1",
        "scale_expr": "if(lt(t-{start},0.1),1.5,1)",  # Note: scale requires different approach
    },
    "bounce": {
        "description": "Text bounces in",
        "x_expr": "(w-tw)/2",
        "y_expr": "{y}+sin((t-{start})*15)*30*exp(-(t-{start})*5)",
        "alpha_expr": "1",
    },
    "word_reveal": {
        "description": "Words appear one by one",
        "x_expr": "(w-tw)/2",
        "y_expr": "{y}",
        "alpha_expr": "if(lt(t,{word_start}),0,1)",  # Per-word timing
    },
    "typewriter": {
        "description": "Text appears character by character",
        "x_expr": "(w-tw)/2",
        "y_expr": "{y}",
        "alpha_expr": "1",
        # Uses text slicing based on time
    },
}


# =============================================================================
# CAPTION STYLES (Premium)
# =============================================================================

CAPTION_STYLES = {
    "tiktok": {
        "font_size": 72,
        "font_family": "Impact",
        "color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 4,
        "animation": "pop",
        "y_ratio": 0.7,
    },
    "youtube": {
        "font_size": 56,
        "font_family": "Montserrat",
        "color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 3,
        "animation": "fade_in",
        "y_ratio": 0.85,
    },
    "cinematic": {
        "font_size": 48,
        "font_family": "Georgia",
        "color": "#FFFFFF",
        "stroke_color": "#222222",
        "stroke_width": 2,
        "animation": "slide_up",
        "y_ratio": 0.9,
    },
    "modern": {
        "font_size": 54,
        "font_family": "Helvetica",
        "color": "#FFFFFF",
        "stroke_color": "#333333",
        "stroke_width": 2,
        "animation": "slide_left",
        "y_ratio": 0.8,
    },
    "bold": {
        "font_size": 80,
        "font_family": "Anton",
        "color": "#FFFF00",
        "stroke_color": "#000000",
        "stroke_width": 5,
        "animation": "bounce",
        "y_ratio": 0.65,
    },
}


class KineticCaptions:
    """
    Generate animated caption layers for premium video production.
    
    Features:
    - Time-aligned caption layers from TTS word timings
    - Animation presets: slide, fade, pop, bounce, word-by-word
    - FFmpeg filter chain generation
    - Mock mode for deterministic testing
    """
    
    def __init__(self,
                 video_width: int = 1080,
                 video_height: int = 1920,
                 default_style: str = "youtube",
                 mock_mode: bool = False):
        """
        Initialize kinetic captions.
        
        Args:
            video_width: Video width in pixels
            video_height: Video height in pixels
            default_style: Default caption style preset
            mock_mode: Use deterministic mock generation
        """
        self.video_width = video_width
        self.video_height = video_height
        self.default_style = default_style
        self.mock_mode = mock_mode or os.getenv("UVG_MOCK_MODE", "true").lower() == "true"
    
    def generate_layers(self,
                        word_timings: List[Any],
                        style: str = "",
                        animation: str = "") -> List[CaptionLayer]:
        """
        Generate caption layers from TTS word timings.
        
        Args:
            word_timings: List of WordTiming objects
            style: Caption style preset
            animation: Animation preset override
            
        Returns:
            List of CaptionLayer objects
        """
        style = style or self.default_style
        style_config = CAPTION_STYLES.get(style, CAPTION_STYLES["youtube"])
        animation = animation or style_config.get("animation", "fade_in")
        
        if not word_timings:
            return []
        
        # Group words into caption chunks (max ~6 words per caption)
        chunks = self._chunk_words(word_timings, max_words=6)
        
        layers = []
        y_position = int(self.video_height * style_config["y_ratio"])
        
        for chunk in chunks:
            words = [w.word if hasattr(w, 'word') else w.get("word", "") for w in chunk]
            text = " ".join(words)
            
            # Get timing from first/last word
            start_ms = chunk[0].start_ms if hasattr(chunk[0], 'start_ms') else chunk[0].get("start_ms", 0)
            end_ms = chunk[-1].end_ms if hasattr(chunk[-1], 'end_ms') else chunk[-1].get("end_ms", start_ms + 1000)
            
            layers.append(CaptionLayer(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                animation=animation,
                position=(self.video_width // 2, y_position),
                font_size=style_config["font_size"],
                font_family=style_config["font_family"],
                color=style_config["color"],
                stroke_color=style_config["stroke_color"],
                stroke_width=style_config["stroke_width"],
            ))
        
        logger.info(f"Generated {len(layers)} kinetic caption layers")
        return layers
    
    def _chunk_words(self, word_timings: List[Any], max_words: int = 6) -> List[List[Any]]:
        """Split word timings into displayable chunks."""
        chunks = []
        current_chunk = []
        
        for word in word_timings:
            current_chunk.append(word)
            
            if len(current_chunk) >= max_words:
                chunks.append(current_chunk)
                current_chunk = []
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def to_ffmpeg_filter(self, layers: List[CaptionLayer]) -> str:
        """
        Convert caption layers to FFmpeg filter_complex string.
        
        Args:
            layers: List of CaptionLayer objects
            
        Returns:
            FFmpeg filter string for -filter_complex
        """
        if not layers:
            return "null"
        
        filter_parts = []
        
        for i, layer in enumerate(layers):
            preset = ANIMATION_PRESETS.get(layer.animation, ANIMATION_PRESETS["fade_in"])
            
            # Calculate y position
            y_pos = layer.position[1] if layer.position[1] > 0 else self.video_height * 0.8
            
            # Build expression with timing
            x_expr = preset["x_expr"].format(start=layer.start_sec, y=y_pos)
            y_expr = preset["y_expr"].format(start=layer.start_sec, y=y_pos)
            alpha_expr = preset["alpha_expr"].format(start=layer.start_sec)
            
            # Escape text for FFmpeg
            escaped_text = layer.text.replace("'", "'\\''").replace(":", "\\:")
            
            # Build drawtext filter
            drawtext = (
                f"drawtext="
                f"text='{escaped_text}':"
                f"fontsize={layer.font_size}:"
                f"fontcolor={layer.color}:"
                f"borderw={layer.stroke_width}:"
                f"bordercolor={layer.stroke_color}:"
                f"x='{x_expr}':"
                f"y='{y_expr}':"
                f"alpha='{alpha_expr}':"
                f"enable='between(t,{layer.start_sec},{layer.end_sec})'"
            )
            
            filter_parts.append(drawtext)
        
        # Chain all drawtext filters
        return ",".join(filter_parts)
    
    def generate_mock_layers(self, text: str, duration_ms: int) -> List[CaptionLayer]:
        """
        # MOCK — Generate deterministic caption layers for testing.
        
        Args:
            text: Full text to split into captions
            duration_ms: Total duration in milliseconds
            
        Returns:
            List of CaptionLayer objects
        """
        import random
        
        # Use debug seed for reproducibility
        seed = int(os.getenv("UVG_DEBUG_SEED", "42"))
        random.seed(seed)
        
        words = text.split()
        if not words:
            return []
        
        # Simulate word timings
        time_per_word = duration_ms / len(words)
        mock_timings = []
        current_time = 0
        
        for word in words:
            mock_timings.append({
                "word": word,
                "start_ms": int(current_time),
                "end_ms": int(current_time + time_per_word * 0.9),
            })
            current_time += time_per_word
        
        return self.generate_layers(mock_timings)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_kinetic_captions(word_timings: List[Any],
                               style: str = "youtube") -> List[CaptionLayer]:
    """Generate kinetic caption layers from word timings."""
    engine = KineticCaptions()
    return engine.generate_layers(word_timings, style)


def captions_to_ffmpeg(layers: List[CaptionLayer]) -> str:
    """Convert caption layers to FFmpeg filter string."""
    engine = KineticCaptions()
    return engine.to_ffmpeg_filter(layers)
