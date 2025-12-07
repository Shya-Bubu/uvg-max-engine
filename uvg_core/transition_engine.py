"""
UVG MAX Transition Engine Module

Context-aware premium transitions.
"""

import logging
import subprocess
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransitionSpec:
    """Specification for a transition."""
    type: str
    duration: float
    params: Dict[str, Any]


# Transition type definitions
TRANSITIONS = {
    "fade": {
        "ffmpeg_filter": "xfade=transition=fade:duration={duration}:offset={offset}",
        "description": "Simple crossfade",
        "mood": ["calm", "peaceful", "neutral"],
    },
    "dissolve": {
        "ffmpeg_filter": "xfade=transition=dissolve:duration={duration}:offset={offset}",
        "description": "Film dissolve",
        "mood": ["nostalgic", "romantic", "calm"],
    },
    "wipeleft": {
        "ffmpeg_filter": "xfade=transition=wipeleft:duration={duration}:offset={offset}",
        "description": "Wipe left",
        "mood": ["energetic", "dynamic"],
    },
    "wiperight": {
        "ffmpeg_filter": "xfade=transition=wiperight:duration={duration}:offset={offset}",
        "description": "Wipe right", 
        "mood": ["energetic", "dynamic"],
    },
    "slideup": {
        "ffmpeg_filter": "xfade=transition=slideup:duration={duration}:offset={offset}",
        "description": "Slide up",
        "mood": ["uplifting", "inspirational"],
    },
    "slidedown": {
        "ffmpeg_filter": "xfade=transition=slidedown:duration={duration}:offset={offset}",
        "description": "Slide down",
        "mood": ["calm", "resolution"],
    },
    "circlecrop": {
        "ffmpeg_filter": "xfade=transition=circlecrop:duration={duration}:offset={offset}",
        "description": "Circle crop",
        "mood": ["focus", "dramatic"],
    },
    "radial": {
        "ffmpeg_filter": "xfade=transition=radial:duration={duration}:offset={offset}",
        "description": "Radial wipe",
        "mood": ["epic", "dramatic"],
    },
    "smoothleft": {
        "ffmpeg_filter": "xfade=transition=smoothleft:duration={duration}:offset={offset}",
        "description": "Smooth left",
        "mood": ["modern", "tech"],
    },
    "smoothright": {
        "ffmpeg_filter": "xfade=transition=smoothright:duration={duration}:offset={offset}",
        "description": "Smooth right",
        "mood": ["modern", "tech"],
    },
    "pixelize": {
        "ffmpeg_filter": "xfade=transition=pixelize:duration={duration}:offset={offset}",
        "description": "Pixelize",
        "mood": ["glitch", "tech", "tense"],
    },
    "diagtl": {
        "ffmpeg_filter": "xfade=transition=diagtl:duration={duration}:offset={offset}",
        "description": "Diagonal top-left",
        "mood": ["dynamic", "energetic"],
    },
    "zoomin": {
        "ffmpeg_filter": "xfade=transition=zoomin:duration={duration}:offset={offset}",
        "description": "Zoom in",
        "mood": ["focus", "intense"],
    },
}


class TransitionEngine:
    """
    Context-aware transition selection and application.
    
    Selects transitions based on:
    - Motion direction
    - Mood/emotion
    - VFX profile
    - Shot types
    """
    
    DEFAULT_DURATION = 0.5
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize transition engine.
        
        Args:
            output_dir: Output directory for processed clips
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/transitions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def select_transition(self,
                          motion_dir: str = "",
                          mood: str = "neutral",
                          vfx_profile: str = "cinematic",
                          tension: float = 0.5) -> TransitionSpec:
        """
        Select appropriate transition based on context.
        
        Args:
            motion_dir: Camera motion direction (pan_left, zoom_in, etc.)
            mood: Scene mood
            vfx_profile: VFX profile
            tension: Tension level 0-1
            
        Returns:
            TransitionSpec with selected transition
        """
        # Match motion direction
        motion_map = {
            "pan-left": "wipeleft",
            "pan-right": "wiperight",
            "zoom-in": "zoomin",
            "slow-zoom-in": "fade",
            "slow-zoom-out": "dissolve",
            "tilt-up": "slideup",
            "tilt-down": "slidedown",
        }
        
        # Select based on motion first
        if motion_dir in motion_map:
            trans_type = motion_map[motion_dir]
        # Then by mood
        elif mood in ["calm", "peaceful", "romantic"]:
            trans_type = "dissolve"
        elif mood in ["energetic", "dynamic", "tense"]:
            trans_type = "wipeleft" if hash(mood) % 2 == 0 else "pixelize"
        elif mood in ["epic", "dramatic"]:
            trans_type = "radial"
        elif mood in ["tech", "modern"]:
            trans_type = "smoothleft"
        else:
            trans_type = "fade"
        
        # Adjust duration by tension
        if tension > 0.7:
            duration = 0.3  # Fast cuts for high tension
        elif tension < 0.3:
            duration = 0.8  # Slow transitions for calm
        else:
            duration = 0.5
        
        return TransitionSpec(
            type=trans_type,
            duration=duration,
            params=TRANSITIONS.get(trans_type, TRANSITIONS["fade"])
        )
    
    def get_ffmpeg_filter(self,
                          transition: TransitionSpec,
                          offset: float) -> str:
        """
        Get FFmpeg filter string for transition.
        
        Args:
            transition: Transition specification
            offset: Offset time in seconds
            
        Returns:
            FFmpeg filter string
        """
        template = TRANSITIONS.get(transition.type, TRANSITIONS["fade"])["ffmpeg_filter"]
        return template.format(duration=transition.duration, offset=offset)
    
    def apply_transition(self,
                          clip1_path: str,
                          clip2_path: str,
                          transition: TransitionSpec,
                          output_path: Optional[str] = None) -> str:
        """
        Apply transition between two clips.
        
        Args:
            clip1_path: First clip path
            clip2_path: Second clip path
            transition: Transition specification
            output_path: Output path
            
        Returns:
            Path to output clip
        """
        if output_path is None:
            output_path = self.output_dir / f"trans_{Path(clip1_path).stem}_{Path(clip2_path).stem}.mp4"
        
        # Get clip1 duration for offset calculation
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                clip1_path
            ], capture_output=True, text=True, timeout=30)
            
            clip1_duration = float(result.stdout.strip())
            offset = clip1_duration - transition.duration
            
        except Exception:
            offset = 3.0  # Default
        
        filter_str = self.get_ffmpeg_filter(transition, offset)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip1_path,
            "-i", clip2_path,
            "-filter_complex", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            return str(output_path)
        except Exception as e:
            logger.warning(f"Transition failed: {e}")
            return clip1_path
    
    def build_transition_chain(self,
                                clips: List[str],
                                transitions: List[TransitionSpec]) -> str:
        """
        Build a chain of clips with transitions.
        
        Args:
            clips: List of clip paths
            transitions: List of transitions between clips
            
        Returns:
            FFmpeg filter_complex string
        """
        if len(clips) < 2:
            return ""
        
        # Build filter chain
        filter_parts = []
        current_input = "[0:v]"
        
        for i in range(1, len(clips)):
            trans = transitions[i-1] if i-1 < len(transitions) else TransitionSpec("fade", 0.5, {})
            
            # Calculate offset (need to get duration of current accumulated output)
            # This is simplified - in practice you'd need to track accumulated duration
            offset = 3.0 * i  # Placeholder
            
            filter_str = self.get_ffmpeg_filter(trans, offset)
            filter_parts.append(f"{current_input}[{i}:v]{filter_str}[v{i}]")
            current_input = f"[v{i}]"
        
        return ";".join(filter_parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_transition(mood: str = "neutral", tension: float = 0.5) -> TransitionSpec:
    """Select appropriate transition."""
    engine = TransitionEngine()
    return engine.select_transition(mood=mood, tension=tension)


def apply_transition(clip1: str, clip2: str, trans_type: str = "fade") -> str:
    """Apply transition between clips."""
    engine = TransitionEngine()
    spec = TransitionSpec(type=trans_type, duration=0.5, params={})
    return engine.apply_transition(clip1, clip2, spec)
