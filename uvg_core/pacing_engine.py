"""
UVG MAX Pacing Engine Module

Beat-level editing control for professional flow.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PacingStyle(Enum):
    """Pacing style presets."""
    SLOW = "slow"          # Documentary style
    MEDIUM = "medium"      # Standard
    FAST = "fast"          # TikTok style
    DYNAMIC = "dynamic"    # Varies with tension


@dataclass
class PacingPoint:
    """A pacing point in the timeline."""
    time_ms: int
    beat_type: str  # "cut", "transition", "beat_hit"
    intensity: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_ms": self.time_ms,
            "beat_type": self.beat_type,
            "intensity": self.intensity,
        }


class PacingEngine:
    """
    Beat-level editing and pacing control.
    
    Features:
    - Tension-based pacing
    - Beat detection sync
    - Shot duration optimization
    - Cut timing prediction
    """
    
    # Beat timing multipliers
    PACING_MULTIPLIERS = {
        PacingStyle.SLOW: 1.5,
        PacingStyle.MEDIUM: 1.0,
        PacingStyle.FAST: 0.6,
        PacingStyle.DYNAMIC: 1.0,  # Varies
    }
    
    def __init__(self, style: PacingStyle = PacingStyle.MEDIUM):
        """
        Initialize pacing engine.
        
        Args:
            style: Default pacing style
        """
        self.style = style
    
    def calculate_scene_duration(self,
                                   text: str,
                                   tension: float = 0.5,
                                   min_duration: float = 2.0,
                                   max_duration: float = 8.0) -> float:
        """
        Calculate optimal scene duration based on content.
        
        Args:
            text: Scene narration text
            tension: Tension level 0-1
            min_duration: Minimum duration
            max_duration: Maximum duration
            
        Returns:
            Optimal duration in seconds
        """
        # Base duration from word count (150 WPM speaking rate)
        words = len(text.split())
        base_duration = words / 2.5  # ~2.5 words per second
        
        # Apply pacing multiplier
        if self.style == PacingStyle.DYNAMIC:
            # Higher tension = faster pacing
            multiplier = 1.3 - (tension * 0.6)  # 1.3 at tension=0, 0.7 at tension=1
        else:
            multiplier = self.PACING_MULTIPLIERS[self.style]
        
        duration = base_duration * multiplier
        
        # Clamp to limits
        return max(min_duration, min(max_duration, duration))
    
    def optimize_cut_timing(self,
                             scene_duration: float,
                             tension: float,
                             beat_times: List[float] = None) -> List[float]:
        """
        Optimize cut timing for professional feel.
        
        Args:
            scene_duration: Scene duration
            tension: Tension level
            beat_times: Optional beat times from music
            
        Returns:
            Suggested cut times within scene
        """
        cuts = []
        
        # High tension = more frequent cuts
        if tension > 0.7:
            cut_interval = scene_duration / 3
        elif tension > 0.4:
            cut_interval = scene_duration / 2
        else:
            cut_interval = scene_duration  # No internal cuts
        
        current = cut_interval
        while current < scene_duration - 0.5:
            # Snap to nearest beat if available
            if beat_times:
                nearest = min(beat_times, key=lambda t: abs(t - current))
                if abs(nearest - current) < 0.3:
                    current = nearest
            
            cuts.append(current)
            current += cut_interval
        
        return cuts
    
    def generate_pacing_points(self,
                                scenes: List[Dict[str, Any]],
                                beat_times: List[float] = None) -> List[PacingPoint]:
        """
        Generate pacing points for timeline.
        
        Args:
            scenes: List of scene dicts
            beat_times: Music beat times
            
        Returns:
            List of PacingPoint
        """
        points = []
        current_time = 0
        
        for scene in scenes:
            duration = scene.get("duration", 4.0) * 1000  # to ms
            tension = scene.get("tension", 0.5)
            
            # Scene start = cut
            points.append(PacingPoint(
                time_ms=int(current_time),
                beat_type="cut",
                intensity=0.3 + tension * 0.5
            ))
            
            # Add beat hits at music beats
            if beat_times:
                for beat in beat_times:
                    beat_ms = beat * 1000
                    if current_time < beat_ms < current_time + duration:
                        points.append(PacingPoint(
                            time_ms=int(beat_ms),
                            beat_type="beat_hit",
                            intensity=0.5
                        ))
            
            current_time += duration
        
        return points
    
    def adjust_durations_for_music(self,
                                    scenes: List[Dict[str, Any]],
                                    beat_times: List[float],
                                    total_duration: float) -> List[float]:
        """
        Adjust scene durations to fit music beats.
        
        Args:
            scenes: List of scenes
            beat_times: Music beat times
            total_duration: Target total duration
            
        Returns:
            New durations aligned to beats
        """
        if not beat_times:
            return [s.get("duration", 4.0) for s in scenes]
        
        # Find downbeats (every 4th beat typically)
        downbeats = beat_times[::4]
        
        new_durations = []
        target_per_scene = total_duration / len(scenes)
        
        for i, scene in enumerate(scenes):
            target_end = (i + 1) * target_per_scene
            
            # Find nearest downbeat to target end
            nearest_beat = min(downbeats, key=lambda t: abs(t - target_end))
            
            if i == 0:
                duration = nearest_beat
            else:
                duration = nearest_beat - sum(new_durations)
            
            # Clamp to reasonable range
            duration = max(2.0, min(10.0, duration))
            new_durations.append(duration)
        
        return new_durations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_pacing(text: str, 
                     tension: float = 0.5,
                     style: str = "medium") -> float:
    """Calculate optimal duration for text."""
    engine = PacingEngine(PacingStyle(style))
    return engine.calculate_scene_duration(text, tension)
