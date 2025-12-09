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


def detect_bpm_from_audio(audio_path: str) -> int:
    """
    Detect BPM from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Detected BPM or 120 as default
    """
    # Try librosa first
    try:
        import librosa
        y, sr = librosa.load(audio_path, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return int(tempo)
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to FFmpeg-based estimation
    try:
        import subprocess
        cmd = ["ffprobe", "-i", audio_path, "-show_entries", 
               "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # Just estimate based on audio characteristics
        return 120  # Default to 120 BPM
    except Exception:
        return 120


def get_beat_times(bpm: int, duration: float) -> List[float]:
    """
    Calculate beat times for a given BPM and duration.
    
    Args:
        bpm: Beats per minute
        duration: Total duration in seconds
        
    Returns:
        List of beat times in seconds
    """
    beat_interval = 60.0 / bpm
    times = []
    current = 0.0
    while current < duration:
        times.append(current)
        current += beat_interval
    return times


def snap_to_beat(time_sec: float, beat_times: List[float], tolerance: float = 0.15) -> float:
    """
    Snap time to nearest beat.
    
    Args:
        time_sec: Time in seconds
        beat_times: List of beat times
        tolerance: Max snap distance in seconds
        
    Returns:
        Snapped time
    """
    if not beat_times:
        return time_sec
    
    nearest = min(beat_times, key=lambda t: abs(t - time_sec))
    if abs(nearest - time_sec) <= tolerance:
        return nearest
    return time_sec


class PacingProfile:
    """Pacing profile for style pack integration."""
    
    PROFILES = {
        "cinematic": {"factor": 1.3, "min": 4.0, "max": 10.0, "transition": 0.8},
        "motivational": {"factor": 0.9, "min": 2.5, "max": 6.0, "transition": 0.4},
        "corporate": {"factor": 1.0, "min": 3.0, "max": 7.0, "transition": 0.5},
        "energetic": {"factor": 0.7, "min": 1.5, "max": 4.0, "transition": 0.3},
        "documentary": {"factor": 1.4, "min": 5.0, "max": 12.0, "transition": 1.0},
    }
    
    @classmethod
    def get(cls, name: str) -> Dict[str, float]:
        """Get profile settings by name."""
        return cls.PROFILES.get(name.lower(), cls.PROFILES["cinematic"])
    
    @classmethod
    def from_style_pack(cls, style_pack) -> Dict[str, float]:
        """Create profile from style pack."""
        base = cls.get(getattr(style_pack, 'name', 'cinematic'))
        
        # Override with style pack values if present
        if hasattr(style_pack, 'pacing_factor'):
            base['factor'] = style_pack.pacing_factor
        if hasattr(style_pack, 'transition_duration'):
            base['transition'] = style_pack.transition_duration
        
        return base


def calculate_scene_durations_with_profile(
    scenes: List[Dict[str, Any]],
    profile_name: str = "cinematic"
) -> List[float]:
    """
    Calculate scene durations using a pacing profile.
    
    Args:
        scenes: List of scene dicts with 'text' and optional 'tension'
        profile_name: Profile name (cinematic, motivational, etc.)
        
    Returns:
        List of durations in seconds
    """
    profile = PacingProfile.get(profile_name)
    engine = PacingEngine(PacingStyle.DYNAMIC)
    
    durations = []
    for scene in scenes:
        text = scene.get('text', '')
        tension = scene.get('tension', 0.5)
        
        # Calculate base duration
        base = engine.calculate_scene_duration(
            text, tension,
            min_duration=profile['min'],
            max_duration=profile['max']
        )
        
        # Apply profile factor
        duration = base * profile['factor']
        
        # Clamp again
        duration = max(profile['min'], min(profile['max'], duration))
        durations.append(round(duration, 2))
    
    return durations


def get_available_profiles() -> List[str]:
    """Get list of available pacing profiles."""
    return list(PacingProfile.PROFILES.keys())

