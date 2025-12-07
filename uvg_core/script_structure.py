"""
UVG MAX Script Structure Module

Cinematic story arc engine with duration normalization.
Structure: Hook → Buildup → Body → Peak → Resolution → CTA
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class StoryBeat(Enum):
    """Story structure beats."""
    HOOK = "hook"              # 0-5s: Grab attention
    BUILDUP = "buildup"        # 5-15s: Context and setup
    BODY = "body"              # 15-40s: Main content
    PEAK = "peak"              # 40-50s: Emotional climax
    RESOLUTION = "resolution"  # 50-55s: Wrap up
    CTA = "cta"                # 55-60s: Call to action


@dataclass
class BeatConfig:
    """Configuration for a story beat."""
    beat: StoryBeat
    duration_ratio: float  # Percentage of total duration
    tension_range: Tuple[float, float]  # Min, max tension
    pacing: str  # slow, medium, fast
    visual_tone: str  # warm, cold, neutral, vibrant
    emotion_target: str


# Default beat configurations for a 60-second video
BEAT_CONFIGS: Dict[StoryBeat, BeatConfig] = {
    StoryBeat.HOOK: BeatConfig(
        beat=StoryBeat.HOOK,
        duration_ratio=0.08,  # ~5s
        tension_range=(0.4, 0.6),
        pacing="fast",
        visual_tone="vibrant",
        emotion_target="awe",
    ),
    StoryBeat.BUILDUP: BeatConfig(
        beat=StoryBeat.BUILDUP,
        duration_ratio=0.17,  # ~10s
        tension_range=(0.3, 0.5),
        pacing="medium",
        visual_tone="neutral",
        emotion_target="neutral",
    ),
    StoryBeat.BODY: BeatConfig(
        beat=StoryBeat.BODY,
        duration_ratio=0.42,  # ~25s
        tension_range=(0.5, 0.7),
        pacing="medium",
        visual_tone="warm",
        emotion_target="tension",
    ),
    StoryBeat.PEAK: BeatConfig(
        beat=StoryBeat.PEAK,
        duration_ratio=0.17,  # ~10s
        tension_range=(0.8, 1.0),
        pacing="slow",
        visual_tone="vibrant",
        emotion_target="awe",
    ),
    StoryBeat.RESOLUTION: BeatConfig(
        beat=StoryBeat.RESOLUTION,
        duration_ratio=0.08,  # ~5s
        tension_range=(0.4, 0.6),
        pacing="slow",
        visual_tone="warm",
        emotion_target="peace",
    ),
    StoryBeat.CTA: BeatConfig(
        beat=StoryBeat.CTA,
        duration_ratio=0.08,  # ~5s
        tension_range=(0.5, 0.7),
        pacing="medium",
        visual_tone="vibrant",
        emotion_target="energetic",
    ),
}


@dataclass
class StructuredScene:
    """A scene with story structure metadata."""
    index: int
    text: str
    duration: float
    beat: StoryBeat
    tension: float
    pacing: str
    visual_tone: str
    emotion: str
    voice_style: str = "calm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "duration": self.duration,
            "beat": self.beat.value,
            "tension": self.tension,
            "pacing": self.pacing,
            "visual_tone": self.visual_tone,
            "emotion": self.emotion,
            "voice_style": self.voice_style,
        }


@dataclass
class StructuredScript:
    """Script with cinematic structure."""
    title: str
    scenes: List[StructuredScene] = field(default_factory=list)
    total_duration: float = 0.0
    tension_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "scenes": [s.to_dict() for s in self.scenes],
            "total_duration": self.total_duration,
            "tension_curve": self.tension_curve,
        }
    
    def get_peak_scene(self) -> Optional[StructuredScene]:
        """Get the scene with highest tension (peak)."""
        if not self.scenes:
            return None
        return max(self.scenes, key=lambda s: s.tension)


class ScriptStructure:
    """
    Cinematic story structure engine.
    
    Creates proper story arcs:
    Hook → Buildup → Body → Peak → Resolution → CTA
    """
    
    # Duration constraints
    MIN_SCENE_DURATION = 2.0
    MAX_SCENE_DURATION = 12.0
    
    def __init__(self, style: str = "cinematic"):
        """
        Initialize script structure engine.
        
        Args:
            style: Video style preset
        """
        self.style = style
        self._beat_configs = BEAT_CONFIGS.copy()
        
        # Adjust configs based on style
        self._apply_style_adjustments()
    
    def _apply_style_adjustments(self) -> None:
        """Adjust beat configs based on style."""
        if self.style == "tiktok":
            # Faster pacing, shorter hook
            self._beat_configs[StoryBeat.HOOK].pacing = "fast"
            self._beat_configs[StoryBeat.BODY].pacing = "fast"
        elif self.style == "corporate":
            # More measured pacing
            self._beat_configs[StoryBeat.HOOK].pacing = "medium"
            self._beat_configs[StoryBeat.BODY].pacing = "slow"
        elif self.style == "documentary":
            # Slower, more contemplative
            for config in self._beat_configs.values():
                config.pacing = "slow"
    
    def _get_tension_for_beat(self, beat: StoryBeat, position: float) -> float:
        """
        Calculate tension for a position within a beat.
        
        Args:
            beat: Story beat
            position: Position within beat (0.0 - 1.0)
            
        Returns:
            Tension value (0.0 - 1.0)
        """
        config = self._beat_configs[beat]
        min_t, max_t = config.tension_range
        
        # Apply easing based on beat type
        if beat == StoryBeat.HOOK:
            # Start high, slight dip
            tension = max_t - (position * 0.1)
        elif beat == StoryBeat.BUILDUP:
            # Gradual rise
            tension = min_t + (position * (max_t - min_t))
        elif beat == StoryBeat.BODY:
            # Wave pattern
            import math
            wave = math.sin(position * math.pi * 2) * 0.1
            tension = (min_t + max_t) / 2 + wave
        elif beat == StoryBeat.PEAK:
            # Build to maximum
            tension = min_t + (position * (max_t - min_t))
        elif beat == StoryBeat.RESOLUTION:
            # Gradual decrease
            tension = max_t - (position * (max_t - min_t))
        elif beat == StoryBeat.CTA:
            # Moderate, call to action
            tension = (min_t + max_t) / 2
        else:
            tension = 0.5
        
        return round(max(0.0, min(1.0, tension)), 2)
    
    def _assign_beats_to_scenes(self, 
                                 num_scenes: int,
                                 total_duration: float) -> List[Tuple[StoryBeat, float]]:
        """
        Assign story beats and durations to scenes.
        
        Args:
            num_scenes: Number of scenes
            total_duration: Total video duration
            
        Returns:
            List of (beat, duration) tuples
        """
        beats_order = [
            StoryBeat.HOOK,
            StoryBeat.BUILDUP,
            StoryBeat.BODY,
            StoryBeat.PEAK,
            StoryBeat.RESOLUTION,
            StoryBeat.CTA,
        ]
        
        assignments = []
        
        if num_scenes <= 3:
            # Very short video
            assignments = [
                (StoryBeat.HOOK, 0.2),
                (StoryBeat.BODY, 0.5),
                (StoryBeat.CTA, 0.3),
            ][:num_scenes]
        elif num_scenes <= 5:
            # Short video
            assignments = [
                (StoryBeat.HOOK, 0.15),
                (StoryBeat.BUILDUP, 0.2),
                (StoryBeat.BODY, 0.35),
                (StoryBeat.PEAK, 0.2),
                (StoryBeat.CTA, 0.1),
            ][:num_scenes]
        else:
            # Standard or long video
            scenes_per_beat = {
                StoryBeat.HOOK: 1,
                StoryBeat.BUILDUP: max(1, num_scenes // 6),
                StoryBeat.BODY: max(2, num_scenes // 2),
                StoryBeat.PEAK: max(1, num_scenes // 6),
                StoryBeat.RESOLUTION: 1,
                StoryBeat.CTA: 1,
            }
            
            # Adjust to match num_scenes
            total_assigned = sum(scenes_per_beat.values())
            while total_assigned < num_scenes:
                scenes_per_beat[StoryBeat.BODY] += 1
                total_assigned += 1
            while total_assigned > num_scenes:
                if scenes_per_beat[StoryBeat.BODY] > 1:
                    scenes_per_beat[StoryBeat.BODY] -= 1
                    total_assigned -= 1
                elif scenes_per_beat[StoryBeat.BUILDUP] > 1:
                    scenes_per_beat[StoryBeat.BUILDUP] -= 1
                    total_assigned -= 1
                else:
                    break
            
            # Build assignments
            for beat in beats_order:
                count = scenes_per_beat.get(beat, 0)
                ratio = self._beat_configs[beat].duration_ratio / max(1, count)
                for _ in range(count):
                    assignments.append((beat, ratio))
        
        # Normalize ratios
        total_ratio = sum(r for _, r in assignments)
        if total_ratio > 0:
            assignments = [(b, r / total_ratio) for b, r in assignments]
        
        # Convert to actual durations
        result = []
        for beat, ratio in assignments:
            duration = ratio * total_duration
            duration = max(self.MIN_SCENE_DURATION, 
                          min(self.MAX_SCENE_DURATION, duration))
            result.append((beat, duration))
        
        return result
    
    def structure_script(self, 
                         scenes: List[Dict[str, Any]],
                         target_duration: float = 60.0) -> StructuredScript:
        """
        Apply cinematic structure to a script.
        
        Args:
            scenes: List of scene dicts with 'text', 'emotion', etc.
            target_duration: Target total duration
            
        Returns:
            StructuredScript with proper story arc
        """
        num_scenes = len(scenes)
        if num_scenes == 0:
            return StructuredScript(title="Empty", total_duration=0)
        
        # Assign beats and durations
        beat_assignments = self._assign_beats_to_scenes(num_scenes, target_duration)
        
        structured = StructuredScript(
            title=scenes[0].get("title", "Untitled"),
            total_duration=0,
        )
        
        for i, scene_data in enumerate(scenes):
            if i < len(beat_assignments):
                beat, duration = beat_assignments[i]
            else:
                beat = StoryBeat.BODY
                duration = target_duration / num_scenes
            
            config = self._beat_configs[beat]
            
            # Calculate tension based on position within beat
            position = (i / max(1, num_scenes - 1)) if num_scenes > 1 else 0.5
            tension = self._get_tension_for_beat(beat, position)
            
            # Determine voice style from tension
            if tension >= 0.8:
                voice_style = "dramatic"
            elif tension >= 0.6:
                voice_style = "energetic"
            elif tension <= 0.3:
                voice_style = "calm"
            else:
                voice_style = "inspirational"
            
            structured_scene = StructuredScene(
                index=i,
                text=scene_data.get("text", ""),
                duration=duration,
                beat=beat,
                tension=tension,
                pacing=config.pacing,
                visual_tone=config.visual_tone,
                emotion=scene_data.get("emotion", config.emotion_target),
                voice_style=voice_style,
            )
            
            structured.scenes.append(structured_scene)
            structured.total_duration += duration
            structured.tension_curve.append(tension)
        
        logger.info(f"Structured script with {num_scenes} scenes, "
                   f"duration {structured.total_duration:.1f}s")
        
        return structured
    
    def normalize_duration(self, 
                           script: StructuredScript,
                           target_duration: float) -> StructuredScript:
        """
        Normalize script duration to match target.
        
        If script is too long → compress scenes
        If script is too short → stretch scenes
        
        Args:
            script: Script to normalize
            target_duration: Target duration in seconds
            
        Returns:
            Normalized script
        """
        if not script.scenes:
            return script
        
        current_duration = sum(s.duration for s in script.scenes)
        
        if abs(current_duration - target_duration) < 0.5:
            return script  # Close enough
        
        ratio = target_duration / current_duration if current_duration > 0 else 1.0
        
        logger.info(f"Normalizing duration from {current_duration:.1f}s to {target_duration:.1f}s "
                   f"(ratio: {ratio:.2f})")
        
        # Adjust each scene's duration
        new_total = 0.0
        for scene in script.scenes:
            new_duration = scene.duration * ratio
            
            # Clamp to constraints
            new_duration = max(self.MIN_SCENE_DURATION, 
                              min(self.MAX_SCENE_DURATION, new_duration))
            
            scene.duration = round(new_duration, 2)
            new_total += scene.duration
        
        script.total_duration = new_total
        
        # If still off, distribute remainder
        remainder = target_duration - new_total
        if abs(remainder) > 0.1 and len(script.scenes) > 0:
            per_scene = remainder / len(script.scenes)
            for scene in script.scenes:
                scene.duration = max(
                    self.MIN_SCENE_DURATION,
                    min(self.MAX_SCENE_DURATION, scene.duration + per_scene)
                )
            script.total_duration = sum(s.duration for s in script.scenes)
        
        return script
    
    def get_tension_curve(self, script: StructuredScript) -> List[float]:
        """Get the tension curve for the script."""
        return [s.tension for s in script.scenes]
    
    def adjust_pacing(self, 
                      script: StructuredScript,
                      pacing_style: str) -> StructuredScript:
        """
        Adjust script pacing based on style.
        
        Args:
            script: Script to adjust
            pacing_style: slow, medium, fast, dynamic
            
        Returns:
            Adjusted script
        """
        pacing_multipliers = {
            "slow": {"hook": 1.2, "body": 1.3, "peak": 1.4},
            "medium": {"hook": 1.0, "body": 1.0, "peak": 1.0},
            "fast": {"hook": 0.7, "body": 0.8, "peak": 0.9},
            "dynamic": {"hook": 0.8, "body": 1.0, "peak": 1.3},
        }
        
        multipliers = pacing_multipliers.get(pacing_style, pacing_multipliers["medium"])
        
        for scene in script.scenes:
            beat_key = scene.beat.value if scene.beat.value in ["hook", "body", "peak"] else "body"
            multiplier = multipliers.get(beat_key, 1.0)
            scene.duration *= multiplier
            scene.pacing = pacing_style
        
        script.total_duration = sum(s.duration for s in script.scenes)
        
        return script


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def structure_script(scenes: List[Dict[str, Any]], 
                     duration: float = 60.0,
                     style: str = "cinematic") -> StructuredScript:
    """Structure a script with cinematic story arc."""
    engine = ScriptStructure(style=style)
    return engine.structure_script(scenes, duration)


def normalize_duration(script: StructuredScript, 
                       target: float) -> StructuredScript:
    """Normalize script to target duration."""
    engine = ScriptStructure()
    return engine.normalize_duration(script, target)


def get_tension_curve(script: StructuredScript) -> List[float]:
    """Get tension curve for visualization."""
    return script.tension_curve
