# uvg_core/script_loader.py
"""
Script Loader for UVG MAX.

Unified interface for loading scripts from:
- Manual JSON files
- AI model output

Includes validation and anti-cliché filtering.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SceneData:
    """Data for a single scene."""
    index: int
    text: str
    duration: float = 4.0
    emotion: str = "neutral"  # Legacy field, use scene_emotion
    visual_query: str = ""
    camera_motion: str = "slow-zoom-in"
    transition: str = "fade"
    
    # NEW: Voice preset (documentary, motivational, cinematic, etc.)
    voice_style: str = "documentary"
    
    # NEW: Scene emotion for VFX/SFX/music (calm, exciting, dramatic, etc.)
    scene_emotion: str = "neutral"
    
    # NEW: Speed ramp effect {"in": 1.0, "out": 1.2}
    speed_ramp: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        # Sync legacy emotion to scene_emotion if not set
        if self.scene_emotion == "neutral" and self.emotion != "neutral":
            self.scene_emotion = self.emotion
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "index": self.index,
            "text": self.text,
            "duration": self.duration,
            "emotion": self.emotion,
            "visual_query": self.visual_query,
            "camera_motion": self.camera_motion,
            "transition": self.transition,
            "voice_style": self.voice_style,
            "scene_emotion": self.scene_emotion,
        }
        if self.speed_ramp:
            result["speed_ramp"] = self.speed_ramp
        return result


@dataclass
class ScriptData:
    """Complete script data."""
    title: str
    scenes: List[SceneData]
    style: str = "cinematic"
    total_duration: float = 0.0
    target_platform: str = "youtube"
    
    # Schema version for backward compatibility
    version: str = "2.0"
    
    # Deterministic mode for reproducibility
    # When True: strict reproducibility (no randomness in clip selection)
    # When False: allow randomness for variety
    deterministic_mode: bool = False
    random_seed: Optional[int] = None  # Used when deterministic_mode=True
    
    # Global voice style (can be overridden per-scene)
    voice_style: str = "documentary"
    
    # Language/locale for Whisper and captions
    language: str = "en-US"
    
    # Music configuration
    music: Optional[Dict[str, Any]] = None  # {path, search_query, sync_mode}
    
    def __post_init__(self):
        if self.total_duration == 0 and self.scenes:
            self.total_duration = sum(s.duration for s in self.scenes)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "title": self.title,
            "scenes": [s.to_dict() for s in self.scenes],
            "style": self.style,
            "total_duration": self.total_duration,
            "target_platform": self.target_platform,
            "voice_style": self.voice_style,
            "language": self.language,
            "deterministic_mode": self.deterministic_mode,
        }
        if self.random_seed is not None:
            result["random_seed"] = self.random_seed
        if self.music:
            result["music"] = self.music
        return result


@dataclass
class ValidationResult:
    """Script validation result."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Anti-cliché phrases to flag
CLICHE_PHRASES = [
    "believe in yourself",
    "chase your dreams",
    "never give up",
    "sky is the limit",
    "be the change",
    "live your best life",
    "everything happens for a reason",
    "follow your heart",
    "reach for the stars",
    "make it happen",
    "you can do it",
    "tomorrow is a new day",
    "life is short",
    "carpe diem",
    "the journey of a thousand miles",
]


class ScriptLoader:
    """
    Load and validate scripts from various sources.
    
    Features:
    - JSON file loading
    - Model output parsing
    - Validation
    - Anti-cliché filtering
    - Refinement hooks
    """
    
    def __init__(self, enable_cliche_filter: bool = True):
        """
        Initialize script loader.
        
        Args:
            enable_cliche_filter: Enable anti-cliché warnings
        """
        self.enable_cliche_filter = enable_cliche_filter
    
    def load_from_json(self, path: str) -> ScriptData:
        """
        Load script from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            ScriptData object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Script file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate against schema v2.x
        try:
            from uvg_core.schema_v2 import validate_schema
            errors = validate_schema(data)
            if errors:
                logger.warning(f"Schema validation warnings: {errors}")
        except ImportError:
            pass
        
        return self._parse_dict(data)
    
    def load_from_dict(self, data: Dict[str, Any]) -> ScriptData:
        """
        Load script from dictionary.
        
        Args:
            data: Script dictionary
            
        Returns:
            ScriptData object
        """
        # Validate against schema v2.x
        try:
            from uvg_core.schema_v2 import validate_schema
            errors = validate_schema(data)
            if errors:
                logger.warning(f"Schema validation warnings: {errors}")
        except ImportError:
            pass
        
        return self._parse_dict(data)
    
    def load_from_model_output(self, text: str) -> ScriptData:
        """
        Parse script from AI model text output.
        
        Args:
            text: Raw model output (may contain JSON or structured text)
            
        Returns:
            ScriptData object
        """
        # Try to extract JSON from text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._parse_dict(data)
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON array
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            try:
                scenes = json.loads(array_match.group())
                return self._parse_dict({"title": "Generated Script", "scenes": scenes})
            except json.JSONDecodeError:
                pass
        
        # Fallback: parse as plain text scenes
        return self._parse_plain_text(text)
    
    def _parse_dict(self, data: Dict[str, Any]) -> ScriptData:
        """Parse script from dictionary."""
        scenes = []
        
        # Get global voice style (can be overridden per-scene)
        global_voice_style = data.get("voice_style", "documentary")
        
        for i, scene_data in enumerate(data.get("scenes", [])):
            scenes.append(SceneData(
                index=scene_data.get("index", i),
                text=scene_data.get("text", scene_data.get("narration", "")),
                duration=scene_data.get("duration", 4.0),
                emotion=scene_data.get("emotion", "neutral"),
                visual_query=scene_data.get("visual_query", scene_data.get("visual", "")),
                camera_motion=scene_data.get("camera_motion", "slow-zoom-in"),
                transition=scene_data.get("transition", "fade"),
                # NEW fields
                voice_style=scene_data.get("voice_style", global_voice_style),
                scene_emotion=scene_data.get("scene_emotion", scene_data.get("emotion", "neutral")),
                speed_ramp=scene_data.get("speed_ramp"),
            ))
        
        return ScriptData(
            title=data.get("title", "Untitled"),
            scenes=scenes,
            style=data.get("style", "cinematic"),
            target_platform=data.get("target_platform", "youtube"),
            voice_style=global_voice_style,
            music=data.get("music"),
        )
    
    def _parse_plain_text(self, text: str) -> ScriptData:
        """Parse plain text into scenes (fallback)."""
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        
        scenes = []
        for i, line in enumerate(lines):
            # Skip headers/metadata
            if line.startswith('#') or ':' in line[:20]:
                continue
            
            scenes.append(SceneData(
                index=i,
                text=line,
                duration=4.0,
                emotion="neutral",
            ))
        
        return ScriptData(
            title="Parsed Script",
            scenes=scenes,
        )
    
    def validate(self, script: ScriptData) -> ValidationResult:
        """
        Validate script structure and content.
        
        Args:
            script: ScriptData to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not script.title:
            warnings.append("Missing title")
        
        if not script.scenes:
            errors.append("No scenes in script")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validate each scene
        for scene in script.scenes:
            if not scene.text:
                errors.append(f"Scene {scene.index} has no text")
            
            if scene.duration <= 0:
                warnings.append(f"Scene {scene.index} has invalid duration")
            
            if scene.duration > 30:
                warnings.append(f"Scene {scene.index} duration unusually long ({scene.duration}s)")
        
        # Check total duration
        if script.total_duration < 5:
            warnings.append(f"Total duration very short ({script.total_duration}s)")
        
        if script.total_duration > 300:
            warnings.append(f"Total duration very long ({script.total_duration}s)")
        
        # Anti-cliché check
        if self.enable_cliche_filter:
            cliche_warnings = self._check_cliches(script)
            warnings.extend(cliche_warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _check_cliches(self, script: ScriptData) -> List[str]:
        """Check for cliché phrases."""
        warnings = []
        
        for scene in script.scenes:
            text_lower = scene.text.lower()
            for cliche in CLICHE_PHRASES:
                if cliche in text_lower:
                    warnings.append(
                        f"Scene {scene.index} contains cliché: '{cliche}'"
                    )
        
        return warnings
    
    def refine(self, script: ScriptData) -> ScriptData:
        """
        Refine script (hook for future AI refinement).
        
        Args:
            script: Original script
            
        Returns:
            Refined script
        """
        # Currently just validates and logs
        result = self.validate(script)
        
        if result.warnings:
            logger.warning(f"Script has {len(result.warnings)} warnings")
            for w in result.warnings[:5]:
                logger.warning(f"  - {w}")
        
        return script


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_script(path_or_dict) -> ScriptData:
    """Load script from path or dict."""
    loader = ScriptLoader()
    
    if isinstance(path_or_dict, dict):
        return loader.load_from_dict(path_or_dict)
    elif isinstance(path_or_dict, str) and Path(path_or_dict).exists():
        return loader.load_from_json(path_or_dict)
    elif isinstance(path_or_dict, str):
        return loader.load_from_model_output(path_or_dict)
    else:
        raise ValueError("Invalid script input")


def validate_script(script: ScriptData) -> ValidationResult:
    """Validate a script."""
    loader = ScriptLoader()
    return loader.validate(script)
