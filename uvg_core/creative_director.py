"""
UVG MAX Creative Director Module

Master AI layer for all creative decisions.
Includes Scene Visualizer for detailed visual descriptors.
"""

import os
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SceneDirection:
    """Creative direction for a single scene (Premium v2)."""
    scene_idx: int
    camera_motion: str = "slow-zoom-in"  # slow-zoom-in, pan-left, dolly-in, static, zoom-out
    transition_type: str = "fade"  # fade, whip_pan, cross_zoom, light_leak, dissolve
    transition_duration: float = 0.5
    vfx_preset: str = "cinematic"  # emotional_warm, dramatic_dark, high_energy, etc.
    caption_style: str = "modern"  # modern, bold, elegant, minimal
    caption_animation: str = "fade_slide"  # fade_slide, pop_bounce, typewriter, slide_left
    voice_style: str = "calm"  # energetic, serious, inspirational, calm, dramatic
    music_intensity: float = 0.5  # 0.0 - 1.0
    visual_descriptor: str = ""  # Detailed visual description
    # Premium v2 fields:
    emotion_tag: str = "neutral"  # Classified emotion from text
    color_grade: str = "cinematic"  # warm, cold, cinematic, documentary, desaturated
    shot_type: str = "medium"  # wide, medium, closeup, aerial
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_idx": self.scene_idx,
            "camera_motion": self.camera_motion,
            "transition_type": self.transition_type,
            "transition_duration": self.transition_duration,
            "vfx_preset": self.vfx_preset,
            "caption_style": self.caption_style,
            "caption_animation": self.caption_animation,
            "voice_style": self.voice_style,
            "music_intensity": self.music_intensity,
            "visual_descriptor": self.visual_descriptor,
            "emotion_tag": self.emotion_tag,
            "color_grade": self.color_grade,
            "shot_type": self.shot_type,
        }


@dataclass
class CreativeBrief:
    """Complete creative direction for a video."""
    title: str
    style_preset: str
    scenes: List[SceneDirection] = field(default_factory=list)
    thumbnail_concept: str = ""
    overall_mood: str = "inspirational"
    color_palette: str = "warm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "style_preset": self.style_preset,
            "scenes": [s.to_dict() for s in self.scenes],
            "thumbnail_concept": self.thumbnail_concept,
            "overall_mood": self.overall_mood,
            "color_palette": self.color_palette,
        }
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# STYLE MAPPINGS
# =============================================================================

MOTION_BY_EMOTION = {
    "joy": ["slow-zoom-in", "pan-left", "dolly-in"],
    "tension": ["slow-zoom-in", "static", "tilt-up"],
    "awe": ["slow-zoom-out", "drone", "dolly-in"],
    "peace": ["slow-zoom-in", "static", "pan-right"],
    "hope": ["tilt-up", "slow-zoom-in", "dolly-in"],
    "energetic": ["pan-left", "pan-right", "dolly-in"],
    "neutral": ["slow-zoom-in", "static"],
    "sad": ["slow-zoom-out", "static", "tilt-down"],
    "happy": ["slow-zoom-in", "pan-right", "dolly-in"],
    "motivational": ["tilt-up", "slow-zoom-in", "drone"],
}

# =============================================================================
# EMOTION CLASSIFICATION (Premium v2)
# =============================================================================

EMOTION_KEYWORDS = {
    "sad": ["loss", "miss", "cry", "grief", "pain", "hurt", "alone", "goodbye", "tears", "sorrow"],
    "happy": ["joy", "celebrate", "smile", "laugh", "fun", "excited", "wonderful", "amazing", "love"],
    "motivational": ["achieve", "rise", "conquer", "power", "success", "dream", "goal", "strong", "overcome"],
    "peaceful": ["calm", "serene", "quiet", "rest", "relax", "gentle", "soft", "harmony", "peace"],
    "tense": ["danger", "urgent", "fear", "struggle", "fight", "challenge", "crisis", "dark", "threat"],
    "awe": ["wonder", "magnificent", "breathtaking", "incredible", "spectacular", "vast", "endless"],
    "hope": ["future", "believe", "possible", "tomorrow", "light", "dawn", "new", "beginning"],
}

# MOOD → SHOT MAPPING (Premium v2)
MOOD_SHOT_MAPPING = {
    "sad": {"color_grade": "cold", "shot_type": "wide", "camera_motion": "slow-zoom-out"},
    "happy": {"color_grade": "warm", "shot_type": "closeup", "camera_motion": "slow-zoom-in"},
    "motivational": {"color_grade": "cinematic", "shot_type": "medium", "camera_motion": "tilt-up"},
    "peaceful": {"color_grade": "soft", "shot_type": "wide", "camera_motion": "static"},
    "tense": {"color_grade": "desaturated", "shot_type": "closeup", "camera_motion": "handheld"},
    "awe": {"color_grade": "cinematic", "shot_type": "wide", "camera_motion": "drone"},
    "hope": {"color_grade": "warm", "shot_type": "medium", "camera_motion": "tilt-up"},
    "neutral": {"color_grade": "documentary", "shot_type": "medium", "camera_motion": "slow-zoom-in"},
}

TRANSITION_BY_MOOD = {
    "calm": ["fade", "film_dissolve"],
    "tense": ["glitch", "flash", "whip_pan"],
    "epic": ["cross_zoom", "light_leak", "flash"],
    "nostalgic": ["film_dissolve", "light_leak", "fade"],
    "energetic": ["whip_pan", "cross_zoom", "glitch"],
    "peaceful": ["fade", "film_dissolve"],
}

VFX_BY_TENSION = {
    (0.0, 0.3): "minimal",
    (0.3, 0.5): "subtle_warm",
    (0.5, 0.7): "emotional_warm",
    (0.7, 0.85): "dramatic_dark",
    (0.85, 1.0): "high_energy",
}

VOICE_BY_TENSION = {
    (0.0, 0.3): "calm",
    (0.3, 0.5): "inspirational",
    (0.5, 0.7): "serious",
    (0.7, 0.85): "dramatic",
    (0.85, 1.0): "energetic",
}


# =============================================================================
# VISUAL DESCRIPTOR PROMPTS
# =============================================================================

VISUAL_DESCRIPTOR_PROMPT = """
For this video scene, describe exactly what visuals should appear on screen.
Be extremely specific and cinematic.

Scene narration: "{scene_text}"
Emotion: {emotion}
Tension level: {tension}
Style: {style}

Describe the visual in rich detail:
- Exact subject matter (what/who is shown)
- Shot type (wide, medium, close-up, aerial)
- Composition (rule of thirds, centered, leading lines)
- Lighting (golden hour, dramatic shadows, soft diffused)
- Colors (warm oranges, cool blues, desaturated)
- Movement (slow motion, time lapse, steady)
- Atmosphere (misty, clear, dusty particles)

Output ONLY the visual description. Keep it under 80 words. Be specific enough that a stock video search will find exactly this.
"""


class CreativeDirector:
    """
    Master AI layer for creative decisions.
    
    Responsibilities:
    - Camera motion per scene
    - Transition selection
    - VFX preset selection
    - Caption style
    - Voice style
    - Music intensity
    - Visual descriptors (Scene Visualizer)
    - Thumbnail concept
    
    In mock mode, returns deterministic fallback descriptors.
    """
    
    # MOCK: Deterministic visual descriptor for testing
    MOCK_VISUAL_DESCRIPTOR = "Wide cinematic shot with golden hour lighting, showing dynamic movement, warm color palette, professional stock footage quality, inspirational atmosphere"
    
    def __init__(self, 
                 gemini_api_key: str = "",
                 enable_gemini: bool = True,
                 style_preset: str = "cinematic",
                 cache_dir: Optional[Path] = None,
                 model_name: str = "",
                 mock_mode: bool = False,
                 style_pack_name: str = None):
        """
        Initialize Creative Director.
        
        Args:
            gemini_api_key: Gemini API key
            enable_gemini: Enable AI-assisted decisions
            style_preset: Default style preset
            cache_dir: Directory to cache results
            model_name: Gemini model name (default from env/config)
            mock_mode: Force mock responses for testing
            style_pack_name: Style pack to use (cinematic, motivational, etc.)
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.enable_gemini = enable_gemini and bool(self.gemini_api_key)
        self.style_preset = style_preset
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name or os.getenv("UVG_GEMINI_CREATIVE_MODEL", "gemini-2.5-flash")
        self.mock_mode = mock_mode or os.getenv("UVG_MOCK_MODE", "false").lower() == "true"
        
        self._gemini_model = None
        self._style_pack = None
        
        # Load style pack if specified
        pack_name = style_pack_name or style_preset
        self._load_style_pack(pack_name)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_style_pack(self, pack_name: str) -> None:
        """Load style pack by name."""
        try:
            from .style_pack import load_style_pack
            self._style_pack = load_style_pack(pack_name)
            logger.info(f"Loaded style pack: {self._style_pack.name}")
        except ImportError:
            logger.debug("Style pack module not available")
            self._style_pack = None
        except Exception as e:
            logger.warning(f"Failed to load style pack '{pack_name}': {e}")
            self._style_pack = None
    
    @property
    def style_pack(self):
        """Get active style pack."""
        return self._style_pack
    
    def get_style_pack_transition(self, scene_idx: int) -> str:
        """Get transition from style pack for scene."""
        if self._style_pack and self._style_pack.transitions:
            idx = scene_idx % len(self._style_pack.transitions)
            return self._style_pack.transitions[idx]
        return "fade"
    
    def get_style_pack_lut(self) -> str:
        """Get LUT path from style pack."""
        if self._style_pack:
            return self._style_pack.lut_path
        return ""
    
    def get_style_pack_color_grade(self) -> str:
        """Get color grade from style pack."""
        if self._style_pack:
            return self._style_pack.color_grade
        return "cinematic"
    
    def get_pacing_factor(self) -> float:
        """Get pacing factor from style pack."""
        if self._style_pack:
            return self._style_pack.pacing_factor
        return 1.0
    
    def _get_cache_key(self, content: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _check_cache(self, key: str) -> Optional[str]:
        """Check cache."""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"cd_{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f).get("result")
            except Exception:
                pass
        return None
    
    def _save_cache(self, key: str, result: str) -> None:
        """Save to cache."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"cd_{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({"result": result}, f)
        except Exception:
            pass
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API with configurable model. NO FALLBACK to 1.5."""
        # MOCK: Return deterministic response in mock mode
        if self.mock_mode:
            logger.info("[MOCK] Using mock Gemini response for creative direction")
            return self.MOCK_VISUAL_DESCRIPTOR
        
        if not self.enable_gemini:
            return None
        
        try:
            import google.generativeai as genai
            
            if self._gemini_model is None:
                genai.configure(api_key=self.gemini_api_key)
                # NO FALLBACK to gemini-1.5-flash - use mock or fail
                self._gemini_model = genai.GenerativeModel(self.model_name)
                logger.info(f"Using Gemini model for creative: {self.model_name}")
            
            response = self._gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except ImportError:
            logger.error("google-generativeai not installed. Use UVG_MOCK_MODE=true for testing.")
            return None
        except Exception as e:
            logger.error(f"Gemini API failed for model {self.model_name}: {e}. Use UVG_MOCK_MODE=true.")
            return None
    
    def classify_emotion(self, text: str) -> str:
        """Classify emotion from scene text using keyword matching."""
        import random
        text_lower = text.lower()
        
        # Use debug seed for deterministic results
        seed = int(os.getenv("UVG_DEBUG_SEED", "42"))
        random.seed(seed + hash(text) % 1000)
        
        scores = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "neutral"
    
    def get_mood_shot_settings(self, emotion: str) -> Dict[str, str]:
        """Get camera motion, color grade, shot type for emotion."""
        return MOOD_SHOT_MAPPING.get(emotion, MOOD_SHOT_MAPPING["neutral"])
    
    def _select_camera_motion(self, emotion: str, tension: float) -> str:
        """Select camera motion based on emotion and tension."""
        import random
        options = MOTION_BY_EMOTION.get(emotion, ["slow-zoom-in"])
        
        # Higher tension = more dynamic motion
        if tension > 0.7:
            dynamic_options = ["dolly-in", "pan-left", "tilt-up"]
            options = [o for o in options if o in dynamic_options] or options
        
        return random.choice(options)
    
    def _select_transition(self, mood: str, tension: float) -> Tuple[str, float]:
        """Select transition type and duration."""
        import random
        
        # Map tension to mood if not provided
        if mood == "neutral":
            if tension > 0.7:
                mood = "epic"
            elif tension < 0.3:
                mood = "peaceful"
            else:
                mood = "calm"
        
        options = TRANSITION_BY_MOOD.get(mood, ["fade"])
        transition = random.choice(options)
        
        # Duration based on transition type
        durations = {
            "fade": 0.6,
            "film_dissolve": 0.8,
            "whip_pan": 0.3,
            "cross_zoom": 0.4,
            "light_leak": 0.7,
            "glitch": 0.25,
            "flash": 0.2,
        }
        
        duration = durations.get(transition, 0.5)
        
        return transition, duration
    
    def _select_vfx_preset(self, tension: float) -> str:
        """Select VFX preset based on tension."""
        for (low, high), preset in VFX_BY_TENSION.items():
            if low <= tension < high:
                return preset
        return "emotional_warm"
    
    def _select_voice_style(self, tension: float) -> str:
        """Select voice style based on tension."""
        for (low, high), style in VOICE_BY_TENSION.items():
            if low <= tension < high:
                return style
        return "calm"
    
    def generate_visual_descriptor(self, 
                                    scene_text: str,
                                    emotion: str = "neutral",
                                    tension: float = 0.5,
                                    style: str = "cinematic") -> str:
        """
        Generate detailed visual description for a scene.
        
        This is the Scene Visualizer - critical for clip relevance.
        
        Args:
            scene_text: Scene narration text
            emotion: Scene emotion
            tension: Tension level 0.0-1.0
            style: Video style
            
        Returns:
            Detailed visual descriptor for stock search
        """
        cache_key = self._get_cache_key(f"visual_{scene_text}_{emotion}_{tension}")
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        prompt = VISUAL_DESCRIPTOR_PROMPT.format(
            scene_text=scene_text,
            emotion=emotion,
            tension=tension,
            style=style,
        )
        
        result = self._call_gemini(prompt)
        
        if result:
            self._save_cache(cache_key, result)
            return result
        
        # Fallback: rule-based descriptor
        shot_types = {
            "awe": "wide establishing shot",
            "tension": "close-up detailed shot",
            "joy": "medium shot with movement",
            "peace": "slow wide panoramic shot",
            "energetic": "dynamic tracking shot",
        }
        
        lighting = {
            "warm": "golden hour warm lighting",
            "cold": "cool blue tones",
            "dramatic": "high contrast dramatic shadows",
            "soft": "soft diffused natural light",
        }
        
        shot = shot_types.get(emotion, "medium cinematic shot")
        light = lighting.get("warm" if tension < 0.6 else "dramatic", "natural lighting")
        
        fallback = f"{shot} showing {scene_text.lower()}, {light}, {emotion} atmosphere, professional stock footage quality"
        
        return fallback
    
    def get_scene_direction(self, 
                            scene_idx: int,
                            scene_text: str,
                            emotion: str = "neutral",
                            tension: float = 0.5,
                            pacing: str = "medium") -> SceneDirection:
        """
        Get complete creative direction for a scene.
        
        Args:
            scene_idx: Scene index
            scene_text: Scene narration
            emotion: Scene emotion
            tension: Tension level
            pacing: Pacing style
            
        Returns:
            SceneDirection with all creative decisions
        """
        # Seed for determinism
        import random
        seed = hash(f"{scene_idx}_{scene_text[:20]}")
        random.seed(seed)
        
        camera_motion = self._select_camera_motion(emotion, tension)
        transition, trans_duration = self._select_transition(emotion, tension)
        vfx = self._select_vfx_preset(tension)
        voice = self._select_voice_style(tension)
        
        # Caption style based on preset
        caption_styles = {
            "cinematic": ("elegant", "fade_slide"),
            "tiktok": ("bold", "pop_bounce"),
            "corporate": ("modern", "fade_slide"),
            "motivational": ("bold", "scale_bounce"),
        }
        caption_style, caption_anim = caption_styles.get(
            self.style_preset, ("modern", "fade_slide")
        )
        
        # Music intensity follows tension
        music_intensity = 0.3 + (tension * 0.5)
        
        # Generate visual descriptor
        visual = self.generate_visual_descriptor(
            scene_text, emotion, tension, self.style_preset
        )
        
        return SceneDirection(
            scene_idx=scene_idx,
            camera_motion=camera_motion,
            transition_type=transition,
            transition_duration=trans_duration,
            vfx_preset=vfx,
            caption_style=caption_style,
            caption_animation=caption_anim,
            voice_style=voice,
            music_intensity=music_intensity,
            visual_descriptor=visual,
        )
    
    def generate_creative_brief(self, 
                                 title: str,
                                 scenes: List[Dict[str, Any]]) -> CreativeBrief:
        """
        Generate complete creative brief for a video.
        
        Args:
            title: Video title
            scenes: List of scene dicts with text, emotion, tension
            
        Returns:
            CreativeBrief with direction for all scenes
        """
        brief = CreativeBrief(
            title=title,
            style_preset=self.style_preset,
        )
        
        for scene_data in scenes:
            direction = self.get_scene_direction(
                scene_idx=scene_data.get("index", 0),
                scene_text=scene_data.get("text", ""),
                emotion=scene_data.get("emotion", "neutral"),
                tension=scene_data.get("tension", 0.5),
                pacing=scene_data.get("pacing", "medium"),
            )
            brief.scenes.append(direction)
        
        # Generate thumbnail concept
        brief.thumbnail_concept = self._generate_thumbnail_concept(title, scenes)
        
        # Determine overall mood
        avg_tension = sum(s.get("tension", 0.5) for s in scenes) / max(1, len(scenes))
        if avg_tension > 0.7:
            brief.overall_mood = "epic"
        elif avg_tension > 0.5:
            brief.overall_mood = "dramatic"
        elif avg_tension < 0.3:
            brief.overall_mood = "peaceful"
        else:
            brief.overall_mood = "inspirational"
        
        logger.info(f"Generated creative brief for {len(scenes)} scenes")
        
        return brief
    
    def _generate_thumbnail_concept(self, 
                                     title: str,
                                     scenes: List[Dict[str, Any]]) -> str:
        """Generate thumbnail concept."""
        prompt = f"""
Generate a thumbnail concept for a video titled: "{title}"

The thumbnail should:
- Have a clear focal point
- Use bold, contrasting colors
- Include space for text overlay
- Be attention-grabbing

Describe the ideal thumbnail in 2-3 sentences.
"""
        
        result = self._call_gemini(prompt)
        
        if result:
            return result
        
        # Fallback
        return f"Bold centered composition featuring key visual from the video, " \
               f"warm color grading, large title text with glow effect, " \
               f"subject isolated against blurred background"


# Need to import Tuple
from typing import Tuple


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_scene_direction(scene_text: str,
                        emotion: str = "neutral",
                        tension: float = 0.5) -> SceneDirection:
    """Get creative direction for a single scene."""
    director = CreativeDirector()
    return director.get_scene_direction(0, scene_text, emotion, tension)


def generate_visual_descriptor(scene_text: str,
                                emotion: str = "neutral") -> str:
    """Generate visual descriptor for stock search."""
    director = CreativeDirector()
    return director.generate_visual_descriptor(scene_text, emotion)
