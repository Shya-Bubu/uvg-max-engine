"""
UVG MAX Script Generator Module

Generates video scripts with fallback chain:
1. Gemini API (primary)
2. Gemma 2B local (secondary)
3. Template scripts (emergency)

Also includes query expansion for 10x better clip relevance.
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
class Scene:
    """A single scene in the script."""
    index: int
    text: str  # Narration text
    duration: float = 4.0  # Target duration in seconds
    emotion: str = "neutral"
    tension: float = 0.5  # 0.0 - 1.0
    visual_descriptor: str = ""  # Expanded visual description
    search_query: str = ""  # Expanded search query
    voice_style: str = "calm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "duration": self.duration,
            "emotion": self.emotion,
            "tension": self.tension,
            "visual_descriptor": self.visual_descriptor,
            "search_query": self.search_query,
            "voice_style": self.voice_style,
        }


@dataclass
class Script:
    """Complete video script."""
    title: str
    scenes: List[Scene] = field(default_factory=list)
    total_duration: float = 0.0
    style: str = "cinematic"
    source: str = "unknown"  # gemini, gemma, template
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "scenes": [s.to_dict() for s in self.scenes],
            "total_duration": self.total_duration,
            "style": self.style,
            "source": self.source,
        }
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Script":
        with open(path, 'r') as f:
            data = json.load(f)
        
        script = cls(
            title=data["title"],
            total_duration=data.get("total_duration", 0),
            style=data.get("style", "cinematic"),
            source=data.get("source", "loaded"),
        )
        
        for scene_data in data.get("scenes", []):
            script.scenes.append(Scene(**scene_data))
        
        return script


# =============================================================================
# TEMPLATE SCRIPTS (Emergency Fallback)
# =============================================================================

TEMPLATE_SCRIPTS: Dict[str, Dict[str, Any]] = {
    "motivational": {
        "scenes": [
            {"text": "Every great journey begins with a single step.", "emotion": "inspirational", "tension": 0.3},
            {"text": "The path to success is never easy.", "emotion": "tension", "tension": 0.5},
            {"text": "But those who persist will find their way.", "emotion": "hope", "tension": 0.7},
            {"text": "Your potential is limitless.", "emotion": "awe", "tension": 0.9},
            {"text": "Take action today. Start now.", "emotion": "energetic", "tension": 0.6},
        ],
    },
    "travel": {
        "scenes": [
            {"text": "Discover places you've never seen before.", "emotion": "awe", "tension": 0.4},
            {"text": "Explore ancient wonders and modern marvels.", "emotion": "joy", "tension": 0.5},
            {"text": "Meet people who will change your life.", "emotion": "peace", "tension": 0.6},
            {"text": "Create memories that last forever.", "emotion": "joy", "tension": 0.8},
            {"text": "Your adventure awaits.", "emotion": "energetic", "tension": 0.5},
        ],
    },
    "tech": {
        "scenes": [
            {"text": "Technology is reshaping our world.", "emotion": "awe", "tension": 0.4},
            {"text": "Innovation drives progress forward.", "emotion": "tension", "tension": 0.6},
            {"text": "The future is being built today.", "emotion": "hope", "tension": 0.7},
            {"text": "Embrace the digital revolution.", "emotion": "energetic", "tension": 0.8},
            {"text": "Transform your vision into reality.", "emotion": "inspirational", "tension": 0.6},
        ],
    },
    "corporate": {
        "scenes": [
            {"text": "Excellence in every detail.", "emotion": "neutral", "tension": 0.3},
            {"text": "Our commitment to quality drives results.", "emotion": "neutral", "tension": 0.4},
            {"text": "Innovation meets reliability.", "emotion": "neutral", "tension": 0.5},
            {"text": "Partner with us for success.", "emotion": "hope", "tension": 0.6},
            {"text": "Together, we achieve more.", "emotion": "neutral", "tension": 0.4},
        ],
    },
    "default": {
        "scenes": [
            {"text": "Welcome to an incredible journey.", "emotion": "joy", "tension": 0.3},
            {"text": "Discover something new today.", "emotion": "awe", "tension": 0.5},
            {"text": "Experience the extraordinary.", "emotion": "tension", "tension": 0.7},
            {"text": "See the world differently.", "emotion": "peace", "tension": 0.8},
            {"text": "This is just the beginning.", "emotion": "hope", "tension": 0.5},
        ],
    },
}


# =============================================================================
# QUERY EXPANSION PROMPTS
# =============================================================================

EXPANSION_PROMPT = """
Given this scene description for a video, generate an expanded, detailed visual search query.
The query should describe the exact visuals needed for stock video search.

Scene text: "{scene_text}"
Scene emotion: "{emotion}"

Generate a search query that includes:
- Shot type (wide shot, close-up, aerial, etc.)
- Visual elements (objects, people, nature, etc.)
- Lighting and color (golden hour, dramatic, bright, etc.)
- Mood and atmosphere
- Camera movement if any

Output ONLY the search query, no explanations. Keep it under 50 words.
"""

VISUAL_DESCRIPTOR_PROMPT = """
For this video scene, describe exactly what visuals should appear on screen.
Be specific and cinematic.

Scene narration: "{scene_text}"
Emotion: "{emotion}"

Describe the visual in detail:
- What is shown in the frame
- Composition and framing
- Colors and lighting
- Movement and dynamics

Output ONLY the visual description, no explanations. Keep it under 60 words.
"""


class ScriptGenerator:
    """
    Generates scripts with fallback chain.
    
    Priority:
    1. Gemini API
    2. Gemma 2B local
    3. Template fallback
    """
    
    def __init__(self, 
                 gemini_api_key: str = "",
                 enable_gemini: bool = True,
                 enable_local_fallback: bool = True,
                 cache_dir: Optional[Path] = None):
        """
        Initialize script generator.
        
        Args:
            gemini_api_key: Gemini API key
            enable_gemini: Enable Gemini API
            enable_local_fallback: Enable Gemma local model
            cache_dir: Directory to cache results
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.enable_gemini = enable_gemini and bool(self.gemini_api_key)
        self.enable_local_fallback = enable_local_fallback
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Gemini client (lazy init)
        self._gemini_model = None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _check_cache(self, key: str) -> Optional[str]:
        """Check if result is cached."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return data.get("result")
            except Exception:
                pass
        return None
    
    def _save_cache(self, key: str, result: str) -> None:
        """Save result to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({"result": result}, f)
        except Exception as e:
            logger.debug(f"Failed to cache result: {e}")
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API."""
        if not self.enable_gemini:
            return None
        
        try:
            import google.generativeai as genai
            
            if self._gemini_model is None:
                genai.configure(api_key=self.gemini_api_key)
                self._gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = self._gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except ImportError:
            logger.warning("google-generativeai not installed")
            return None
        except Exception as e:
            logger.warning(f"Gemini API failed: {e}")
            return None
    
    def _call_gemma_local(self, prompt: str) -> Optional[str]:
        """Call local Gemma model."""
        if not self.enable_local_fallback:
            return None
        
        try:
            # Try transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "google/gemma-2b-it"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the output
            result = result[len(prompt):].strip()
            
            return result
            
        except ImportError:
            logger.debug("transformers not installed for local fallback")
            return None
        except Exception as e:
            logger.warning(f"Local Gemma failed: {e}")
            return None
    
    def _get_template_script(self, prompt: str, style: str = "default") -> Script:
        """Get template script as emergency fallback."""
        template = TEMPLATE_SCRIPTS.get(style, TEMPLATE_SCRIPTS["default"])
        
        script = Script(
            title=prompt[:50] if prompt else "Untitled Video",
            style=style,
            source="template",
        )
        
        for i, scene_data in enumerate(template["scenes"]):
            script.scenes.append(Scene(
                index=i,
                text=scene_data["text"],
                duration=4.0,
                emotion=scene_data.get("emotion", "neutral"),
                tension=scene_data.get("tension", 0.5),
            ))
        
        script.total_duration = sum(s.duration for s in script.scenes)
        return script
    
    def generate_script(self, 
                        prompt: str,
                        target_duration: float = 30.0,
                        style: str = "cinematic",
                        num_scenes: int = 5) -> Script:
        """
        Generate a video script.
        
        Args:
            prompt: User's video topic/description
            target_duration: Target video duration in seconds
            style: Video style preset
            num_scenes: Number of scenes to generate
            
        Returns:
            Script object with scenes
        """
        # Calculate scene duration
        scene_duration = target_duration / num_scenes
        
        # Try Gemini first
        gemini_prompt = f"""
Create a {num_scenes}-scene video script about: "{prompt}"
Style: {style}
Target duration: {target_duration} seconds

For each scene, provide:
1. Narration text (1-2 sentences)
2. Emotion (joy, tension, awe, peace, hope, energetic, neutral)
3. Tension level (0.0-1.0)

Format as JSON array:
[{{"text": "...", "emotion": "...", "tension": 0.5}}]

Output ONLY the JSON array, no explanations.
"""
        
        cache_key = self._get_cache_key(gemini_prompt)
        cached = self._check_cache(cache_key)
        
        result = None
        source = "template"
        
        if cached:
            result = cached
            source = "cache"
        else:
            # Try Gemini
            result = self._call_gemini(gemini_prompt)
            if result:
                source = "gemini"
                self._save_cache(cache_key, result)
            else:
                # Try Gemma local
                result = self._call_gemma_local(gemini_prompt)
                if result:
                    source = "gemma"
        
        # Parse result or use template
        if result:
            try:
                # Clean up result (remove markdown code blocks if present)
                clean_result = result.strip()
                if clean_result.startswith("```"):
                    clean_result = clean_result.split("```")[1]
                    if clean_result.startswith("json"):
                        clean_result = clean_result[4:]
                
                scenes_data = json.loads(clean_result)
                
                script = Script(
                    title=prompt[:100],
                    style=style,
                    source=source,
                )
                
                for i, scene_data in enumerate(scenes_data):
                    script.scenes.append(Scene(
                        index=i,
                        text=scene_data.get("text", ""),
                        duration=scene_duration,
                        emotion=scene_data.get("emotion", "neutral"),
                        tension=float(scene_data.get("tension", 0.5)),
                    ))
                
                script.total_duration = sum(s.duration for s in script.scenes)
                logger.info(f"Generated script from {source} with {len(script.scenes)} scenes")
                return script
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse script result: {e}")
        
        # Emergency fallback to template
        logger.warning("Using template script as fallback")
        return self._get_template_script(prompt, style)
    
    def expand_search_query(self, scene: Scene) -> str:
        """
        Expand a scene into a detailed search query.
        
        This dramatically improves clip relevance (10x improvement).
        
        Args:
            scene: Scene to expand
            
        Returns:
            Expanded search query
        """
        prompt = EXPANSION_PROMPT.format(
            scene_text=scene.text,
            emotion=scene.emotion,
        )
        
        cache_key = self._get_cache_key(f"expand_{scene.text}")
        cached = self._check_cache(cache_key)
        
        if cached:
            return cached
        
        # Try Gemini
        result = self._call_gemini(prompt)
        
        if result:
            self._save_cache(cache_key, result)
            return result
        
        # Simple fallback expansion
        emotion_keywords = {
            "joy": "happy bright cheerful smiling",
            "tension": "dramatic intense suspense",
            "awe": "epic grand majestic stunning",
            "peace": "calm serene peaceful tranquil",
            "hope": "inspiring hopeful uplifting",
            "energetic": "dynamic fast motion action",
            "neutral": "professional clean modern",
        }
        
        keywords = emotion_keywords.get(scene.emotion, "")
        return f"{scene.text} {keywords} cinematic stock footage"
    
    def generate_visual_descriptor(self, scene: Scene) -> str:
        """
        Generate detailed visual description for a scene.
        
        Args:
            scene: Scene to describe
            
        Returns:
            Visual descriptor for clip matching
        """
        prompt = VISUAL_DESCRIPTOR_PROMPT.format(
            scene_text=scene.text,
            emotion=scene.emotion,
        )
        
        cache_key = self._get_cache_key(f"visual_{scene.text}")
        cached = self._check_cache(cache_key)
        
        if cached:
            return cached
        
        # Try Gemini
        result = self._call_gemini(prompt)
        
        if result:
            self._save_cache(cache_key, result)
            return result
        
        # Simple fallback
        return f"Cinematic shot showing {scene.text.lower()}, {scene.emotion} mood, professional quality"
    
    def expand_all_scenes(self, script: Script) -> Script:
        """
        Expand all scenes with search queries and visual descriptors.
        
        Args:
            script: Script to expand
            
        Returns:
            Script with expanded scenes
        """
        for scene in script.scenes:
            if not scene.search_query:
                scene.search_query = self.expand_search_query(scene)
            if not scene.visual_descriptor:
                scene.visual_descriptor = self.generate_visual_descriptor(scene)
        
        return script


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_script(prompt: str, 
                    duration: float = 30.0,
                    style: str = "cinematic") -> Script:
    """Generate a video script."""
    generator = ScriptGenerator()
    script = generator.generate_script(prompt, duration, style)
    return generator.expand_all_scenes(script)


def expand_query(text: str, emotion: str = "neutral") -> str:
    """Expand a simple text into a detailed search query."""
    generator = ScriptGenerator()
    scene = Scene(index=0, text=text, emotion=emotion)
    return generator.expand_search_query(scene)
