"""
UVG MAX Configuration Module

Central configuration for the Universal Video Generator engine.
Standard library only - no external dependencies.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import warnings
import json


# =============================================================================
# STYLE PRESETS (8 total)
# =============================================================================

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "cinematic": {
        "name": "Cinematic",
        "description": "Warm LUTs, bloom, film grain, elegant captions",
        "lut": "cinematic_warm",
        "vfx": "bloom_grain",
        "caption_style": "elegant",
        "thumbnail_style": "film_poster",
        "transition_style": "film_dissolve",
        "motion_intensity": 0.7,
        "pacing_curve": "slow",
        "music_mood": "orchestral",
    },
    "motivational": {
        "name": "Motivational",
        "description": "High contrast, bold captions, epic VFX",
        "lut": "dramatic_dark",
        "vfx": "high_contrast",
        "caption_style": "bold",
        "thumbnail_style": "cta_text",
        "transition_style": "cross_zoom",
        "motion_intensity": 1.0,
        "pacing_curve": "dynamic",
        "music_mood": "epic",
    },
    "tiktok": {
        "name": "TikTok / Reels",
        "description": "Vibrant, fast cuts, pop captions, emoji overlays",
        "lut": "vibrant",
        "vfx": "fast_energy",
        "caption_style": "pop_bounce",
        "thumbnail_style": "emoji_3d",
        "transition_style": "whip_pan",
        "motion_intensity": 1.3,
        "pacing_curve": "fast",
        "music_mood": "energetic",
    },
    "corporate": {
        "name": "Corporate / Business",
        "description": "Neutral, minimal VFX, clean captions",
        "lut": "neutral",
        "vfx": "minimal",
        "caption_style": "clean",
        "thumbnail_style": "professional",
        "transition_style": "fade",
        "motion_intensity": 0.4,
        "pacing_curve": "measured",
        "music_mood": "ambient",
    },
    "travel": {
        "name": "Travel / Vlog",
        "description": "Saturated colors, lens flare, wanderlust feel",
        "lut": "saturated",
        "vfx": "lens_flare",
        "caption_style": "minimal",
        "thumbnail_style": "wanderlust",
        "transition_style": "light_leak",
        "motion_intensity": 0.9,
        "pacing_curve": "medium",
        "music_mood": "upbeat",
    },
    "documentary": {
        "name": "Documentary",
        "description": "Natural colors, no VFX, subtitle-style captions",
        "lut": "natural",
        "vfx": "none",
        "caption_style": "subtitle",
        "thumbnail_style": "informative",
        "transition_style": "fade",
        "motion_intensity": 0.3,
        "pacing_curve": "slow",
        "music_mood": "ambient",
    },
    "romantic": {
        "name": "Romantic",
        "description": "Soft glow, warm tones, script font captions",
        "lut": "soft_warm",
        "vfx": "soft_glow",
        "caption_style": "script",
        "thumbnail_style": "hearts",
        "transition_style": "film_dissolve",
        "motion_intensity": 0.5,
        "pacing_curve": "slow",
        "music_mood": "romantic",
    },
    "tech": {
        "name": "Tech / Modern",
        "description": "Cool tones, clean look, modern captions",
        "lut": "cool_tech",
        "vfx": "clean",
        "caption_style": "modern",
        "thumbnail_style": "gradient",
        "transition_style": "glitch",
        "motion_intensity": 0.8,
        "pacing_curve": "medium",
        "music_mood": "electronic",
    },
}


# =============================================================================
# AZURE VOICE STYLES
# =============================================================================

AZURE_VOICE_STYLES: Dict[str, Dict[str, Any]] = {
    "energetic": {
        "style": "cheerful",
        "pitch": "+5%",
        "rate": "+10%",
        "description": "Upbeat and lively delivery",
    },
    "serious": {
        "style": "newscast-formal",
        "pitch": "-2%",
        "rate": "-5%",
        "description": "Professional and authoritative",
    },
    "inspirational": {
        "style": "empathetic",
        "pitch": "+3%",
        "rate": "-3%",
        "description": "Warm and motivating",
    },
    "calm": {
        "style": "gentle",
        "pitch": "-3%",
        "rate": "-8%",
        "description": "Soothing and relaxed",
    },
    "dramatic": {
        "style": "narration-professional",
        "pitch": "0%",
        "rate": "-5%",
        "description": "Theatrical and impactful",
    },
}


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

@dataclass
class UVGConfig:
    """
    Central configuration for UVG MAX engine.
    
    All paths auto-create on initialization.
    API keys loaded from environment variables.
    """
    
    # =========================================================================
    # PATHS (auto-created on init)
    # =========================================================================
    base_dir: Path = field(default_factory=lambda: Path("./uvg_output"))
    cache_dir: Path = field(default_factory=lambda: Path("./uvg_output/cache"))
    clips_dir: Path = field(default_factory=lambda: Path("./uvg_output/clips"))
    frames_dir: Path = field(default_factory=lambda: Path("./uvg_output/frames"))
    trimmed_dir: Path = field(default_factory=lambda: Path("./uvg_output/trimmed"))
    prepared_dir: Path = field(default_factory=lambda: Path("./uvg_output/prepared"))
    assets_dir: Path = field(default_factory=lambda: Path("./assets"))
    logs_dir: Path = field(default_factory=lambda: Path("./uvg_output/logs"))
    output_dir: Path = field(default_factory=lambda: Path("./uvg_output/final"))
    
    # =========================================================================
    # API KEYS (loaded from environment - NEVER hardcoded)
    # =========================================================================
    PEXELS_KEY: str = field(default_factory=lambda: os.getenv("PEXELS_KEY", ""))
    PIXABAY_KEY: str = field(default_factory=lambda: os.getenv("PIXABAY_KEY", ""))
    UNSPLASH_KEY: str = field(default_factory=lambda: os.getenv("UNSPLASH_KEY", ""))
    COVERR_KEY: str = field(default_factory=lambda: os.getenv("COVERR_KEY", ""))
    ARCHIVE_S3_KEY: str = field(default_factory=lambda: os.getenv("ARCHIVE_S3_KEY", ""))
    ARCHIVE_S3_SECRET: str = field(default_factory=lambda: os.getenv("ARCHIVE_S3_SECRET", ""))
    FREESOUND_KEY: str = field(default_factory=lambda: os.getenv("FREESOUND_KEY", ""))
    AZURE_TTS_KEY: str = field(default_factory=lambda: os.getenv("AZURE_TTS_KEY", ""))
    AZURE_TTS_REGION: str = field(default_factory=lambda: os.getenv("AZURE_TTS_REGION", ""))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    # =========================================================================
    # VIDEO DEFAULTS
    # =========================================================================
    target_width: int = 1080
    target_height: int = 1920
    fps: int = 30
    max_clip_size_mb: int = 120
    max_candidates: int = 60
    n_prune_heuristic: int = 20
    use_cuda: bool = True
    local_test_mode: bool = False
    
    # =========================================================================
    # AI MODEL SELECTION
    # =========================================================================
    GEMINI_SCRIPT_MODEL: str = field(default_factory=lambda: os.getenv("UVG_GEMINI_SCRIPT_MODEL", "gemini-2.5-flash"))
    GEMINI_CREATIVE_MODEL: str = field(default_factory=lambda: os.getenv("UVG_GEMINI_CREATIVE_MODEL", "gemini-2.5-flash-live"))
    GEMINI_TTS_MODEL: str = field(default_factory=lambda: os.getenv("UVG_GEMINI_TTS_MODEL", "gemini-2.5-flash-tts"))
    TTS_PROVIDER: str = field(default_factory=lambda: os.getenv("UVG_TTS_PROVIDER", "mock"))  # mock, gemini, azure
    
    # =========================================================================
    # DEBUG & MOCK SETTINGS
    # =========================================================================
    UVG_DEBUG_SEED: int = field(default_factory=lambda: int(os.getenv("UVG_DEBUG_SEED", "42")) if os.getenv("UVG_DEBUG_SEED") else 42)
    UVG_MOCK_MODE: bool = field(default_factory=lambda: os.getenv("UVG_MOCK_MODE", "true").lower() == "true")
    MAX_DOWNLOAD_WORKERS: int = field(default_factory=lambda: int(os.getenv("MAX_DOWNLOAD_WORKERS", "6")))
    
    # =========================================================================
    # SCORING WEIGHTS (Option B: relevance-first)
    # =========================================================================
    w_relevance: float = 0.50
    w_heuristics: float = 0.20
    w_emotion: float = 0.15
    w_motion: float = 0.10
    w_color: float = 0.05
    
    # =========================================================================
    # TRIM SETTINGS
    # =========================================================================
    min_segment: float = 1.8
    max_segment: float = 6.0
    window_step: float = 0.5
    
    # =========================================================================
    # CREATIVE DIRECTOR DEFAULTS
    # =========================================================================
    default_pacing_curve: str = "dynamic"
    default_caption_style: str = "modern_minimal"
    default_transition_style: str = "cinematic"
    default_vfx_style: str = "emotional_warm"
    default_music_mood: str = "inspirational"
    
    # =========================================================================
    # CLEANUP SETTINGS
    # =========================================================================
    cleanup_after_stage: bool = True
    disk_threshold_gb: int = 70
    disk_cleanup_target_gb: int = 50
    
    # =========================================================================
    # SCENE RELEVANCE SETTINGS
    # =========================================================================
    relevance_threshold: float = 0.45
    enable_prompt_expansion: bool = True
    max_prompt_retries: int = 3
    
    # =========================================================================
    # CURRENT STYLE PRESET
    # =========================================================================
    style_preset: str = "cinematic"
    
    def __post_init__(self) -> None:
        """Auto-create all directories on initialization."""
        directories = [
            self.base_dir,
            self.cache_dir,
            self.clips_dir,
            self.frames_dir,
            self.trimmed_dir,
            self.prepared_dir,
            self.assets_dir,
            self.logs_dir,
            self.output_dir,
        ]
        
        for directory in directories:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of warning messages (empty if all valid)
        """
        validation_warnings: List[str] = []
        
        # Check API keys - warn if missing (not error)
        api_keys = {
            "PEXELS_KEY": self.PEXELS_KEY,
            "PIXABAY_KEY": self.PIXABAY_KEY,
            "UNSPLASH_KEY": self.UNSPLASH_KEY,
            "AZURE_TTS_KEY": self.AZURE_TTS_KEY,
            "AZURE_TTS_REGION": self.AZURE_TTS_REGION,
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
        }
        
        missing_keys = [name for name, value in api_keys.items() if not value]
        if missing_keys:
            validation_warnings.append(
                f"Missing API keys: {', '.join(missing_keys)}. "
                "Some features may not work."
            )
        
        # Check output directories exist
        directories = [
            ("base_dir", self.base_dir),
            ("output_dir", self.output_dir),
            ("cache_dir", self.cache_dir),
        ]
        
        for name, path in directories:
            if not Path(path).exists():
                validation_warnings.append(
                    f"Directory {name} does not exist: {path}"
                )
        
        # Check max_clip_size_mb > 10
        if self.max_clip_size_mb <= 10:
            validation_warnings.append(
                f"max_clip_size_mb should be > 10, got {self.max_clip_size_mb}"
            )
        
        # Check fps is between 24-60
        if not (24 <= self.fps <= 60):
            validation_warnings.append(
                f"fps should be between 24-60, got {self.fps}"
            )
        
        # Check scoring weights sum to ~1.0
        weight_sum = (
            self.w_relevance + 
            self.w_heuristics + 
            self.w_emotion + 
            self.w_motion + 
            self.w_color
        )
        if abs(weight_sum - 1.0) > 0.01:
            validation_warnings.append(
                f"Scoring weights should sum to 1.0, got {weight_sum:.2f}"
            )
        
        # Check trim settings are sensible
        if self.min_segment >= self.max_segment:
            validation_warnings.append(
                f"min_segment ({self.min_segment}) should be < max_segment ({self.max_segment})"
            )
        
        if self.window_step <= 0:
            validation_warnings.append(
                f"window_step should be > 0, got {self.window_step}"
            )
        
        # Check cleanup settings
        if self.disk_cleanup_target_gb >= self.disk_threshold_gb:
            validation_warnings.append(
                f"disk_cleanup_target_gb ({self.disk_cleanup_target_gb}) "
                f"should be < disk_threshold_gb ({self.disk_threshold_gb})"
            )
        
        # Check style preset is valid
        if self.style_preset not in STYLE_PRESETS:
            validation_warnings.append(
                f"Unknown style preset: {self.style_preset}. "
                f"Available: {list(STYLE_PRESETS.keys())}"
            )
        
        # Check relevance threshold
        if not (0.0 <= self.relevance_threshold <= 1.0):
            validation_warnings.append(
                f"relevance_threshold should be 0.0-1.0, got {self.relevance_threshold}"
            )
        
        # Issue warnings
        for warning_msg in validation_warnings:
            warnings.warn(warning_msg, UserWarning)
        
        return validation_warnings
    
    def summary(self) -> Dict[str, Any]:
        """
        Return a readable dict for debugging.
        
        Returns:
            Dictionary with all configuration values (API keys masked)
        """
        def mask_key(key: str) -> str:
            if not key:
                return "(not set)"
            if len(key) <= 8:
                return "***"
            return f"{key[:4]}...{key[-4:]}"
        
        return {
            "paths": {
                "base_dir": str(self.base_dir),
                "cache_dir": str(self.cache_dir),
                "clips_dir": str(self.clips_dir),
                "frames_dir": str(self.frames_dir),
                "trimmed_dir": str(self.trimmed_dir),
                "prepared_dir": str(self.prepared_dir),
                "assets_dir": str(self.assets_dir),
                "logs_dir": str(self.logs_dir),
                "output_dir": str(self.output_dir),
            },
            "api_keys": {
                "PEXELS_KEY": mask_key(self.PEXELS_KEY),
                "PIXABAY_KEY": mask_key(self.PIXABAY_KEY),
                "UNSPLASH_KEY": mask_key(self.UNSPLASH_KEY),
                "COVERR_KEY": mask_key(self.COVERR_KEY),
                "ARCHIVE_S3_KEY": mask_key(self.ARCHIVE_S3_KEY),
                "FREESOUND_KEY": mask_key(self.FREESOUND_KEY),
                "AZURE_TTS_KEY": mask_key(self.AZURE_TTS_KEY),
                "AZURE_TTS_REGION": self.AZURE_TTS_REGION or "(not set)",
                "GEMINI_API_KEY": mask_key(self.GEMINI_API_KEY),
            },
            "video": {
                "target_width": self.target_width,
                "target_height": self.target_height,
                "fps": self.fps,
                "max_clip_size_mb": self.max_clip_size_mb,
                "max_candidates": self.max_candidates,
                "n_prune_heuristic": self.n_prune_heuristic,
                "use_cuda": self.use_cuda,
                "local_test_mode": self.local_test_mode,
            },
            "scoring_weights": {
                "w_relevance": self.w_relevance,
                "w_heuristics": self.w_heuristics,
                "w_emotion": self.w_emotion,
                "w_motion": self.w_motion,
                "w_color": self.w_color,
            },
            "trim": {
                "min_segment": self.min_segment,
                "max_segment": self.max_segment,
                "window_step": self.window_step,
            },
            "creative_defaults": {
                "default_pacing_curve": self.default_pacing_curve,
                "default_caption_style": self.default_caption_style,
                "default_transition_style": self.default_transition_style,
                "default_vfx_style": self.default_vfx_style,
                "default_music_mood": self.default_music_mood,
            },
            "cleanup": {
                "cleanup_after_stage": self.cleanup_after_stage,
                "disk_threshold_gb": self.disk_threshold_gb,
                "disk_cleanup_target_gb": self.disk_cleanup_target_gb,
            },
            "relevance": {
                "relevance_threshold": self.relevance_threshold,
                "enable_prompt_expansion": self.enable_prompt_expansion,
                "max_prompt_retries": self.max_prompt_retries,
            },
            "style": {
                "current_preset": self.style_preset,
                "preset_config": self.get_style_config(),
            },
        }
    
    def get_style_config(self) -> Dict[str, Any]:
        """Get the current style preset configuration."""
        return STYLE_PRESETS.get(self.style_preset, STYLE_PRESETS["cinematic"])
    
    def get_voice_style(self, style_name: str) -> Dict[str, Any]:
        """Get Azure voice style configuration."""
        return AZURE_VOICE_STYLES.get(style_name, AZURE_VOICE_STYLES["calm"])
    
    def apply_preset(self, preset_name: str) -> None:
        """Apply a style preset to update creative defaults."""
        if preset_name not in STYLE_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available: {list(STYLE_PRESETS.keys())}")
        
        preset = STYLE_PRESETS[preset_name]
        self.style_preset = preset_name
        self.default_pacing_curve = preset.get("pacing_curve", self.default_pacing_curve)
        self.default_caption_style = preset.get("caption_style", self.default_caption_style)
        self.default_transition_style = preset.get("transition_style", self.default_transition_style)
        self.default_vfx_style = preset.get("vfx", self.default_vfx_style)
        self.default_music_mood = preset.get("music_mood", self.default_music_mood)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "UVGConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert paths
        path_fields = [
            'base_dir', 'cache_dir', 'clips_dir', 'frames_dir',
            'trimmed_dir', 'prepared_dir', 'assets_dir', 'logs_dir', 'output_dir'
        ]
        for key in path_fields:
            if key in data:
                data[key] = Path(data[key])
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "UVGConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Override from environment
        if os.getenv("UVG_PRESET"):
            config.apply_preset(os.getenv("UVG_PRESET"))
        if os.getenv("UVG_OUTPUT_WIDTH"):
            config.target_width = int(os.getenv("UVG_OUTPUT_WIDTH"))
        if os.getenv("UVG_OUTPUT_HEIGHT"):
            config.target_height = int(os.getenv("UVG_OUTPUT_HEIGHT"))
        if os.getenv("UVG_FPS"):
            config.fps = int(os.getenv("UVG_FPS"))
        if os.getenv("UVG_USE_CUDA"):
            config.use_cuda = os.getenv("UVG_USE_CUDA").lower() == "true"
        if os.getenv("UVG_LOCAL_TEST"):
            config.local_test_mode = os.getenv("UVG_LOCAL_TEST").lower() == "true"
        
        return config
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UVGConfig("
            f"preset={self.style_preset}, "
            f"resolution={self.target_width}x{self.target_height}, "
            f"fps={self.fps}, "
            f"cuda={self.use_cuda})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_config: Optional[UVGConfig] = None


def get_config() -> UVGConfig:
    """Get global config instance (creates if not exists)."""
    global _global_config
    if _global_config is None:
        _global_config = UVGConfig.from_env()
        _global_config.validate()
    return _global_config


def set_config(config: UVGConfig) -> None:
    """Set global config instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset global config to None."""
    global _global_config
    _global_config = None


def list_presets() -> List[str]:
    """List available style presets."""
    return list(STYLE_PRESETS.keys())


def get_preset_info(preset_name: str) -> Dict[str, Any]:
    """Get information about a style preset."""
    if preset_name not in STYLE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    return STYLE_PRESETS[preset_name]
