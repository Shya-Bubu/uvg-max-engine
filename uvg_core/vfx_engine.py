"""
UVG MAX VFX Engine Module

Emotional VFX presets with LUT, grain, bloom, and more.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VFXPreset:
    """A VFX preset configuration."""
    name: str
    description: str
    filters: List[str] = field(default_factory=list)
    lut: str = ""


# VFX preset definitions
VFX_PRESETS: Dict[str, VFXPreset] = {
    "cinematic": VFXPreset(
        name="cinematic",
        description="Warm LUTs, subtle bloom, film grain",
        filters=[
            "eq=saturation=1.1:contrast=1.05",
            "colorbalance=rs=0.05:gs=0:bs=-0.05",  # Warm shadows
            "unsharp=5:5:0.5",  # Subtle sharpening
        ],
        lut="cinematic_warm",
    ),
    "motivational": VFXPreset(
        name="motivational",
        description="High contrast, bold colors, vignette",
        filters=[
            "eq=saturation=1.2:contrast=1.15:brightness=0.02",
            "vignette=PI/4",
            "unsharp=5:5:0.8",
        ],
        lut="dramatic_dark",
    ),
    "dramatic": VFXPreset(
        name="dramatic",
        description="Dark shadows, high contrast, desaturated",
        filters=[
            "eq=saturation=0.85:contrast=1.2:brightness=-0.05",
            "vignette=PI/3",
            "curves=m=0/0 0.3/0.25 0.7/0.75 1/1",
        ],
        lut="dramatic_dark",
    ),
    "travel": VFXPreset(
        name="travel",
        description="Vibrant saturation, lens flare simulation",
        filters=[
            "eq=saturation=1.3:contrast=1.05:brightness=0.03",
            "colorbalance=rm=0.03:gm=0.02:bm=-0.02",
        ],
        lut="vibrant",
    ),
    "documentary": VFXPreset(
        name="documentary",
        description="Natural colors, minimal processing",
        filters=[
            "eq=saturation=1.0:contrast=1.02",
            "unsharp=3:3:0.3",
        ],
        lut="",
    ),
    "romantic": VFXPreset(
        name="romantic",
        description="Soft glow, warm tones, dreamy",
        filters=[
            "eq=saturation=0.95:contrast=0.95:brightness=0.02",
            "colorbalance=rs=0.08:gs=0.02:bs=-0.03",
            "gblur=sigma=0.5",  # Slight softness
        ],
        lut="soft_warm",
    ),
    "tech": VFXPreset(
        name="tech",
        description="Cool tones, clean, sharp",
        filters=[
            "eq=saturation=0.9:contrast=1.1",
            "colorbalance=rs=-0.05:gs=-0.02:bs=0.1",  # Cool
            "unsharp=7:7:0.6",
        ],
        lut="cool_tech",
    ),
    "tiktok": VFXPreset(
        name="tiktok",
        description="High energy, vibrant, punchy",
        filters=[
            "eq=saturation=1.4:contrast=1.2:brightness=0.02",
            "vibrance=intensity=0.3",
        ],
        lut="vibrant",
    ),
    "high_energy": VFXPreset(
        name="high_energy",
        description="Maximum impact, bold contrast",
        filters=[
            "eq=saturation=1.3:contrast=1.25",
            "vignette=PI/5",
            "unsharp=7:7:0.8",
        ],
        lut="dramatic_dark",
    ),
    "minimal": VFXPreset(
        name="minimal",
        description="Clean, no heavy effects",
        filters=[
            "eq=saturation=1.0:contrast=1.0",
        ],
        lut="",
    ),
    "noir": VFXPreset(
        name="noir",
        description="Black and white with high contrast",
        filters=[
            "hue=s=0",  # Desaturate
            "eq=contrast=1.3:brightness=-0.05",
            "vignette=PI/3.5",
        ],
        lut="",
    ),
    "vintage": VFXPreset(
        name="vintage",
        description="Retro look with faded colors",
        filters=[
            "eq=saturation=0.7:contrast=0.9:brightness=0.03",
            "colorbalance=rs=0.1:gs=0.05:bs=-0.05",
            "curves=m=0/0.1 0.5/0.5 1/0.9",  # Lifted blacks, crushed whites
        ],
        lut="",
    ),
}


class VFXEngine:
    """
    VFX engine for applying visual effects.
    
    Features:
    - Preset-based VFX (cinematic, motivational, travel, etc.)
    - LUT application
    - Film grain
    - Bloom
    - Chromatic aberration
    - Dynamic vignette
    """
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 lut_dir: Optional[Path] = None):
        """
        Initialize VFX engine.
        
        Args:
            output_dir: Output directory
            lut_dir: Directory containing LUT files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/vfx")
        self.lut_dir = Path(lut_dir) if lut_dir else Path("./assets/luts")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_preset(self, name: str) -> VFXPreset:
        """Get VFX preset by name."""
        return VFX_PRESETS.get(name, VFX_PRESETS["cinematic"])
    
    def list_presets(self) -> List[str]:
        """List available presets."""
        return list(VFX_PRESETS.keys())
    
    def build_filter_chain(self,
                            preset: VFXPreset,
                            add_grain: bool = True,
                            grain_strength: float = 0.02) -> str:
        """
        Build FFmpeg filter chain for preset.
        
        Args:
            preset: VFX preset
            add_grain: Add film grain
            grain_strength: Grain intensity
            
        Returns:
            FFmpeg filter string
        """
        filters = preset.filters.copy()
        
        # Add LUT if specified and exists
        if preset.lut and self.lut_dir.exists():
            lut_path = self.lut_dir / f"{preset.lut}.cube"
            if lut_path.exists():
                filters.insert(0, f"lut3d={lut_path}")
        
        # Add film grain
        if add_grain and grain_strength > 0:
            # Using noise filter to simulate grain
            filters.append(f"noise=alls={int(grain_strength*100)}:allf=t")
        
        return ",".join(filters) if filters else "null"
    
    def apply_preset(self,
                     clip_path: str,
                     preset_name: str,
                     output_path: Optional[str] = None,
                     add_grain: bool = True) -> str:
        """
        Apply VFX preset to clip.
        
        Args:
            clip_path: Input clip
            preset_name: Preset name
            output_path: Output path
            add_grain: Add film grain
            
        Returns:
            Output path
        """
        if output_path is None:
            output_path = self.output_dir / f"vfx_{preset_name}_{Path(clip_path).stem}.mp4"
        
        preset = self.get_preset(preset_name)
        filter_chain = self.build_filter_chain(preset, add_grain)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                return str(output_path)
            else:
                logger.warning(f"VFX application failed: {result.stderr[:200]}")
                return clip_path
        except Exception as e:
            logger.warning(f"VFX failed: {e}")
            return clip_path
    
    def add_bloom(self,
                  clip_path: str,
                  intensity: float = 0.3,
                  output_path: Optional[str] = None) -> str:
        """
        Add bloom effect.
        
        Args:
            clip_path: Input clip
            intensity: Bloom intensity
            output_path: Output path
            
        Returns:
            Output path
        """
        if output_path is None:
            output_path = self.output_dir / f"bloom_{Path(clip_path).stem}.mp4"
        
        # Simulate bloom with blur and blend
        filter_chain = (
            f"split[a][b];"
            f"[b]gblur=sigma=30,curves=m=0.3/0:0.7/0.3:1/1[bloom];"
            f"[a][bloom]blend=all_mode='screen':all_opacity={intensity}"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-filter_complex", filter_chain,
            "-c:v", "libx264",
            "-preset", "fast",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            return str(output_path)
        except Exception:
            return clip_path
    
    def add_chromatic_aberration(self,
                                  clip_path: str,
                                  strength: int = 3,
                                  output_path: Optional[str] = None) -> str:
        """
        Add chromatic aberration effect.
        
        Args:
            clip_path: Input clip
            strength: Aberration strength in pixels
            output_path: Output path
            
        Returns:
            Output path
        """
        if output_path is None:
            output_path = self.output_dir / f"chroma_{Path(clip_path).stem}.mp4"
        
        # Simulate CA by offsetting color channels
        filter_chain = (
            f"split=3[r][g][b];"
            f"[r]lutrgb=r=val:g=0:b=0,scroll=h=-{strength}[rr];"
            f"[g]lutrgb=r=0:g=val:b=0[gg];"
            f"[b]lutrgb=r=0:g=0:b=val,scroll=h={strength}[bb];"
            f"[rr][gg]blend=all_mode='addition'[rg];"
            f"[rg][bb]blend=all_mode='addition'"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-filter_complex", filter_chain,
            "-c:v", "libx264",
            "-preset", "fast",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            return str(output_path)
        except Exception:
            return clip_path
    
    def select_preset_by_emotion(self, emotion: str, tension: float = 0.5) -> str:
        """
        Select VFX preset based on emotion.
        
        Args:
            emotion: Scene emotion
            tension: Tension level
            
        Returns:
            Preset name
        """
        emotion_map = {
            "joy": "travel",
            "awe": "cinematic",
            "tension": "dramatic",
            "peace": "romantic",
            "hope": "motivational",
            "energetic": "high_energy",
            "neutral": "minimal",
            "sad": "noir",
        }
        
        preset = emotion_map.get(emotion, "cinematic")
        
        # Adjust for tension
        if tension > 0.8:
            preset = "dramatic"
        elif tension < 0.2:
            preset = "romantic"
        
        return preset


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_vfx(clip_path: str, preset: str = "cinematic") -> str:
    """Apply VFX preset to clip."""
    engine = VFXEngine()
    return engine.apply_preset(clip_path, preset)


def get_preset_for_emotion(emotion: str) -> str:
    """Get appropriate VFX preset for emotion."""
    engine = VFXEngine()
    return engine.select_preset_by_emotion(emotion)


def apply_emotion_vfx(
    clip_path: str,
    emotion: str = "neutral",
    output_path: str = None
) -> str:
    """
    Apply VFX based on scene emotion using scene_emotion module.
    
    Integrates with uvg_core.scene_emotion.SceneEmotionController
    to get emotion-specific VFX parameters (bloom, contrast, saturation).
    
    Args:
        clip_path: Input clip path
        emotion: Scene emotion (calm, exciting, dramatic, etc.)
        output_path: Output path (optional)
        
    Returns:
        Path to processed clip
    """
    engine = VFXEngine()
    
    # Try to get parameters from scene_emotion module
    try:
        from uvg_core.scene_emotion import SceneEmotionController
        controller = SceneEmotionController()
        config = controller.get_config(emotion)
        
        # Use emotion config for VFX parameters
        bloom_intensity = config.vfx_bloom
        contrast = config.vfx_contrast
        
        # Map emotion to preset
        preset = engine.select_preset_by_emotion(emotion)
        
        # Apply preset first
        result = engine.apply_preset(clip_path, preset, output_path)
        
        # Apply bloom if intensity > 0.3
        if bloom_intensity > 0.3:
            result = engine.add_bloom(result, bloom_intensity)
        
        return result
        
    except ImportError:
        # Fallback to simple preset selection
        preset = engine.select_preset_by_emotion(emotion)
        return engine.apply_preset(clip_path, preset, output_path)
