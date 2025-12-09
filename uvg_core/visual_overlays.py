# uvg_core/visual_overlays.py
"""
Visual Overlay Engine for UVG MAX.

Cinematic visual effects:
- Film grain
- Lens flares
- Light leaks
- Letterbox
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OverlayResult:
    """Result of overlay application."""
    success: bool
    output_path: str
    overlays_applied: List[str]
    error: str = ""


# Overlay presets
OVERLAY_PRESETS = {
    "cinematic": {
        "grain": {"intensity": 0.08, "type": "35mm"},
        "letterbox": {"ratio": 2.35},
        "flare": {"type": "warm", "probability": 0.3},
    },
    "documentary": {
        "grain": {"intensity": 0.05, "type": "16mm"},
        "letterbox": None,
        "flare": None,
    },
    "vintage": {
        "grain": {"intensity": 0.15, "type": "super8"},
        "letterbox": {"ratio": 1.33},
        "flare": {"type": "warm", "probability": 0.5},
    },
    "clean": {
        "grain": None,
        "letterbox": None,
        "flare": None,
    },
}


class VisualOverlayEngine:
    """
    Apply cinematic visual overlays.
    
    Features:
    - Film grain (multiple types)
    - Lens flares
    - Light leaks
    - Letterbox bars
    - Vignette
    """
    
    # Overlay file paths
    OVERLAYS_DIR = Path("overlays")
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize overlay engine.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_grain(
        self,
        video_path: str,
        intensity: float = 0.08,
        grain_type: str = "35mm",
        output_path: str = None
    ) -> OverlayResult:
        """
        Apply film grain overlay.
        
        Args:
            video_path: Input video
            intensity: Grain intensity (0-1)
            grain_type: "35mm", "16mm", "super8"
            output_path: Output path
            
        Returns:
            OverlayResult
        """
        if not Path(video_path).exists():
            return OverlayResult(
                success=False, output_path="", overlays_applied=[], 
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"grain_{Path(video_path).name}")
        
        # FFmpeg noise filter for grain effect
        # noise=c0s=8:c0f=t creates temporal noise
        noise_strength = int(intensity * 50)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"noise=c0s={noise_strength}:c0f=t+u",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"Applied {grain_type} grain at {intensity}")
                return OverlayResult(
                    success=True, 
                    output_path=output_path,
                    overlays_applied=["grain"]
                )
            else:
                return OverlayResult(
                    success=False, output_path="", overlays_applied=[],
                    error="Grain filter failed"
                )
        except Exception as e:
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error=str(e)
            )
    
    def apply_letterbox(
        self,
        video_path: str,
        aspect_ratio: float = 2.35,
        bar_color: str = "black",
        output_path: str = None
    ) -> OverlayResult:
        """
        Apply letterbox bars for cinematic aspect ratio.
        
        Args:
            video_path: Input video
            aspect_ratio: Target ratio (2.35 for cinemascope)
            bar_color: Color of bars
            output_path: Output path
            
        Returns:
            OverlayResult
        """
        if not Path(video_path).exists():
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"letterbox_{Path(video_path).name}")
        
        # Calculate bar height
        # pad filter: pad=width:height:x:y:color
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"pad=iw:iw/{aspect_ratio}:(ow-iw)/2:(oh-ih)/2:{bar_color}",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return OverlayResult(
                    success=True,
                    output_path=output_path,
                    overlays_applied=["letterbox"]
                )
            else:
                return OverlayResult(
                    success=False, output_path="", overlays_applied=[],
                    error="Letterbox failed"
                )
        except Exception as e:
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error=str(e)
            )
    
    def apply_vignette(
        self,
        video_path: str,
        intensity: float = 0.3,
        output_path: str = None
    ) -> OverlayResult:
        """
        Apply vignette effect.
        
        Args:
            video_path: Input video
            intensity: Vignette intensity (0-1)
            output_path: Output path
            
        Returns:
            OverlayResult
        """
        if not Path(video_path).exists():
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"vignette_{Path(video_path).name}")
        
        # FFmpeg vignette filter
        angle = intensity * 0.5  # PI/5 at max
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"vignette=angle={angle}:mode=forward",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return OverlayResult(
                    success=True,
                    output_path=output_path,
                    overlays_applied=["vignette"]
                )
            else:
                return OverlayResult(
                    success=False, output_path="", overlays_applied=[],
                    error="Vignette failed"
                )
        except Exception as e:
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error=str(e)
            )
    
    def apply_color_temperature(
        self,
        video_path: str,
        temperature: int = 0,
        output_path: str = None
    ) -> OverlayResult:
        """
        Adjust color temperature.
        
        Args:
            video_path: Input video
            temperature: -100 (cool) to +100 (warm)
            output_path: Output path
            
        Returns:
            OverlayResult
        """
        if not Path(video_path).exists():
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"temp_{Path(video_path).name}")
        
        # Use colorbalance filter
        # Positive = warm (more red/yellow), negative = cool (more blue)
        rs = temperature / 200  # Red shadows
        bh = -temperature / 200  # Blue highlights
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"colorbalance=rs={rs}:bh={bh}",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return OverlayResult(
                    success=True,
                    output_path=output_path,
                    overlays_applied=["temperature"]
                )
            else:
                return OverlayResult(
                    success=False, output_path="", overlays_applied=[],
                    error="Temperature failed"
                )
        except Exception as e:
            return OverlayResult(
                success=False, output_path="", overlays_applied=[],
                error=str(e)
            )
    
    def apply_preset(
        self,
        video_path: str,
        preset: str = "cinematic",
        output_path: str = None
    ) -> OverlayResult:
        """
        Apply a preset combination of overlays.
        
        Args:
            video_path: Input video
            preset: Preset name
            output_path: Final output
            
        Returns:
            OverlayResult
        """
        if preset not in OVERLAY_PRESETS:
            preset = "cinematic"
        
        config = OVERLAY_PRESETS[preset]
        current_path = video_path
        applied = []
        
        # Apply grain
        if config.get("grain"):
            result = self.apply_grain(
                current_path,
                intensity=config["grain"]["intensity"],
                grain_type=config["grain"]["type"]
            )
            if result.success:
                current_path = result.output_path
                applied.append("grain")
        
        # Apply letterbox
        if config.get("letterbox"):
            result = self.apply_letterbox(
                current_path,
                aspect_ratio=config["letterbox"]["ratio"]
            )
            if result.success:
                current_path = result.output_path
                applied.append("letterbox")
        
        # Apply vignette (always subtle for cinematic)
        if preset == "cinematic":
            result = self.apply_vignette(current_path, intensity=0.2)
            if result.success:
                current_path = result.output_path
                applied.append("vignette")
        
        # Rename to final output
        if output_path and current_path != video_path:
            Path(current_path).rename(output_path)
            current_path = output_path
        
        return OverlayResult(
            success=True,
            output_path=current_path,
            overlays_applied=applied
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_cinematic_look(video_path: str, output_path: str = None) -> str:
    """Apply cinematic preset to video."""
    engine = VisualOverlayEngine()
    result = engine.apply_preset(video_path, "cinematic", output_path)
    return result.output_path if result.success else video_path


def get_available_presets() -> List[str]:
    """Get list of available overlay presets."""
    return list(OVERLAY_PRESETS.keys())
