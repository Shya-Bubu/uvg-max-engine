# uvg_core/multi_clip_composer.py
"""
Multi-Clip Composer for UVG MAX.

Layer multiple video clips:
- Background
- Midground
- Foreground overlays
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CompositeResult:
    """Composition result."""
    success: bool
    output_path: str
    layers_used: int
    error: str = ""


# Blend modes
BLEND_MODES = {
    "normal": "overlay",
    "screen": "screen",
    "multiply": "multiply",
    "overlay": "overlay",
    "soft_light": "softlight",
    "hard_light": "hardlight",
    "add": "add",
}


class MultiClipComposer:
    """
    Compose multiple video layers.
    
    Features:
    - Background + midground + foreground
    - Blend modes
    - Opacity control
    - Position/scale
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize composer.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/composite")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compose(
        self,
        background: str,
        midground: str = None,
        foreground: str = None,
        mid_opacity: float = 1.0,
        fg_opacity: float = 0.3,
        mid_blend: str = "normal",
        fg_blend: str = "screen",
        output_path: str = None
    ) -> CompositeResult:
        """
        Compose multiple video layers.
        
        Args:
            background: Background video (required)
            midground: Middle layer video
            foreground: Overlay video (particles, grain)
            mid_opacity: Midground opacity 0-1
            fg_opacity: Foreground opacity 0-1
            mid_blend: Midground blend mode
            fg_blend: Foreground blend mode
            output_path: Output path
            
        Returns:
            CompositeResult
        """
        if not Path(background).exists():
            return CompositeResult(
                success=False, output_path="", layers_used=0,
                error="Background not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"composite_{Path(background).name}")
        
        layers_used = 1
        inputs = ["-i", background]
        filter_parts = []
        
        # Start with background
        current_stream = "[0:v]"
        
        # Add midground if provided
        if midground and Path(midground).exists():
            inputs.extend(["-i", midground])
            layers_used += 1
            blend_mode = BLEND_MODES.get(mid_blend, "overlay")
            
            filter_parts.append(
                f"{current_stream}[1:v]blend=all_mode={blend_mode}:all_opacity={mid_opacity}[mid]"
            )
            current_stream = "[mid]"
        
        # Add foreground if provided
        if foreground and Path(foreground).exists():
            inputs.extend(["-i", foreground])
            fg_index = layers_used
            layers_used += 1
            blend_mode = BLEND_MODES.get(fg_blend, "screen")
            
            filter_parts.append(
                f"{current_stream}[{fg_index}:v]blend=all_mode={blend_mode}:all_opacity={fg_opacity}[out]"
            )
            current_stream = "[out]"
        else:
            # Rename last stream to [out]
            filter_parts.append(f"{current_stream}null[out]")
        
        # Build command
        filter_complex = ";".join(filter_parts) if filter_parts else "null"
        
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return CompositeResult(
                    success=True,
                    output_path=output_path,
                    layers_used=layers_used
                )
            else:
                # Fallback: just copy background
                return CompositeResult(
                    success=True,
                    output_path=background,
                    layers_used=1,
                    error="Blend failed, using background only"
                )
        except Exception as e:
            return CompositeResult(
                success=False, output_path="", layers_used=0,
                error=str(e)
            )
    
    def overlay_at_position(
        self,
        base: str,
        overlay: str,
        position: Tuple[int, int] = (0, 0),
        scale: float = 1.0,
        opacity: float = 1.0,
        output_path: str = None
    ) -> CompositeResult:
        """
        Overlay video at specific position.
        
        Args:
            base: Base video
            overlay: Overlay video
            position: (x, y) position
            scale: Scale factor
            opacity: Opacity 0-1
            output_path: Output path
            
        Returns:
            CompositeResult
        """
        if not Path(base).exists() or not Path(overlay).exists():
            return CompositeResult(
                success=False, output_path="", layers_used=0,
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"overlay_{Path(base).name}")
        
        x, y = position
        
        # Build filter
        if scale != 1.0:
            scale_filter = f"[1:v]scale=iw*{scale}:ih*{scale}[ovr];"
            overlay_stream = "[ovr]"
        else:
            scale_filter = ""
            overlay_stream = "[1:v]"
        
        if opacity < 1.0:
            opacity_filter = f"{overlay_stream}format=rgba,colorchannelmixer=aa={opacity}[ovr2];"
            overlay_stream = "[ovr2]"
        else:
            opacity_filter = ""
        
        filter_complex = f"{scale_filter}{opacity_filter}[0:v]{overlay_stream}overlay={x}:{y}[out]"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", base,
            "-i", overlay,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a?",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return CompositeResult(
                    success=True,
                    output_path=output_path,
                    layers_used=2
                )
        except Exception as e:
            return CompositeResult(
                success=False, output_path="", layers_used=0,
                error=str(e)
            )
        
        return CompositeResult(
            success=False, output_path="", layers_used=0,
            error="Overlay failed"
        )
    
    def add_particle_overlay(
        self,
        video_path: str,
        particle_type: str = "dust",
        opacity: float = 0.2,
        output_path: str = None
    ) -> CompositeResult:
        """
        Add particle overlay (dust, bokeh, snow).
        
        Args:
            video_path: Input video
            particle_type: Type of particles
            opacity: Overlay opacity
            output_path: Output path
            
        Returns:
            CompositeResult
        """
        # Check for overlay files
        overlay_dir = Path("overlays")
        particle_file = overlay_dir / f"{particle_type}.webm"
        
        if not particle_file.exists():
            # Return original video
            return CompositeResult(
                success=True,
                output_path=video_path,
                layers_used=1,
                error=f"Particle overlay not found: {particle_type}"
            )
        
        return self.compose(
            video_path,
            foreground=str(particle_file),
            fg_opacity=opacity,
            fg_blend="screen",
            output_path=output_path
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compose_layers(
    background: str,
    midground: str = None,
    foreground: str = None
) -> str:
    """Compose video layers."""
    composer = MultiClipComposer()
    result = composer.compose(background, midground, foreground)
    return result.output_path if result.success else background
