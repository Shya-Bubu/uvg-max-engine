# uvg_core/speed_ramp.py
"""
Speed Ramp VFX Engine for UVG MAX.

FFmpeg-based speed ramping:
- Pure FFmpeg, no Blender/ML
- Smooth speed transitions
- TikTok/cinematic essential

JSON field support:
{
    "speed_ramp": {
        "in": 1.0,
        "out": 1.2,
        "ramp_duration": 0.5
    }
}
"""

import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SpeedRampResult:
    """Result of speed ramp operation."""
    success: bool
    output_path: str
    speed_in: float
    speed_out: float
    error: str = ""


class SpeedRampEngine:
    """
    FFmpeg-based speed ramping for cinematic effects.
    
    Uses setpts filter for smooth speed transitions:
    - Speed up: setpts=PTS/1.3
    - Slow down: setpts=PTS*1.3
    - Ramp: smooth transition between speeds
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize speed ramp engine.
        
        Args:
            output_dir: Output directory for processed videos
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/speed_ramp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_speed_ramp(
        self,
        video_path: str,
        speed_in: float = 1.0,
        speed_out: float = 1.2,
        ramp_duration: float = 0.5,
        output_path: Optional[str] = None
    ) -> SpeedRampResult:
        """
        Apply speed ramp effect to video.
        
        Args:
            video_path: Input video path
            speed_in: Starting speed multiplier (1.0 = normal)
            speed_out: Ending speed multiplier
            ramp_duration: Duration of speed transition in seconds
            output_path: Output path (auto-generated if None)
            
        Returns:
            SpeedRampResult
        """
        if not Path(video_path).exists():
            return SpeedRampResult(
                success=False,
                output_path="",
                speed_in=speed_in,
                speed_out=speed_out,
                error="Video file not found"
            )
        
        if output_path is None:
            stem = Path(video_path).stem
            output_path = str(self.output_dir / f"{stem}_ramp_{speed_in}_{speed_out}.mp4")
        
        # Get video duration
        duration = self._get_duration(video_path)
        if duration <= 0:
            duration = 5.0
        
        try:
            # For simple speed change (no ramp)
            if speed_in == speed_out:
                return self._apply_constant_speed(video_path, speed_in, output_path)
            
            # For ramping effect
            return self._apply_ramp(
                video_path, 
                speed_in, 
                speed_out, 
                ramp_duration, 
                duration,
                output_path
            )
            
        except Exception as e:
            logger.error(f"Speed ramp failed: {e}")
            return SpeedRampResult(
                success=False,
                output_path="",
                speed_in=speed_in,
                speed_out=speed_out,
                error=str(e)
            )
    
    def _apply_constant_speed(
        self,
        video_path: str,
        speed: float,
        output_path: str
    ) -> SpeedRampResult:
        """Apply constant speed change."""
        # setpts: PTS/speed for speedup, PTS*speed for slowdown
        if speed > 1.0:
            pts_expr = f"PTS/{speed}"
        else:
            pts_expr = f"PTS*{1/speed}"
        
        # Audio tempo adjustment (0.5-2.0 range)
        atempo = max(0.5, min(2.0, speed))
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex",
            f"[0:v]setpts={pts_expr}[v];[0:a]atempo={atempo}[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode == 0 and Path(output_path).exists():
            logger.info(f"Applied constant speed {speed}x to {video_path}")
            return SpeedRampResult(
                success=True,
                output_path=output_path,
                speed_in=speed,
                speed_out=speed
            )
        else:
            return SpeedRampResult(
                success=False,
                output_path="",
                speed_in=speed,
                speed_out=speed,
                error=result.stderr.decode()[:200]
            )
    
    def _apply_ramp(
        self,
        video_path: str,
        speed_in: float,
        speed_out: float,
        ramp_duration: float,
        total_duration: float,
        output_path: str
    ) -> SpeedRampResult:
        """
        Apply smooth speed ramp using FFmpeg expression.
        
        Creates a smooth transition from speed_in to speed_out.
        """
        # Calculate ramp parameters
        ramp_start = total_duration - ramp_duration
        
        # Build setpts expression for smooth ramp
        # Speed changes linearly from speed_in to speed_out during ramp_duration
        # Before ramp: constant speed_in
        # During ramp: interpolate
        # This is an approximation - true smooth ramp needs complex expressions
        
        # Simplified: use two-pass approach
        # Pass 1: Apply speed_in to first part
        # Pass 2: Apply speed_out to last part with crossfade
        
        # For now, use a simpler approach: average speed with emphasis
        avg_speed = (speed_in + speed_out) / 2
        
        # Build expression that ramps
        # t = current time, d = duration
        # speed = speed_in + (speed_out - speed_in) * (t - ramp_start) / ramp_duration
        speed_diff = speed_out - speed_in
        
        # FFmpeg setpts expression for ramping
        # if(lt(T, ramp_start), PTS/speed_in, PTS/lerp)
        pts_expr = (
            f"if(lt(T,{ramp_start}),"
            f"PTS/{speed_in},"
            f"PTS/({speed_in}+{speed_diff}*(T-{ramp_start})/{ramp_duration}))"
        )
        
        # Audio: use average tempo (atempo doesn't support expressions)
        atempo = max(0.5, min(2.0, avg_speed))
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex",
            f"[0:v]setpts={pts_expr}[v];[0:a]atempo={atempo}[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode == 0 and Path(output_path).exists():
            logger.info(f"Applied speed ramp {speed_in}x -> {speed_out}x")
            return SpeedRampResult(
                success=True,
                output_path=output_path,
                speed_in=speed_in,
                speed_out=speed_out
            )
        else:
            # Fallback to constant average speed
            logger.warning("Ramp expression failed, using average speed")
            return self._apply_constant_speed(video_path, avg_speed, output_path)
    
    def apply_slow_motion(
        self,
        video_path: str,
        speed: float = 0.5,
        output_path: Optional[str] = None
    ) -> SpeedRampResult:
        """Apply slow motion effect."""
        return self.apply_speed_ramp(video_path, speed, speed, 0, output_path)
    
    def apply_fast_forward(
        self,
        video_path: str,
        speed: float = 2.0,
        output_path: Optional[str] = None
    ) -> SpeedRampResult:
        """Apply fast forward effect."""
        return self.apply_speed_ramp(video_path, speed, speed, 0, output_path)
    
    def apply_dramatic_slowdown(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> SpeedRampResult:
        """Apply cinematic slowdown effect (1.0x -> 0.5x)."""
        return self.apply_speed_ramp(video_path, 1.0, 0.5, 1.0, output_path)
    
    def apply_energy_speedup(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> SpeedRampResult:
        """Apply TikTok-style energy speedup (1.0x -> 1.5x)."""
        return self.apply_speed_ramp(video_path, 1.0, 1.5, 0.5, output_path)
    
    def _get_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception:
            return 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_speed_ramp(
    video_path: str,
    speed_in: float = 1.0,
    speed_out: float = 1.2
) -> str:
    """Apply speed ramp and return output path."""
    engine = SpeedRampEngine()
    result = engine.apply_speed_ramp(video_path, speed_in, speed_out)
    return result.output_path if result.success else video_path


def apply_slow_motion(video_path: str, speed: float = 0.5) -> str:
    """Apply slow motion effect."""
    engine = SpeedRampEngine()
    result = engine.apply_slow_motion(video_path, speed)
    return result.output_path if result.success else video_path


def apply_fast_forward(video_path: str, speed: float = 2.0) -> str:
    """Apply fast forward effect."""
    engine = SpeedRampEngine()
    result = engine.apply_fast_forward(video_path, speed)
    return result.output_path if result.success else video_path
