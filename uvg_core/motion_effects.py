# uvg_core/motion_effects.py
"""
Motion Effects Engine for UVG MAX.

Camera motion and zoom effects:
- Camera shake
- Smart zoom (face detection)
- Ken Burns
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Motion effect result."""
    success: bool
    output_path: str
    effect_applied: str
    error: str = ""


class MotionEffectsEngine:
    """
    Apply motion effects to video.
    
    Features:
    - Camera shake
    - Smart zoom with face detection
    - Ken Burns (zoom/pan)
    - Parallax (future hook)
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize motion effects engine.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/motion")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_camera_shake(
        self,
        video_path: str,
        intensity: float = 0.3,
        frequency: float = 15,
        output_path: str = None
    ) -> MotionResult:
        """
        Add subtle camera shake effect.
        
        Args:
            video_path: Input video
            intensity: Shake intensity (0-1)
            frequency: Shake frequency Hz
            output_path: Output path
            
        Returns:
            MotionResult
        """
        if not Path(video_path).exists():
            return MotionResult(
                success=False, output_path="", effect_applied="",
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"shake_{Path(video_path).name}")
        
        # Use FFmpeg's vidstabtransform for shake
        # Or simulate with zoompan
        amplitude = int(intensity * 10)
        
        # Create shake effect with zoompan
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"zoompan=z='1+0.002*sin({frequency}*PI*t)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"Added camera shake: intensity={intensity}")
                return MotionResult(
                    success=True,
                    output_path=output_path,
                    effect_applied="shake"
                )
            else:
                return MotionResult(
                    success=False, output_path="", effect_applied="",
                    error="Shake failed"
                )
        except Exception as e:
            return MotionResult(
                success=False, output_path="", effect_applied="",
                error=str(e)
            )
    
    def add_ken_burns(
        self,
        video_path: str,
        direction: str = "zoom_in",
        speed: float = 1.0,
        output_path: str = None
    ) -> MotionResult:
        """
        Add Ken Burns zoom/pan effect.
        
        Args:
            video_path: Input video
            direction: "zoom_in", "zoom_out", "pan_left", "pan_right"
            speed: Speed multiplier
            output_path: Output path
            
        Returns:
            MotionResult
        """
        if not Path(video_path).exists():
            return MotionResult(
                success=False, output_path="", effect_applied="",
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"kb_{Path(video_path).name}")
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            duration = 5.0
        
        # Calculate zoom parameters
        zoom_speed = 0.001 * speed
        
        if direction == "zoom_in":
            zoom_expr = f"1+{zoom_speed}*on"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif direction == "zoom_out":
            zoom_expr = f"1.2-{zoom_speed}*on"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif direction == "pan_left":
            zoom_expr = "1.1"
            x_expr = f"iw*0.1*on/{duration}/25"
            y_expr = "ih/2-(ih/zoom/2)"
        elif direction == "pan_right":
            zoom_expr = "1.1"
            x_expr = f"iw*0.9-iw*0.1*on/{duration}/25"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            zoom_expr = f"1+{zoom_speed}*on"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={int(duration*25)}:s=1920x1080",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return MotionResult(
                    success=True,
                    output_path=output_path,
                    effect_applied=f"ken_burns_{direction}"
                )
            else:
                return MotionResult(
                    success=False, output_path="", effect_applied="",
                    error="Ken Burns failed"
                )
        except Exception as e:
            return MotionResult(
                success=False, output_path="", effect_applied="",
                error=str(e)
            )
    
    def smart_zoom(
        self,
        video_path: str,
        target_point: Tuple[float, float] = None,
        zoom_level: float = 1.2,
        output_path: str = None
    ) -> MotionResult:
        """
        Smart zoom towards detected face or specified point.
        
        Args:
            video_path: Input video
            target_point: (x, y) normalized 0-1, or None for auto
            zoom_level: Zoom factor
            output_path: Output path
            
        Returns:
            MotionResult
        """
        if not Path(video_path).exists():
            return MotionResult(
                success=False, output_path="", effect_applied="",
                error="Video not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"smartzoom_{Path(video_path).name}")
        
        # Detect face if no target specified
        if target_point is None:
            target_point = self._detect_focus_point(video_path)
        
        x_norm, y_norm = target_point
        
        # Build zoom filter
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"scale=iw*{zoom_level}:ih*{zoom_level},crop=iw/{zoom_level}:ih/{zoom_level}:{int(x_norm*100)}:{int(y_norm*100)}",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return MotionResult(
                    success=True,
                    output_path=output_path,
                    effect_applied="smart_zoom"
                )
            else:
                # Fallback: simple center zoom
                return self.add_ken_burns(video_path, "zoom_in", 0.5, output_path)
        except Exception as e:
            return MotionResult(
                success=False, output_path="", effect_applied="",
                error=str(e)
            )
    
    def _detect_focus_point(
        self,
        video_path: str
    ) -> Tuple[float, float]:
        """
        Detect focus point (face or center).
        
        Args:
            video_path: Video path
            
        Returns:
            (x, y) normalized coordinates
        """
        try:
            from .object_detector import ObjectDetector
            
            # Extract frame
            frame_path = str(self.output_dir / "temp_frame.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", "1",
                "-i", video_path,
                "-vframes", "1",
                frame_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)
            
            if Path(frame_path).exists():
                detector = ObjectDetector()
                persons = detector.detect_persons(frame_path)
                
                if persons:
                    # Return center of first person
                    p = persons[0]
                    x = (p["x1"] + p["x2"]) / 2
                    y = (p["y1"] + p["y2"]) / 2
                    return (x, y)
                
                Path(frame_path).unlink()
        except Exception:
            pass
        
        # Default to center
        return (0.5, 0.5)
    
    def _get_video_duration(self, video_path: str) -> float:
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
            return 5.0
    
    def parallax_effect(
        self,
        video_path: str,
        depth_map: str = None,
        output_path: str = None
    ) -> MotionResult:
        """
        Apply parallax 2.5D effect (HOOK ONLY).
        
        Args:
            video_path: Input video
            depth_map: Depth map image
            output_path: Output path
            
        Returns:
            MotionResult
        """
        # Future implementation
        raise NotImplementedError("Parallax effect coming in future update")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def add_shake(video_path: str, intensity: float = 0.3) -> str:
    """Add camera shake to video."""
    engine = MotionEffectsEngine()
    result = engine.add_camera_shake(video_path, intensity)
    return result.output_path if result.success else video_path


def add_ken_burns(video_path: str, direction: str = "zoom_in") -> str:
    """Add Ken Burns effect."""
    engine = MotionEffectsEngine()
    result = engine.add_ken_burns(video_path, direction)
    return result.output_path if result.success else video_path
