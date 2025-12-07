"""
UVG MAX Clip Preparer Module

Motion engine with camera path planning and fallback handling.
Includes Ken Burns, freeze-frame extension, pan-scan, and blur-fill.
"""

import logging
import subprocess
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CameraPath:
    """Camera motion path with keyframes."""
    keyframes: List[Dict[str, float]] = field(default_factory=list)  # t, zoom, pan_x, pan_y
    curve_type: str = "ease_in_out"  # bezier, linear, ease_in_out
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keyframes": self.keyframes,
            "curve_type": self.curve_type,
        }


@dataclass 
class PrepareResult:
    """Result of clip preparation."""
    success: bool
    input_path: str
    output_path: str
    motion_type: str
    camera_path: Optional[CameraPath] = None
    error: str = ""


# =============================================================================
# MOTION TYPES
# =============================================================================

MOTION_TYPES = [
    "slow-zoom-in",
    "slow-zoom-out", 
    "pan-left",
    "pan-right",
    "tilt-up",
    "tilt-down",
    "dolly-in",
    "static",
    "ken-burns",
]


class ClipPreparer:
    """
    Prepares clips with motion, LUTs, and corrections.
    
    Features:
    - Deterministic motion (seeded by SHA256 + scene_idx)
    - Camera path planning with easing
    - Fallback motion for static/short clips
    - Portrait smart cropping
    - GPU/CPU encoding paths
    """
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 target_width: int = 1080,
                 target_height: int = 1920,
                 fps: int = 30,
                 use_cuda: bool = True,
                 lut_dir: Optional[Path] = None):
        """
        Initialize clip preparer.
        
        Args:
            output_dir: Directory for prepared clips
            target_width: Output width
            target_height: Output height
            fps: Output framerate
            use_cuda: Use GPU encoding if available
            lut_dir: Directory containing LUT files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/prepared")
        self.target_width = target_width
        self.target_height = target_height
        self.fps = fps
        self.use_cuda = use_cuda
        self.lut_dir = Path(lut_dir) if lut_dir else Path("./assets/luts")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_clip_info(self, clip_path: str) -> Dict[str, Any]:
        """Get clip information using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration,r_frame_rate",
                "-of", "json",
                clip_path
            ], capture_output=True, text=True, timeout=30)
            
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            
            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "duration": float(stream.get("duration", 0)),
                "fps": fps,
            }
        except Exception as e:
            logger.warning(f"Could not get clip info: {e}")
            return {"width": 0, "height": 0, "duration": 0, "fps": 30}
    
    def _get_deterministic_motion(self, 
                                   clip_path: str,
                                   scene_idx: int,
                                   preferred: Optional[str] = None) -> str:
        """Get deterministic motion type based on hash."""
        if preferred and preferred in MOTION_TYPES:
            return preferred
        
        # Create deterministic seed
        hash_input = f"{clip_path}_{scene_idx}"
        hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)
        
        # Select motion type
        motion_types = ["slow-zoom-in", "slow-zoom-out", "pan-left", "pan-right", "dolly-in"]
        return motion_types[hash_val % len(motion_types)]
    
    def _generate_camera_path(self,
                               motion_type: str,
                               duration: float,
                               intensity: float = 1.0) -> CameraPath:
        """Generate camera path keyframes."""
        path = CameraPath(curve_type="ease_in_out")
        
        # Base zoom range based on intensity
        zoom_range = 0.1 * intensity
        pan_range = 0.15 * intensity
        
        if motion_type == "slow-zoom-in":
            path.keyframes = [
                {"t": 0, "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5},
                {"t": duration, "zoom": 1.0 + zoom_range, "pan_x": 0.5, "pan_y": 0.5},
            ]
        
        elif motion_type == "slow-zoom-out":
            path.keyframes = [
                {"t": 0, "zoom": 1.0 + zoom_range, "pan_x": 0.5, "pan_y": 0.5},
                {"t": duration, "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5},
            ]
        
        elif motion_type == "pan-left":
            path.keyframes = [
                {"t": 0, "zoom": 1.05, "pan_x": 0.5 + pan_range, "pan_y": 0.5},
                {"t": duration, "zoom": 1.05, "pan_x": 0.5 - pan_range, "pan_y": 0.5},
            ]
        
        elif motion_type == "pan-right":
            path.keyframes = [
                {"t": 0, "zoom": 1.05, "pan_x": 0.5 - pan_range, "pan_y": 0.5},
                {"t": duration, "zoom": 1.05, "pan_x": 0.5 + pan_range, "pan_y": 0.5},
            ]
        
        elif motion_type == "tilt-up":
            path.keyframes = [
                {"t": 0, "zoom": 1.05, "pan_x": 0.5, "pan_y": 0.5 + pan_range},
                {"t": duration, "zoom": 1.05, "pan_x": 0.5, "pan_y": 0.5 - pan_range},
            ]
        
        elif motion_type == "tilt-down":
            path.keyframes = [
                {"t": 0, "zoom": 1.05, "pan_x": 0.5, "pan_y": 0.5 - pan_range},
                {"t": duration, "zoom": 1.05, "pan_x": 0.5, "pan_y": 0.5 + pan_range},
            ]
        
        elif motion_type == "dolly-in":
            path.keyframes = [
                {"t": 0, "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5},
                {"t": duration * 0.5, "zoom": 1.0 + zoom_range * 0.7, "pan_x": 0.5, "pan_y": 0.48},
                {"t": duration, "zoom": 1.0 + zoom_range, "pan_x": 0.5, "pan_y": 0.45},
            ]
        
        elif motion_type == "ken-burns":
            # Random-ish Ken Burns
            path.keyframes = [
                {"t": 0, "zoom": 1.0, "pan_x": 0.4, "pan_y": 0.4},
                {"t": duration, "zoom": 1.15, "pan_x": 0.6, "pan_y": 0.6},
            ]
        
        else:  # static
            path.keyframes = [
                {"t": 0, "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5},
                {"t": duration, "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5},
            ]
        
        return path
    
    def _build_zoompan_filter(self, 
                               camera_path: CameraPath,
                               input_width: int,
                               input_height: int,
                               duration: float) -> str:
        """Build FFmpeg zoompan filter from camera path."""
        if not camera_path.keyframes:
            return f"scale={self.target_width}:{self.target_height}"
        
        # Simple two-keyframe interpolation
        start = camera_path.keyframes[0]
        end = camera_path.keyframes[-1]
        
        zoom_start = start.get("zoom", 1.0)
        zoom_end = end.get("zoom", 1.0)
        pan_x_start = start.get("pan_x", 0.5)
        pan_x_end = end.get("pan_x", 0.5)
        pan_y_start = start.get("pan_y", 0.5)
        pan_y_end = end.get("pan_y", 0.5)
        
        total_frames = int(duration * self.fps)
        
        # Zoompan expression
        zoom_expr = f"zoom+({zoom_end-zoom_start}/{total_frames})"
        x_expr = f"(iw-iw/zoom)/2 + (iw/zoom)*({pan_x_start}+({pan_x_end}-{pan_x_start})*on/{total_frames}-0.5)"
        y_expr = f"(ih-ih/zoom)/2 + (ih/zoom)*({pan_y_start}+({pan_y_end}-{pan_y_start})*on/{total_frames}-0.5)"
        
        zoompan = (
            f"zoompan=z='{zoom_start}+({zoom_end-zoom_start})*on/{total_frames}':"
            f"x='(iw-iw/zoom)*{pan_x_start}+(iw-iw/zoom)*({pan_x_end-pan_x_start})*on/{total_frames}':"
            f"y='(ih-ih/zoom)*{pan_y_start}+(ih-ih/zoom)*({pan_y_end-pan_y_start})*on/{total_frames}':"
            f"d={total_frames}:s={self.target_width}x{self.target_height}:fps={self.fps}"
        )
        
        return zoompan
    
    def _apply_ken_burns(self, 
                          clip_path: str,
                          duration: float,
                          output_path: str) -> bool:
        """Apply Ken Burns effect to static image/clip."""
        camera_path = self._generate_camera_path("ken-burns", duration)
        info = self._get_clip_info(clip_path)
        
        zoompan = self._build_zoompan_filter(
            camera_path, 
            info["width"], 
            info["height"],
            duration
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1" if clip_path.lower().endswith(('.jpg', '.png')) else "0",
            "-i", clip_path,
            "-vf", zoompan,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            return True
        except Exception as e:
            logger.warning(f"Ken Burns failed: {e}")
            return False
    
    def _extend_with_freeze_frame(self,
                                   clip_path: str,
                                   target_duration: float,
                                   output_path: str) -> bool:
        """Extend short clip with freeze-frame + parallax."""
        info = self._get_clip_info(clip_path)
        current_duration = info.get("duration", 0)
        
        if current_duration >= target_duration:
            return False
        
        extend_by = target_duration - current_duration
        
        # Use tpad to clone last frame
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-vf", f"tpad=stop_mode=clone:stop_duration={extend_by}",
            "-c:v", "libx264",
            "-preset", "fast",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            return True
        except Exception:
            return False
    
    def _intelligent_pan_scan(self,
                               clip_path: str,
                               output_path: str) -> bool:
        """Pan-scan for vertical→horizontal or vice versa."""
        info = self._get_clip_info(clip_path)
        src_width = info["width"]
        src_height = info["height"]
        duration = info["duration"]
        
        src_portrait = src_height > src_width
        target_portrait = self.target_height > self.target_width
        
        if src_portrait == target_portrait:
            # No pan-scan needed
            return False
        
        # Generate pan movement
        total_frames = int(duration * self.fps)
        
        if src_portrait and not target_portrait:
            # Vertical clip → horizontal output: pan vertically
            pan_filter = (
                f"crop=w=iw:h=iw*{self.target_height}/{self.target_width}:"
                f"x=0:y=(ih-oh)*t/{duration},"
                f"scale={self.target_width}:{self.target_height}"
            )
        else:
            # Horizontal clip → vertical output: pan horizontally
            pan_filter = (
                f"crop=w=ih*{self.target_width}/{self.target_height}:h=ih:"
                f"x=(iw-ow)*t/{duration}:y=0,"
                f"scale={self.target_width}:{self.target_height}"
            )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-vf", pan_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            return True
        except Exception as e:
            logger.warning(f"Pan-scan failed: {e}")
            return False
    
    def _blur_fill_background(self,
                               clip_path: str,
                               output_path: str) -> bool:
        """Fill weird aspect ratio with blurred background."""
        info = self._get_clip_info(clip_path)
        src_width = info["width"]
        src_height = info["height"]
        
        # Create blurred background + overlay original centered
        filter_complex = (
            f"[0:v]scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=increase,"
            f"crop={self.target_width}:{self.target_height},boxblur=20:5[bg];"
            f"[0:v]scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            return True
        except Exception as e:
            logger.warning(f"Blur-fill failed: {e}")
            return False
    
    def prepare_clip(self,
                     clip_path: str,
                     scene_idx: int,
                     target_duration: float,
                     motion_type: Optional[str] = None,
                     motion_intensity: float = 1.0,
                     lut_name: Optional[str] = None) -> PrepareResult:
        """
        Prepare a clip with motion and corrections.
        
        Args:
            clip_path: Path to input clip
            scene_idx: Scene index for determinism
            target_duration: Target duration
            motion_type: Preferred motion type
            motion_intensity: Motion intensity multiplier
            lut_name: Optional LUT to apply
            
        Returns:
            PrepareResult with output info
        """
        clip_path = Path(clip_path)
        
        if not clip_path.exists():
            return PrepareResult(
                success=False,
                input_path=str(clip_path),
                output_path="",
                motion_type="",
                error="Input clip not found"
            )
        
        info = self._get_clip_info(str(clip_path))
        
        # Check if clip needs special handling
        needs_extend = info["duration"] < target_duration * 0.8
        needs_aspect_fix = self._check_aspect_mismatch(info)
        
        # Determine motion type
        actual_motion = self._get_deterministic_motion(
            str(clip_path), scene_idx, motion_type
        )
        
        # For very static clips, use Ken Burns
        if needs_extend and info["duration"] < 1.0:
            actual_motion = "ken-burns"
        
        # Generate camera path
        camera_path = self._generate_camera_path(
            actual_motion, target_duration, motion_intensity
        )
        
        # Build output path
        output_path = self.output_dir / f"prepared_{scene_idx}_{clip_path.stem}.mp4"
        
        # Build filter chain
        filters = []
        
        # 1. Aspect ratio handling
        if needs_aspect_fix:
            if self._intelligent_pan_scan(str(clip_path), str(output_path)):
                logger.info(f"Applied pan-scan to {clip_path.name}")
            else:
                self._blur_fill_background(str(clip_path), str(output_path))
                logger.info(f"Applied blur-fill to {clip_path.name}")
            # Continue with already-processed file
            clip_path = output_path
            info = self._get_clip_info(str(clip_path))
        
        # 2. Build zoompan for motion
        zoompan = self._build_zoompan_filter(
            camera_path, info["width"], info["height"], target_duration
        )
        filters.append(zoompan)
        
        # 3. Add LUT if specified
        if lut_name and self.lut_dir.exists():
            lut_path = self.lut_dir / f"{lut_name}.cube"
            if lut_path.exists():
                filters.append(f"lut3d={lut_path}")
        
        # Build FFmpeg command
        filter_str = ",".join(filters)
        
        # Choose encoder
        if self.use_cuda:
            encoder = ["-c:v", "h264_nvenc", "-preset", "p4"]
        else:
            encoder = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-vf", filter_str,
            "-t", str(target_duration),
            *encoder,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                # Fallback to CPU
                if self.use_cuda:
                    cmd = [c if c != "h264_nvenc" else "libx264" for c in cmd]
                    cmd = [c for c in cmd if c not in ["-preset", "p4"]]
                    result = subprocess.run(cmd, capture_output=True, timeout=180)
            
            if result.returncode == 0:
                logger.info(f"Prepared {clip_path.name} with {actual_motion} motion")
                return PrepareResult(
                    success=True,
                    input_path=str(clip_path),
                    output_path=str(output_path),
                    motion_type=actual_motion,
                    camera_path=camera_path
                )
            else:
                return PrepareResult(
                    success=False,
                    input_path=str(clip_path),
                    output_path="",
                    motion_type=actual_motion,
                    error=result.stderr[:200] if result.stderr else "Unknown error"
                )
                
        except Exception as e:
            return PrepareResult(
                success=False,
                input_path=str(clip_path),
                output_path="",
                motion_type=actual_motion,
                error=str(e)
            )
    
    def _check_aspect_mismatch(self, info: Dict) -> bool:
        """Check if aspect ratio needs fixing."""
        src_aspect = info["width"] / max(1, info["height"])
        target_aspect = self.target_width / self.target_height
        
        # Allow 10% difference
        return abs(src_aspect - target_aspect) / target_aspect > 0.1


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prepare_clip(clip_path: str,
                 scene_idx: int,
                 duration: float,
                 motion: Optional[str] = None) -> PrepareResult:
    """Prepare a single clip."""
    preparer = ClipPreparer()
    return preparer.prepare_clip(clip_path, scene_idx, duration, motion)
