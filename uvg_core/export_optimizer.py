# uvg_core/export_optimizer.py
"""
Export Optimizer for UVG MAX.

Platform-specific encoding:
- YouTube
- TikTok
- Instagram
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Export result."""
    success: bool
    output_path: str
    platform: str
    resolution: str
    bitrate: str
    error: str = ""


# Platform presets
PLATFORM_PRESETS = {
    "youtube": {
        "resolution": "1920x1080",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "15M",
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "preset": "slow",
        "crf": 18,
        "pix_fmt": "yuv420p",
    },
    "youtube_4k": {
        "resolution": "3840x2160",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "45M",
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "preset": "slow",
        "crf": 17,
        "pix_fmt": "yuv420p",
    },
    "tiktok": {
        "resolution": "1080x1920",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "8M",
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "preset": "medium",
        "crf": 23,
        "pix_fmt": "yuv420p",
    },
    "instagram_reels": {
        "resolution": "1080x1920",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "6M",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "preset": "medium",
        "crf": 23,
        "pix_fmt": "yuv420p",
    },
    "instagram_feed": {
        "resolution": "1080x1350",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "6M",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "preset": "medium",
        "crf": 23,
        "pix_fmt": "yuv420p",
    },
    "twitter": {
        "resolution": "1280x720",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "5M",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "preset": "medium",
        "crf": 23,
        "pix_fmt": "yuv420p",
    },
    "web": {
        "resolution": "1920x1080",
        "fps": 30,
        "video_codec": "libx264",
        "video_bitrate": "8M",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "preset": "fast",
        "crf": 23,
        "pix_fmt": "yuv420p",
    },
}


class ExportOptimizer:
    """
    Platform-optimized video export.
    
    Features:
    - Platform presets
    - Resolution scaling
    - Bitrate optimization
    - Crop for aspect ratios
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize export optimizer.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/export")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        video_path: str,
        platform: str = "youtube",
        output_path: str = None
    ) -> ExportResult:
        """
        Export video for specific platform.
        
        Args:
            video_path: Input video
            platform: Target platform
            output_path: Output path
            
        Returns:
            ExportResult
        """
        if not Path(video_path).exists():
            return ExportResult(
                success=False, output_path="", platform=platform,
                resolution="", bitrate="", error="Video not found"
            )
        
        if platform not in PLATFORM_PRESETS:
            platform = "youtube"
        
        preset = PLATFORM_PRESETS[platform]
        
        if output_path is None:
            stem = Path(video_path).stem
            output_path = str(self.output_dir / f"{stem}_{platform}.mp4")
        
        # Build FFmpeg command
        width, height = preset["resolution"].split("x")
        
        video_filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            f"fps={preset['fps']}",
            f"format={preset['pix_fmt']}"
        ]
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", ",".join(video_filters),
            "-c:v", preset["video_codec"],
            "-preset", preset["preset"],
            "-crf", str(preset["crf"]),
            "-b:v", preset["video_bitrate"],
            "-c:a", preset["audio_codec"],
            "-b:a", preset["audio_bitrate"],
            "-movflags", "+faststart",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode == 0:
                return ExportResult(
                    success=True,
                    output_path=output_path,
                    platform=platform,
                    resolution=preset["resolution"],
                    bitrate=preset["video_bitrate"]
                )
            else:
                return ExportResult(
                    success=False, output_path="", platform=platform,
                    resolution="", bitrate="",
                    error="Encoding failed"
                )
        except Exception as e:
            return ExportResult(
                success=False, output_path="", platform=platform,
                resolution="", bitrate="",
                error=str(e)
            )
    
    def export_all_platforms(
        self,
        video_path: str,
        platforms: list = None
    ) -> Dict[str, ExportResult]:
        """
        Export for multiple platforms.
        
        Args:
            video_path: Input video
            platforms: List of platforms
            
        Returns:
            Dict of platform -> ExportResult
        """
        if platforms is None:
            platforms = ["youtube", "tiktok", "instagram_reels"]
        
        results = {}
        for platform in platforms:
            results[platform] = self.export(video_path, platform)
        
        return results
    
    def crop_for_aspect(
        self,
        video_path: str,
        aspect_ratio: str = "9:16",
        output_path: str = None
    ) -> ExportResult:
        """
        Crop video to specific aspect ratio.
        
        Args:
            video_path: Input video
            aspect_ratio: Target ratio (e.g., "9:16", "4:5", "1:1")
            output_path: Output path
            
        Returns:
            ExportResult
        """
        if not Path(video_path).exists():
            return ExportResult(
                success=False, output_path="", platform="crop",
                resolution="", bitrate="", error="Video not found"
            )
        
        if output_path is None:
            stem = Path(video_path).stem
            ratio_str = aspect_ratio.replace(":", "x")
            output_path = str(self.output_dir / f"{stem}_{ratio_str}.mp4")
        
        # Parse aspect ratio
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
        
        # Build crop filter
        if w_ratio < h_ratio:
            # Portrait - crop width
            crop_filter = f"crop=ih*{w_ratio}/{h_ratio}:ih"
        else:
            # Landscape - crop height
            crop_filter = f"crop=iw:iw*{h_ratio}/{w_ratio}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", crop_filter,
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return ExportResult(
                    success=True,
                    output_path=output_path,
                    platform="crop",
                    resolution=aspect_ratio,
                    bitrate=""
                )
        except Exception:
            pass
        
        return ExportResult(
            success=False, output_path="", platform="crop",
            resolution="", bitrate="", error="Crop failed"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def export_for_platform(video_path: str, platform: str = "youtube") -> str:
    """Export video for platform."""
    optimizer = ExportOptimizer()
    result = optimizer.export(video_path, platform)
    return result.output_path if result.success else video_path


def get_available_platforms() -> list:
    """Get available export platforms."""
    return list(PLATFORM_PRESETS.keys())
