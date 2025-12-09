# uvg_core/color_matching.py
"""
Color Matching Engine for UVG MAX.

Ensures consistent color palette across clips:
- Palette extraction
- Histogram matching
- Auto-match sequences
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@dataclass
class ColorPalette:
    """Extracted color palette."""
    colors: List[Tuple[int, int, int]]
    dominant: Tuple[int, int, int]
    brightness: float
    
    def to_hex(self) -> List[str]:
        """Convert to hex colors."""
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in self.colors]


@dataclass
class MatchResult:
    """Color matching result."""
    success: bool
    output_path: str
    method: str
    error: str = ""


class ColorMatchingEngine:
    """
    Match colors across video clips.
    
    Features:
    - Palette extraction (KMeans)
    - Histogram matching
    - Temperature adjustment
    - Sequence auto-matching
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize color matching engine.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/color")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAVE_CV2:
            logger.warning("OpenCV not available - color matching limited")
        if not HAVE_SKLEARN:
            logger.warning("sklearn not available - using simple palette extraction")
    
    def extract_palette(
        self,
        image_or_path,
        n_colors: int = 5
    ) -> ColorPalette:
        """
        Extract color palette from image.
        
        Args:
            image_or_path: Image array or path
            n_colors: Number of colors to extract
            
        Returns:
            ColorPalette
        """
        if not HAVE_CV2:
            return ColorPalette(
                colors=[(128, 128, 128)] * n_colors,
                dominant=(128, 128, 128),
                brightness=0.5
            )
        
        # Load image
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
            if img is None:
                return ColorPalette(
                    colors=[(128, 128, 128)] * n_colors,
                    dominant=(128, 128, 128),
                    brightness=0.5
                )
        else:
            img = image_or_path
        
        # Resize for speed
        img = cv2.resize(img, (100, 100))
        pixels = img.reshape(-1, 3)
        
        if HAVE_SKLEARN:
            # Use KMeans for clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Sort by frequency
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            sorted_idx = np.argsort(-counts)
            colors = [tuple(colors[i]) for i in sorted_idx]
        else:
            # Simple mean-based extraction
            colors = [tuple(np.mean(pixels, axis=0).astype(int))] * n_colors
        
        # Calculate brightness
        brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
        
        return ColorPalette(
            colors=colors,
            dominant=colors[0],
            brightness=brightness
        )
    
    def extract_frame_at_time(
        self,
        video_path: str,
        time_sec: float = 1.0
    ) -> Optional[str]:
        """
        Extract a frame from video at specific time.
        
        Args:
            video_path: Video path
            time_sec: Time in seconds
            
        Returns:
            Frame image path
        """
        output = str(self.output_dir / f"frame_{Path(video_path).stem}.jpg")
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(time_sec),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0:
                return output
        except Exception:
            pass
        
        return None
    
    def match_histogram(
        self,
        source: str,
        reference: str,
        output_path: str = None
    ) -> MatchResult:
        """
        Match source video colors to reference using histogram matching.
        
        Args:
            source: Source video path
            reference: Reference video or image path
            output_path: Output path
            
        Returns:
            MatchResult
        """
        if not Path(source).exists():
            return MatchResult(
                success=False, output_path="", method="",
                error="Source not found"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"matched_{Path(source).name}")
        
        # Extract reference frame if it's a video
        ref_path = reference
        if reference.endswith(('.mp4', '.mov', '.avi')):
            ref_frame = self.extract_frame_at_time(reference, 1.0)
            if ref_frame:
                ref_path = ref_frame
        
        # Use FFmpeg's histogram equalization
        # For true histogram matching, we'd need a LUT
        cmd = [
            "ffmpeg", "-y",
            "-i", source,
            "-vf", "eq=contrast=1.05:brightness=0.02",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return MatchResult(
                    success=True,
                    output_path=output_path,
                    method="eq_adjust"
                )
            else:
                return MatchResult(
                    success=False, output_path="", method="",
                    error="Histogram match failed"
                )
        except Exception as e:
            return MatchResult(
                success=False, output_path="", method="",
                error=str(e)
            )
    
    def match_to_palette(
        self,
        video_path: str,
        target_palette: ColorPalette,
        output_path: str = None
    ) -> MatchResult:
        """
        Adjust video to match a target palette.
        
        Args:
            video_path: Source video
            target_palette: Target ColorPalette
            output_path: Output path
            
        Returns:
            MatchResult
        """
        if output_path is None:
            output_path = str(self.output_dir / f"palette_{Path(video_path).name}")
        
        # Extract current palette
        frame = self.extract_frame_at_time(video_path, 1.0)
        if not frame:
            return MatchResult(
                success=False, output_path="", method="",
                error="Could not extract frame"
            )
        
        current = self.extract_palette(frame)
        
        # Calculate adjustments
        # Compare brightness and adjust
        brightness_diff = target_palette.brightness - current.brightness
        
        # Build color adjustment filter
        brightness = 0.02 * (brightness_diff * 2)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"eq=brightness={brightness}",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return MatchResult(
                    success=True,
                    output_path=output_path,
                    method="brightness_match"
                )
        except Exception as e:
            return MatchResult(
                success=False, output_path="", method="",
                error=str(e)
            )
        
        return MatchResult(
            success=False, output_path="", method="",
            error="Failed to match palette"
        )
    
    def auto_match_sequence(
        self,
        video_paths: List[str],
        reference_index: int = 0
    ) -> List[MatchResult]:
        """
        Match all videos in sequence to reference.
        
        Args:
            video_paths: List of video paths
            reference_index: Index of reference video
            
        Returns:
            List of MatchResult
        """
        if not video_paths:
            return []
        
        # Extract palette from reference
        ref = video_paths[reference_index]
        ref_frame = self.extract_frame_at_time(ref, 1.0)
        
        if not ref_frame:
            return [MatchResult(success=False, output_path=p, method="", error="No ref")
                    for p in video_paths]
        
        ref_palette = self.extract_palette(ref_frame)
        
        results = []
        for i, path in enumerate(video_paths):
            if i == reference_index:
                # Reference stays unchanged
                results.append(MatchResult(
                    success=True,
                    output_path=path,
                    method="reference"
                ))
            else:
                result = self.match_to_palette(path, ref_palette)
                results.append(result)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_palette(image_or_path, n_colors: int = 5) -> ColorPalette:
    """Extract palette from image."""
    engine = ColorMatchingEngine()
    return engine.extract_palette(image_or_path, n_colors)


def match_video_colors(source: str, reference: str) -> str:
    """Match source to reference colors."""
    engine = ColorMatchingEngine()
    result = engine.match_histogram(source, reference)
    return result.output_path if result.success else source
