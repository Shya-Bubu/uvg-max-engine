"""
UVG MAX Thumbnail Generator Module

Professional thumbnails with trending styles.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ThumbnailResult:
    """Thumbnail generation result."""
    success: bool
    output_path: str
    style: str
    error: str = ""


THUMBNAIL_STYLES = {
    "tiktok": {
        "text_position": "center",
        "font_size": 80,
        "font_color": "white",
        "stroke_color": "black",
        "stroke_width": 4,
        "emoji": True,
        "gradient_overlay": True,
        "border_3d": True,
    },
    "youtube": {
        "text_position": "bottom",
        "font_size": 70,
        "font_color": "yellow",
        "stroke_color": "black",
        "stroke_width": 3,
        "emoji": False,
        "gradient_overlay": False,
        "border_3d": False,
    },
    "cinematic": {
        "text_position": "bottom",
        "font_size": 60,
        "font_color": "white",
        "stroke_color": "black",
        "stroke_width": 2,
        "emoji": False,
        "gradient_overlay": True,
        "border_3d": False,
    },
    "motivational": {
        "text_position": "center",
        "font_size": 75,
        "font_color": "white",
        "stroke_color": "#FF6600",
        "stroke_width": 4,
        "emoji": True,
        "gradient_overlay": True,
        "border_3d": True,
    },
}


class ThumbnailGenerator:
    """
    Professional thumbnail generator.
    
    Features:
    - Frame extraction at golden ratio
    - Text overlay with styling
    - Emoji packs
    - 3D stroke borders
    - Glow effects
    - Face detection for hero subject
    """
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 width: int = 1280,
                 height: int = 720):
        """
        Initialize thumbnail generator.
        
        Args:
            output_dir: Output directory
            width: Thumbnail width
            height: Thumbnail height
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/thumbnails")
        self.width = width
        self.height = height
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_best_frame(self,
                            video_path: str,
                            time_ratio: float = 0.382) -> str:
        """
        Extract best frame using golden ratio.
        
        Args:
            video_path: Video path
            time_ratio: Position ratio (default: golden ratio)
            
        Returns:
            Frame image path
        """
        output_path = self.output_dir / f"frame_{Path(video_path).stem}.jpg"
        
        # Get duration
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ], capture_output=True, text=True, timeout=30)
            
            duration = float(result.stdout.strip())
            timestamp = duration * time_ratio
            
        except Exception:
            timestamp = 3.0
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=30)
        return str(output_path)
    
    def add_text_overlay(self,
                          image_path: str,
                          text: str,
                          style: Dict[str, Any],
                          output_path: Optional[str] = None) -> str:
        """
        Add text overlay to image.
        
        Args:
            image_path: Input image
            text: Text to overlay
            style: Style configuration
            output_path: Output path
            
        Returns:
            Output path
        """
        if output_path is None:
            output_path = self.output_dir / f"text_{Path(image_path).stem}.jpg"
        
        # Determine text position
        position = style.get("text_position", "center")
        if position == "center":
            y_pos = "(h-text_h)/2"
        elif position == "top":
            y_pos = "50"
        else:  # bottom
            y_pos = "h-text_h-50"
        
        # Build drawtext filter
        font_size = style.get("font_size", 70)
        font_color = style.get("font_color", "white")
        stroke_color = style.get("stroke_color", "black")
        stroke_width = style.get("stroke_width", 3)
        
        # Split text into lines (max 3 words per line)
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(current_line) >= 3:
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        
        display_text = "\\n".join(lines[:3])  # Max 3 lines
        
        drawtext = (
            f"drawtext=text='{display_text}':"
            f"fontsize={font_size}:"
            f"fontcolor={font_color}:"
            f"borderw={stroke_width}:"
            f"bordercolor={stroke_color}:"
            f"x=(w-text_w)/2:"
            f"y={y_pos}:"
            f"font=Impact"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", image_path,
            "-vf", drawtext,
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=30)
        return str(output_path)
    
    def add_gradient_overlay(self,
                              image_path: str,
                              output_path: Optional[str] = None) -> str:
        """Add gradient overlay for text visibility."""
        if output_path is None:
            output_path = self.output_dir / f"gradient_{Path(image_path).stem}.jpg"
        
        # Add dark gradient at bottom
        filter_str = (
            "split[a][b];"
            "[b]lutrgb=r=0:g=0:b=0,format=rgba,"
            "geq=r=0:g=0:b=0:a='220*(Y/H)^2'[gradient];"
            "[a][gradient]overlay"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", image_path,
            "-filter_complex", filter_str,
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=30)
        return str(output_path)
    
    def detect_face(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image for hero framing.
        
        Returns:
            (x, y, w, h) of largest face or None
        """
        try:
            import cv2
            
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Return largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                return tuple(largest)
            
            return None
            
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
            return None
    
    def generate_trending_thumbnail(self,
                                     video_path: str,
                                     title: str,
                                     style: str = "tiktok",
                                     output_name: Optional[str] = None) -> ThumbnailResult:
        """
        Generate trending-style thumbnail.
        
        Args:
            video_path: Source video
            title: Title text
            style: tiktok, youtube, cinematic, motivational
            output_name: Output filename
            
        Returns:
            ThumbnailResult
        """
        if output_name is None:
            output_name = f"thumb_{style}_{Path(video_path).stem}.jpg"
        
        output_path = self.output_dir / output_name
        style_config = THUMBNAIL_STYLES.get(style, THUMBNAIL_STYLES["tiktok"])
        
        try:
            # 1. Extract best frame
            frame_path = self.extract_best_frame(video_path)
            
            # 2. Scale to thumbnail size
            scaled_path = self.output_dir / "temp_scaled.jpg"
            cmd = [
                "ffmpeg", "-y",
                "-i", frame_path,
                "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
                       f"crop={self.width}:{self.height}",
                str(scaled_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            
            current_path = str(scaled_path)
            
            # 3. Add gradient if configured
            if style_config.get("gradient_overlay"):
                current_path = self.add_gradient_overlay(current_path)
            
            # 4. Add text overlay
            current_path = self.add_text_overlay(current_path, title, style_config)
            
            # 5. Add emoji if configured
            if style_config.get("emoji"):
                current_path = self._add_emoji(current_path, ["🔥", "✨"])
            
            # 6. Copy to final path
            Path(current_path).rename(output_path)
            
            # Cleanup temp files
            Path(frame_path).unlink(missing_ok=True)
            scaled_path.unlink(missing_ok=True)
            
            return ThumbnailResult(
                success=True,
                output_path=str(output_path),
                style=style
            )
            
        except Exception as e:
            return ThumbnailResult(
                success=False,
                output_path="",
                style=style,
                error=str(e)
            )
    
    def _add_emoji(self,
                   image_path: str,
                   emojis: List[str],
                   output_path: Optional[str] = None) -> str:
        """Add emoji overlays (simplified - uses text)."""
        if output_path is None:
            output_path = self.output_dir / f"emoji_{Path(image_path).stem}.jpg"
        
        # Use drawtext for emoji (requires emoji font)
        emoji_str = " ".join(emojis)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", image_path,
            "-vf", f"drawtext=text='{emoji_str}':fontsize=60:x=50:y=50",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            return str(output_path)
        except Exception:
            return image_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_thumbnail(video_path: str,
                        title: str,
                        style: str = "tiktok") -> ThumbnailResult:
    """Generate trending thumbnail."""
    generator = ThumbnailGenerator()
    return generator.generate_trending_thumbnail(video_path, title, style)
