"""
UVG MAX Visual Density Score Module

Measures visual complexity and interest.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DensityMetrics:
    """Visual density metrics."""
    edge_density: float = 0.0    # Amount of edges/detail
    color_variety: float = 0.0   # Color histogram spread
    motion_amount: float = 0.0   # Optical flow magnitude
    texture_score: float = 0.0   # Texture complexity
    brightness_var: float = 0.0  # Dynamic range
    
    final_score: float = 0.0     # Combined 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_density": self.edge_density,
            "color_variety": self.color_variety,
            "motion_amount": self.motion_amount,
            "texture_score": self.texture_score,
            "brightness_var": self.brightness_var,
            "final_score": self.final_score,
        }


class VisualDensityScorer:
    """
    Visual density scoring for clips.
    
    Higher density = more visually interesting content.
    Used to prefer dynamic, detailed clips over static/bland ones.
    """
    
    # Score weights
    W_EDGES = 0.25
    W_COLOR = 0.20
    W_MOTION = 0.25
    W_TEXTURE = 0.15
    W_BRIGHTNESS = 0.15
    
    def __init__(self):
        """Initialize scorer."""
        pass
    
    def compute_edge_density(self, frame) -> float:
        """Compute edge density using Canny."""
        try:
            import cv2
            import numpy as np
            
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            edges = cv2.Canny(gray, 100, 200)
            density = np.sum(edges > 0) / edges.size
            
            return min(100, density * 500)  # Scale to 0-100
            
        except Exception:
            return 50.0
    
    def compute_color_variety(self, frame) -> float:
        """Compute color variety from histogram."""
        try:
            import cv2
            import numpy as np
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Hue histogram
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist = hist.flatten() / hist.sum()
            
            # Entropy as variety measure
            non_zero = hist[hist > 0]
            entropy = -np.sum(non_zero * np.log2(non_zero))
            
            # Normalize to 0-100 (max entropy ~7.5 for 180 bins)
            return min(100, entropy * 13)
            
        except Exception:
            return 50.0
    
    def compute_motion(self, frame1, frame2) -> float:
        """Compute motion between frames."""
        try:
            import cv2
            import numpy as np
            
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = magnitude.mean()
            
            return min(100, avg_motion * 10)
            
        except Exception:
            return 50.0
    
    def compute_texture(self, frame) -> float:
        """Compute texture score using Laplacian variance."""
        try:
            import cv2
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            return min(100, variance / 10)
            
        except Exception:
            return 50.0
    
    def compute_brightness_variance(self, frame) -> float:
        """Compute brightness dynamic range."""
        try:
            import cv2
            import numpy as np
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = gray.std()
            
            return min(100, variance * 1.5)
            
        except Exception:
            return 50.0
    
    def score_clip(self, clip_path: str) -> DensityMetrics:
        """
        Score a video clip's visual density.
        
        Args:
            clip_path: Path to video clip
            
        Returns:
            DensityMetrics
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(clip_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < 2:
                cap.release()
                return DensityMetrics(final_score=50)
            
            # Sample frames
            sample_indices = [
                total_frames // 4,
                total_frames // 2,
                (3 * total_frames) // 4,
            ]
            
            frames = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                return DensityMetrics(final_score=50)
            
            # Compute metrics
            metrics = DensityMetrics()
            
            # Average over frames
            edge_scores = [self.compute_edge_density(f) for f in frames]
            color_scores = [self.compute_color_variety(f) for f in frames]
            texture_scores = [self.compute_texture(f) for f in frames]
            brightness_scores = [self.compute_brightness_variance(f) for f in frames]
            
            metrics.edge_density = sum(edge_scores) / len(edge_scores)
            metrics.color_variety = sum(color_scores) / len(color_scores)
            metrics.texture_score = sum(texture_scores) / len(texture_scores)
            metrics.brightness_var = sum(brightness_scores) / len(brightness_scores)
            
            # Motion between consecutive samples
            if len(frames) >= 2:
                motion_scores = []
                for i in range(len(frames) - 1):
                    motion_scores.append(self.compute_motion(frames[i], frames[i+1]))
                metrics.motion_amount = sum(motion_scores) / len(motion_scores)
            
            # Final weighted score
            metrics.final_score = (
                self.W_EDGES * metrics.edge_density +
                self.W_COLOR * metrics.color_variety +
                self.W_MOTION * metrics.motion_amount +
                self.W_TEXTURE * metrics.texture_score +
                self.W_BRIGHTNESS * metrics.brightness_var
            )
            
            return metrics
            
        except Exception as e:
            logger.debug(f"Density scoring failed: {e}")
            return DensityMetrics(final_score=50)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score_visual_density(clip_path: str) -> float:
    """Get visual density score for clip."""
    scorer = VisualDensityScorer()
    return scorer.score_clip(clip_path).final_score
