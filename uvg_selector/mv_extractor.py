# uvg_selector/mv_extractor.py
"""
Motion vector extraction using FFmpeg or OpenCV optical flow.
"""

import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Try OpenCV for optical flow fallback
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def get_motion_stats(video_path: str, sample_rate: float = 1.0) -> dict:
    """
    Extract motion statistics from video.
    
    Args:
        video_path: Path to video file
        sample_rate: Frames per second to sample
        
    Returns:
        Dict with mean_motion, var_motion, motion_hist
    """
    # Try FFmpeg motion vector extraction first
    result = _extract_ffmpeg_motion(video_path)
    
    if result is None and HAVE_CV2:
        # Fallback to optical flow
        result = _extract_optical_flow(video_path, sample_rate)
    
    if result is None:
        # Return default values
        return {
            "mean_motion": 0.0,
            "var_motion": 0.0,
            "motion_hist": [],
            "method": "none"
        }
    
    return result


def _extract_ffmpeg_motion(video_path: str) -> dict | None:
    """
    Extract motion vectors using FFmpeg.
    
    Note: This is a simplified version. Full MV extraction requires
    the mv-extractor library or ffmpeg debug output parsing.
    """
    try:
        # Check if ffprobe is available
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pkt_pts_time", "-of", "csv=p=0",
             video_path],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        # Parse frame timestamps to estimate motion from frame timing
        times = []
        for line in result.stdout.strip().split('\n'):
            try:
                times.append(float(line))
            except ValueError:
                continue
        
        if len(times) < 2:
            return None
        
        # Estimate motion from frame intervals (crude but fast)
        intervals = np.diff(times)
        mean_interval = np.mean(intervals) if len(intervals) > 0 else 0.033
        
        # Lower interval variance might indicate static content
        var_interval = np.var(intervals) if len(intervals) > 0 else 0
        
        # Convert to motion proxy (this is a placeholder)
        # Real implementation would use actual motion vectors
        mean_motion = 1.0 / (mean_interval + 0.001) * 0.01  # Normalize
        
        return {
            "mean_motion": float(mean_motion),
            "var_motion": float(var_interval),
            "motion_hist": [],
            "method": "ffmpeg_timing"
        }
        
    except Exception as e:
        logger.debug(f"FFmpeg motion extraction failed: {e}")
        return None


def _extract_optical_flow(video_path: str, sample_rate: float) -> dict | None:
    """
    Extract motion using OpenCV Farneback optical flow.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(fps / max(0.1, sample_rate)))
        
        prev_gray = None
        motion_magnitudes = []
        frame_idx = 0
        max_samples = 30  # Limit samples for speed
        
        while len(motion_magnitudes) < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))  # Small size for speed
                
                if prev_gray is not None:
                    # Compute optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    
                    # Compute magnitude
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_magnitudes.append(float(np.mean(mag)))
                
                prev_gray = gray
            
            frame_idx += 1
        
        cap.release()
        
        if not motion_magnitudes:
            return None
        
        motion_arr = np.array(motion_magnitudes)
        
        # Create histogram (10 bins)
        hist, _ = np.histogram(motion_arr, bins=10, range=(0, max(motion_arr.max(), 1)))
        hist = hist.astype(float) / (hist.sum() + 1e-12)
        
        return {
            "mean_motion": float(np.mean(motion_arr)),
            "var_motion": float(np.var(motion_arr)),
            "motion_hist": hist.tolist(),
            "method": "optical_flow"
        }
        
    except Exception as e:
        logger.debug(f"Optical flow extraction failed: {e}")
        return None


def compute_motion_match(motion_stats: dict, target_motion: str = "medium") -> float:
    """
    Compute how well motion matches target.
    
    Args:
        motion_stats: Output from get_motion_stats
        target_motion: "low", "medium", "high"
        
    Returns:
        Match score 0-1
    """
    mean_motion = motion_stats.get("mean_motion", 0.5)
    
    # Define target ranges
    targets = {
        "low": (0.0, 0.3),
        "medium": (0.3, 0.7),
        "high": (0.7, 1.0)
    }
    
    low, high = targets.get(target_motion, (0.3, 0.7))
    
    # Score based on distance to target range
    if low <= mean_motion <= high:
        return 1.0
    elif mean_motion < low:
        return max(0, 1.0 - (low - mean_motion))
    else:
        return max(0, 1.0 - (mean_motion - high))
