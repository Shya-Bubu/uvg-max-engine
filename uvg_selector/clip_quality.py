# uvg_selector/clip_quality.py
"""
UVG MAX Clip Quality Filter.

Filters out bad stock clips before selection using:
- Brightness analysis
- Blur detection
- Motion stability
- Black frame detection
"""

import logging
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    logger.warning("OpenCV not available - clip quality filtering disabled")


def check_brightness(frame: np.ndarray) -> float:
    """
    Check frame brightness.
    
    Args:
        frame: BGR image
        
    Returns:
        Brightness score 0-1 (0.4-0.6 is optimal)
    """
    if not HAVE_CV2:
        return 0.5
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)


def check_blur(frame: np.ndarray) -> float:
    """
    Check frame blur using Laplacian variance.
    
    Args:
        frame: BGR image
        
    Returns:
        Blur score 0-1 (higher = sharper)
    """
    if not HAVE_CV2:
        return 0.5
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize: 0-500 range maps to 0-1
    score = min(1.0, laplacian_var / 500.0)
    return float(score)


def check_contrast(frame: np.ndarray) -> float:
    """
    Check frame contrast.
    
    Args:
        frame: BGR image
        
    Returns:
        Contrast score 0-1
    """
    if not HAVE_CV2:
        return 0.5
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    
    # Normalize: 0-80 range maps to 0-1
    score = min(1.0, std / 80.0)
    return float(score)


def detect_black_frame(frame: np.ndarray, threshold: float = 0.05) -> bool:
    """
    Detect if frame is mostly black.
    
    Args:
        frame: BGR image
        threshold: Brightness threshold
        
    Returns:
        True if frame is black
    """
    brightness = check_brightness(frame)
    return brightness < threshold


def check_motion_stability(frames: List[np.ndarray]) -> float:
    """
    Check motion stability across frames.
    
    Args:
        frames: List of BGR images
        
    Returns:
        Stability score 0-1 (higher = more stable)
    """
    if not HAVE_CV2 or len(frames) < 2:
        return 0.5
    
    motion_magnitudes = []
    prev_gray = None
    
    for frame in frames[:10]:  # Sample first 10
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # Small for speed
        
        if prev_gray is not None:
            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitudes.append(float(np.mean(mag)))
        
        prev_gray = gray
    
    if not motion_magnitudes:
        return 0.5
    
    # High variance = unstable (shaky)
    variance = np.var(motion_magnitudes)
    
    # Lower variance = higher stability score
    stability = 1.0 / (1.0 + variance * 2)
    return float(stability)


def assess_clip_quality(
    video_path: str,
    sample_frames: int = 10,
    threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Assess overall clip quality.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample
        threshold: Minimum overall score to pass
        
    Returns:
        Dict with:
            - overall: 0-1 quality score
            - brightness: average brightness
            - blur: average sharpness
            - contrast: average contrast
            - black_frames: count of black frames
            - stability: motion stability
            - pass: True if overall >= threshold
    """
    if not HAVE_CV2:
        return {
            "overall": 0.5,
            "brightness": 0.5,
            "blur": 0.5,
            "contrast": 0.5,
            "black_frames": 0,
            "stability": 0.5,
            "pass": True,
            "error": "opencv_not_available"
        }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "overall": 0.0,
            "pass": False,
            "error": "cannot_open_video"
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return {
            "overall": 0.0,
            "pass": False,
            "error": "empty_video"
        }
    
    step = max(1, total_frames // sample_frames)
    
    frames = []
    brightness_scores = []
    blur_scores = []
    contrast_scores = []
    black_frame_count = 0
    
    for i in range(0, min(total_frames, sample_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frames.append(frame)
        
        # Brightness
        brightness = check_brightness(frame)
        brightness_scores.append(brightness)
        
        # Blur
        blur = check_blur(frame)
        blur_scores.append(blur)
        
        # Contrast
        contrast = check_contrast(frame)
        contrast_scores.append(contrast)
        
        # Black frame
        if detect_black_frame(frame):
            black_frame_count += 1
    
    cap.release()
    
    if not frames:
        return {
            "overall": 0.0,
            "pass": False,
            "error": "no_frames_read"
        }
    
    # Calculate averages
    avg_brightness = float(np.mean(brightness_scores))
    avg_blur = float(np.mean(blur_scores))
    avg_contrast = float(np.mean(contrast_scores))
    
    # Motion stability
    stability = check_motion_stability(frames)
    
    # Brightness penalty (too dark or too bright)
    brightness_penalty = 1.0 - abs(avg_brightness - 0.5) * 2
    brightness_penalty = max(0.0, min(1.0, brightness_penalty))
    
    # Black frame penalty
    black_ratio = black_frame_count / len(frames)
    black_penalty = 1.0 - black_ratio
    
    # Overall score
    overall = (
        brightness_penalty * 0.2 +
        avg_blur * 0.3 +
        avg_contrast * 0.2 +
        stability * 0.2 +
        black_penalty * 0.1
    )
    
    result = {
        "overall": float(overall),
        "brightness": avg_brightness,
        "blur": avg_blur,
        "contrast": avg_contrast,
        "stability": stability,
        "black_frames": black_frame_count,
        "frames_analyzed": len(frames),
        "pass": overall >= threshold
    }
    
    logger.debug(f"Clip quality for {video_path}: {overall:.3f}")
    return result


def filter_quality_clips(
    clip_paths: List[str],
    threshold: float = 0.4,
    sample_frames: int = 10
) -> List[str]:
    """
    Filter clips by quality threshold.
    
    Args:
        clip_paths: List of video paths
        threshold: Minimum quality to keep
        sample_frames: Frames to sample per clip
        
    Returns:
        List of paths that passed quality check
    """
    passed = []
    
    for path in clip_paths:
        try:
            quality = assess_clip_quality(path, sample_frames, threshold)
            
            if quality.get("pass", False):
                passed.append(path)
                logger.debug(f"PASS: {path} (score={quality['overall']:.3f})")
            else:
                logger.debug(f"FAIL: {path} (score={quality.get('overall', 0):.3f})")
        except Exception as e:
            logger.warning(f"Error assessing {path}: {e}")
            # Keep clip if we can't assess it
            passed.append(path)
    
    logger.info(f"Quality filter: {len(passed)}/{len(clip_paths)} clips passed")
    return passed


def get_quality_report(clip_paths: List[str]) -> Dict[str, Any]:
    """
    Generate quality report for multiple clips.
    
    Args:
        clip_paths: List of video paths
        
    Returns:
        Dict with per-clip results and summary
    """
    results = {}
    passed = 0
    failed = 0
    
    for path in clip_paths:
        quality = assess_clip_quality(path)
        results[path] = quality
        
        if quality.get("pass", False):
            passed += 1
        else:
            failed += 1
    
    return {
        "clips": results,
        "passed": passed,
        "failed": failed,
        "total": len(clip_paths),
        "pass_rate": passed / len(clip_paths) if clip_paths else 0
    }
