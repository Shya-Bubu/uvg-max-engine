# uvg_selector/scene_detector.py
"""
Scene Boundary Detection for UVG MAX.

Detects scene cuts in videos for clean trimming.
Uses PySceneDetect when available, falls back to histogram analysis.
"""

import logging
import numpy as np
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PySceneDetect
try:
    from scenedetect import detect, ContentDetector, ThresholdDetector
    HAVE_SCENEDETECT = True
except ImportError:
    HAVE_SCENEDETECT = False
    logger.debug("PySceneDetect not available - using histogram fallback")

# Try to import OpenCV
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def detect_scenes_pyscene(
    video_path: str,
    threshold: float = 27.0
) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries using PySceneDetect.
    
    Args:
        video_path: Path to video file
        threshold: Detection threshold (lower = more sensitive)
        
    Returns:
        List of (start_sec, end_sec) tuples for each scene
    """
    if not HAVE_SCENEDETECT:
        return []
    
    try:
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        
        scenes = []
        for scene in scene_list:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes.append((start, end))
        
        logger.debug(f"PySceneDetect found {len(scenes)} scenes")
        return scenes
        
    except Exception as e:
        logger.warning(f"PySceneDetect failed: {e}")
        return []


def detect_scenes_histogram(
    video_path: str,
    threshold: float = 30.0,
    sample_fps: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries using histogram comparison.
    
    Fallback method when PySceneDetect is not available.
    
    Args:
        video_path: Path to video file
        threshold: Chi-square difference threshold
        sample_fps: Frames per second to sample
        
    Returns:
        List of (start_sec, end_sec) tuples for each scene
    """
    if not HAVE_CV2:
        logger.warning("OpenCV not available for scene detection")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps / sample_fps))
    
    scene_cuts = [0.0]  # Always start at 0
    prev_hist = None
    
    for frame_idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        if prev_hist is not None:
            # Chi-square comparison
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            
            if diff > threshold:
                time_sec = frame_idx / fps
                # Avoid cuts too close together (< 1 second)
                if time_sec - scene_cuts[-1] > 1.0:
                    scene_cuts.append(time_sec)
                    logger.debug(f"Scene cut at {time_sec:.2f}s (diff={diff:.1f})")
        
        prev_hist = hist.copy()
    
    cap.release()
    
    # Add video end
    duration = total_frames / fps
    scene_cuts.append(duration)
    
    # Convert cuts to scene ranges
    scenes = []
    for i in range(len(scene_cuts) - 1):
        scenes.append((scene_cuts[i], scene_cuts[i + 1]))
    
    logger.debug(f"Histogram detection found {len(scenes)} scenes")
    return scenes


def detect_scene_boundaries(
    video_path: str,
    threshold: float = 30.0,
    use_pyscene: bool = True
) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries in a video.
    
    Uses PySceneDetect if available, falls back to histogram method.
    
    Args:
        video_path: Path to video file
        threshold: Detection threshold
        use_pyscene: Try PySceneDetect first
        
    Returns:
        List of (start_sec, end_sec) tuples for each scene
    """
    scenes = []
    
    # Try PySceneDetect first
    if use_pyscene and HAVE_SCENEDETECT:
        scenes = detect_scenes_pyscene(video_path, threshold)
    
    # Fallback to histogram
    if not scenes and HAVE_CV2:
        scenes = detect_scenes_histogram(video_path, threshold)
    
    # If nothing detected, return whole video as one scene
    if not scenes:
        try:
            if HAVE_CV2:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps
                cap.release()
                scenes = [(0.0, duration)]
            else:
                scenes = [(0.0, 60.0)]  # Default 60s
        except:
            scenes = [(0.0, 60.0)]
    
    return scenes


def get_scene_at_time(
    scenes: List[Tuple[float, float]],
    time_sec: float
) -> int:
    """
    Get scene index for a given time.
    
    Args:
        scenes: List of (start, end) tuples
        time_sec: Time in seconds
        
    Returns:
        Scene index (0-based), or -1 if not found
    """
    for i, (start, end) in enumerate(scenes):
        if start <= time_sec < end:
            return i
    return -1


def check_crosses_boundary(
    scenes: List[Tuple[float, float]],
    start_sec: float,
    end_sec: float
) -> bool:
    """
    Check if a time range crosses a scene boundary.
    
    Args:
        scenes: List of (start, end) tuples
        start_sec: Segment start
        end_sec: Segment end
        
    Returns:
        True if segment crosses a scene boundary
    """
    start_scene = get_scene_at_time(scenes, start_sec)
    end_scene = get_scene_at_time(scenes, end_sec - 0.01)  # Subtract small epsilon
    
    return start_scene != end_scene and start_scene >= 0 and end_scene >= 0


def find_safe_trim_within_scene(
    scenes: List[Tuple[float, float]],
    ideal_start: float,
    ideal_duration: float,
    min_duration: float = 2.0
) -> Tuple[float, float]:
    """
    Find a safe trim range that doesn't cross scene boundaries.
    
    Args:
        scenes: List of (start, end) tuples
        ideal_start: Desired start time
        ideal_duration: Desired duration
        min_duration: Minimum acceptable duration
        
    Returns:
        (adjusted_start, adjusted_duration) tuple
    """
    scene_idx = get_scene_at_time(scenes, ideal_start)
    
    if scene_idx < 0:
        return (ideal_start, ideal_duration)
    
    scene_start, scene_end = scenes[scene_idx]
    
    # Clamp start to scene
    adjusted_start = max(ideal_start, scene_start)
    
    # Clamp end to scene
    ideal_end = adjusted_start + ideal_duration
    adjusted_end = min(ideal_end, scene_end)
    adjusted_duration = adjusted_end - adjusted_start
    
    # If too short, try to extend within scene
    if adjusted_duration < min_duration:
        # Try to start earlier
        available_before = ideal_start - scene_start
        available_after = scene_end - ideal_end
        
        if available_before > 0:
            adjusted_start = max(scene_start, adjusted_start - (min_duration - adjusted_duration))
            adjusted_duration = adjusted_end - adjusted_start
        elif available_after > 0:
            adjusted_end = min(scene_end, adjusted_end + (min_duration - adjusted_duration))
            adjusted_duration = adjusted_end - adjusted_start
    
    return (adjusted_start, max(min_duration, adjusted_duration))


# Convenience function for backward compatibility
def detect_scene_cuts(video_path: str, threshold: float = 30.0) -> List[float]:
    """
    Detect scene cut times.
    
    Args:
        video_path: Path to video
        threshold: Detection threshold
        
    Returns:
        List of cut times in seconds
    """
    scenes = detect_scene_boundaries(video_path, threshold)
    
    cuts = [0.0]
    for start, end in scenes:
        if end not in cuts:
            cuts.append(end)
    
    return sorted(set(cuts))
