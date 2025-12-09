# uvg_selector/decord_loader.py
"""
Fast frame sampling using Decord (with OpenCV fallback).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Decord
try:
    import decord
    from decord import VideoReader, cpu
    HAVE_DECORD = True
    logger.debug("Decord available for fast video loading")
except ImportError:
    HAVE_DECORD = False
    logger.debug("Decord not available, using OpenCV fallback")

# OpenCV is required as fallback
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    logger.warning("OpenCV not available - frame loading will fail")


def sample_frames(video_path: str, fps: float = 1.0, max_frames: int = 30) -> list:
    """
    Sample frames from video at specified rate.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to sample (default 1.0)
        max_frames: Maximum frames to return
        
    Returns:
        List of BGR numpy arrays (OpenCV format)
    """
    if HAVE_DECORD:
        return _sample_decord(video_path, fps, max_frames)
    elif HAVE_CV2:
        return _sample_opencv(video_path, fps, max_frames)
    else:
        logger.error("No video loading library available")
        return []


def _sample_decord(video_path: str, fps: float, max_frames: int) -> list:
    """Sample frames using Decord (faster)."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        native_fps = vr.get_avg_fps()
        
        if total_frames == 0:
            return []
        
        # Calculate frame indices to sample
        frame_step = max(1, int(round(native_fps / max(0.1, fps))))
        indices = list(range(0, total_frames, frame_step))
        
        # Limit to max_frames
        if len(indices) > max_frames:
            step = len(indices) // max_frames
            indices = indices[::step][:max_frames]
        
        # Extract frames and convert to BGR
        frames = []
        for i in indices[:max_frames]:
            frame = vr[i].asnumpy()
            # Decord returns RGB, convert to BGR for CLIP preprocessing
            frame_bgr = frame[:, :, ::-1].copy()
            frames.append(frame_bgr)
        
        logger.debug(f"Decord sampled {len(frames)} frames from {video_path}")
        return frames
        
    except Exception as e:
        logger.warning(f"Decord failed for {video_path}: {e}, falling back to OpenCV")
        return _sample_opencv(video_path, fps, max_frames)


def _sample_opencv(video_path: str, fps: float, max_frames: int) -> list:
    """Sample frames using OpenCV (fallback)."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame step
        frame_step = max(1, int(round(native_fps / max(0.1, fps))))
        
        frames = []
        frame_idx = 0
        saved = 0
        
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_step == 0:
                frames.append(frame)
                saved += 1
            
            frame_idx += 1
        
        cap.release()
        logger.debug(f"OpenCV sampled {len(frames)} frames from {video_path}")
        return frames
        
    except Exception as e:
        logger.error(f"OpenCV frame sampling failed: {e}")
        return []


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Returns:
        Dict with duration, fps, width, height, frame_count
    """
    try:
        if HAVE_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            frame_count = len(vr)
            # Get first frame for dimensions
            if frame_count > 0:
                frame = vr[0].asnumpy()
                height, width = frame.shape[:2]
            else:
                width, height = 0, 0
            duration = frame_count / fps if fps > 0 else 0
        else:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        
        return {
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count
        }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {"duration": 0, "fps": 0, "width": 0, "height": 0, "frame_count": 0}
