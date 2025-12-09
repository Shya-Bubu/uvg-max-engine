# uvg_selector/trim_segment.py
"""
Smart video segment trimming using sliding window algorithm.
"""

import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .decord_loader import sample_frames, get_video_info
from .onnx_clip import ONNXCLIP
from .aesthetic import score_aesthetic

logger = logging.getLogger(__name__)


def find_best_trim(
    video_path: str,
    prompt: str,
    min_dur: float = 3.0,
    max_dur: float = 6.0,
    step: float = 0.5,
    sample_fps: float = 2.0
) -> Dict[str, Any]:
    """
    Find optimal trim segment using sliding window.
    
    Args:
        video_path: Path to video file
        prompt: Text prompt for semantic matching
        min_dur: Minimum segment duration (seconds)
        max_dur: Maximum segment duration (seconds)
        step: Duration step for window scan
        sample_fps: Frames per second for sampling
        
    Returns:
        Dict with start_sec, duration, score
    """
    logger.info(f"Finding best trim for: {video_path}")
    
    # Get video info
    info = get_video_info(video_path)
    video_duration = info.get("duration", 0)
    
    if video_duration <= 0:
        return {"start": 0.0, "duration": min_dur, "score": 0.0, "error": "invalid_video"}
    
    # If video is already short enough, return full video
    if video_duration <= max_dur:
        return {"start": 0.0, "duration": video_duration, "score": 1.0}
    
    # Sample frames at specified rate
    frames = sample_frames(video_path, fps=sample_fps, max_frames=300)
    n_frames = len(frames)
    
    if n_frames == 0:
        return {"start": 0.0, "duration": min_dur, "score": 0.0, "error": "no_frames"}
    
    # Compute per-frame signals
    try:
        clip = ONNXCLIP()
        text_emb = clip.embed_text(prompt)
        frame_embs = clip.embed_batch(frames)
        
        # Semantic similarity per frame
        sem_scores = np.array([
            np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-12)
            for emb in frame_embs
        ])
        
        # Aesthetic scores per frame
        aes_scores = score_aesthetic(frame_embs)
        
    except Exception as e:
        logger.warning(f"Signal computation failed: {e}")
        sem_scores = np.zeros(n_frames)
        aes_scores = np.ones(n_frames) * 5.0
    
    # Compute stability (inverse of variance in local windows)
    stability = _compute_stability(sem_scores, window=5)
    
    # Sliding window scan
    results = []
    
    for duration in np.arange(min_dur, max_dur + 0.001, step):
        # Window size in frames
        win_frames = int(round(duration * sample_fps))
        if win_frames > n_frames:
            continue
        
        for start_idx in range(0, n_frames - win_frames + 1):
            end_idx = start_idx + win_frames
            
            # Compute window scores
            win_sem = sem_scores[start_idx:end_idx]
            win_aes = aes_scores[start_idx:end_idx]
            win_stab = stability[start_idx:end_idx]
            
            # Aggregate scores
            seg_sem = float(np.mean(win_sem))
            seg_aes = float(np.mean(win_aes)) / 10.0  # Normalize to 0-1
            seg_stab = float(np.mean(win_stab))
            
            # Penalize high variance (prefer consistent segments)
            seg_var = float(np.var(win_sem))
            
            # Combined score
            score = seg_sem * 0.5 + seg_aes * 0.3 + seg_stab * 0.2 - seg_var * 0.3
            
            results.append({
                "score": score,
                "start_idx": start_idx,
                "duration": duration,
                "seg_sem": seg_sem,
                "seg_aes": seg_aes * 10
            })
    
    if not results:
        return {"start": 0.0, "duration": min_dur, "score": 0.0}
    
    # Find best result
    best = max(results, key=lambda x: x["score"])
    
    # Convert frame index to seconds
    start_sec = best["start_idx"] / sample_fps
    
    logger.info(f"Best trim: start={start_sec:.2f}s, dur={best['duration']:.1f}s, score={best['score']:.3f}")
    
    return {
        "start": float(start_sec),
        "duration": float(best["duration"]),
        "score": float(best["score"]),
        "signals": {
            "semantic": best["seg_sem"],
            "aesthetic": best["seg_aes"]
        }
    }


def _compute_stability(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute local stability (inverse variance).
    """
    n = len(scores)
    stability = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        local_var = np.var(scores[start:end])
        stability[i] = 1.0 / (1.0 + local_var * 10)
    
    return stability


def detect_scene_boundaries(video_path: str, threshold: float = 30.0) -> list:
    """
    Detect scene boundaries using histogram differences.
    
    Args:
        video_path: Path to video
        threshold: Histogram difference threshold
        
    Returns:
        List of (start_sec, end_sec) tuples for each scene
    """
    try:
        import cv2
    except ImportError:
        # Fallback: return whole video as one scene
        info = get_video_info(video_path)
        return [(0.0, info.get("duration", 10.0))]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        info = get_video_info(video_path)
        return [(0.0, info.get("duration", 10.0))]
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    prev_hist = None
    boundaries = [0.0]
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Compute histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-12)
        
        if prev_hist is not None:
            # Compare histograms
            diff = cv2.compareHist(
                prev_hist.astype(np.float32),
                hist.astype(np.float32),
                cv2.HISTCMP_CHISQR
            )
            
            if diff > threshold:
                boundaries.append(frame_idx / fps)
        
        prev_hist = hist
        frame_idx += 1
    
    # Add end time
    boundaries.append(frame_idx / fps)
    cap.release()
    
    # Convert to scene tuples
    scenes = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    
    return scenes


def extract_ffmpeg_segment(
    input_path: str,
    start: float,
    duration: float,
    output_path: str,
    crf: int = 23,
    preset: str = "fast"
) -> str:
    """
    Extract segment using FFmpeg.
    
    Args:
        input_path: Input video path
        start: Start time in seconds
        duration: Duration in seconds
        output_path: Output video path
        crf: Quality (lower = better, 18-28 reasonable)
        preset: Encoding speed preset
        
    Returns:
        Output path if successful
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info(f"Extracted segment: {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg failed: {result.stderr[:500]}")
            return ""
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out")
        return ""
    except Exception as e:
        logger.error(f"Segment extraction failed: {e}")
        return ""


def auto_trim_clip(
    video_path: str,
    prompt: str,
    output_dir: str = "uvg_output/clips",
    min_dur: float = 3.0,
    max_dur: float = 6.0
) -> Dict[str, Any]:
    """
    Automatically find and extract best segment from video.
    
    Args:
        video_path: Input video path
        prompt: Text prompt for matching
        output_dir: Output directory
        min_dur: Minimum duration
        max_dur: Maximum duration
        
    Returns:
        Dict with trimmed_path, start, duration, score
    """
    # Find best segment
    trim_info = find_best_trim(video_path, prompt, min_dur, max_dur)
    
    if "error" in trim_info:
        return trim_info
    
    # Create output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(video_path).stem
    output_name = f"{input_name}_trim_{trim_info['start']:.1f}s.mp4"
    output_path = str(output_dir / output_name)
    
    # Extract segment
    result = extract_ffmpeg_segment(
        video_path,
        trim_info["start"],
        trim_info["duration"],
        output_path
    )
    
    if result:
        trim_info["trimmed_path"] = result
    else:
        trim_info["error"] = "extraction_failed"
    
    return trim_info
