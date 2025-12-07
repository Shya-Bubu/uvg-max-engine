"""
UVG MAX Clip Trimmer Module

Sliding-window extraction of best segments from clips.
Extracts the most relevant 3-6 second segment using frame embeddings.
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrimResult:
    """Result of clip trimming operation."""
    success: bool
    original_path: str
    trimmed_path: str
    start_time: float
    end_time: float
    duration: float
    relevance_score: float
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "original_path": self.original_path,
            "trimmed_path": self.trimmed_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "relevance_score": self.relevance_score,
            "error": self.error,
        }


class ClipTrimmer:
    """
    Sliding-window clip trimmer.
    
    Extracts the most relevant subsegment from longer clips
    using frame-level relevance scoring.
    """
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 min_segment: float = 1.8,
                 max_segment: float = 6.0,
                 window_step: float = 0.5,
                 tolerance: float = 0.15):
        """
        Initialize clip trimmer.
        
        Args:
            output_dir: Directory for trimmed clips
            min_segment: Minimum segment duration
            max_segment: Maximum segment duration
            window_step: Sliding window step size
            tolerance: Duration tolerance (±seconds)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/trimmed")
        self.min_segment = min_segment
        self.max_segment = max_segment
        self.window_step = window_step
        self.tolerance = tolerance
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vision scorer for frame analysis
        self._scorer = None
    
    def _get_scorer(self):
        """Lazy load vision scorer."""
        if self._scorer is None:
            from .vision_scorer import VisionScorer
            self._scorer = VisionScorer()
        return self._scorer
    
    def _get_clip_duration(self, clip_path: str) -> float:
        """Get clip duration using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                clip_path
            ], capture_output=True, text=True, timeout=30)
            
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get duration: {e}")
            return 0.0
    
    def _sample_window_frames(self, 
                               clip_path: str,
                               start: float,
                               end: float,
                               num_frames: int = 3) -> List[Any]:
        """Sample frames from a specific time window."""
        try:
            import cv2
            from PIL import Image
            
            cap = cv2.VideoCapture(clip_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frames = []
            duration = end - start
            
            for i in range(num_frames):
                # Calculate time position
                t = start + (i + 0.5) * duration / num_frames
                frame_num = int(t * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.debug(f"Frame sampling failed: {e}")
            return []
    
    def _score_window(self, 
                      clip_path: str,
                      start: float,
                      end: float,
                      prompt: str) -> float:
        """Score a time window for relevance."""
        frames = self._sample_window_frames(clip_path, start, end)
        
        if not frames:
            return 0.0
        
        try:
            scorer = self._get_scorer()
            
            # Compute embedding for window frames
            embedding = scorer._compute_clip_embedding(frames)
            if embedding is None:
                return 50.0
            
            # Compute relevance
            relevance = scorer._compute_relevance(embedding, prompt)
            return relevance
            
        except Exception as e:
            logger.debug(f"Window scoring failed: {e}")
            return 50.0
    
    def find_best_segment(self, 
                          clip_path: str,
                          target_duration: float,
                          prompt: str) -> Tuple[float, float, float]:
        """
        Find the best segment using sliding window.
        
        Args:
            clip_path: Path to clip
            target_duration: Desired segment duration
            prompt: Text prompt for relevance
            
        Returns:
            (start_time, end_time, relevance_score)
        """
        clip_duration = self._get_clip_duration(clip_path)
        
        if clip_duration <= 0:
            return 0.0, target_duration, 50.0
        
        # Clamp target duration
        window_size = max(
            self.min_segment,
            min(self.max_segment, target_duration, clip_duration)
        )
        
        # If clip is shorter than window, use whole clip
        if clip_duration <= window_size:
            score = self._score_window(clip_path, 0, clip_duration, prompt)
            return 0.0, clip_duration, score
        
        # Sliding window
        stride = max(self.window_step, window_size / 10)
        
        best_start = 0.0
        best_score = 0.0
        
        current = 0.0
        while current + window_size <= clip_duration:
            score = self._score_window(clip_path, current, current + window_size, prompt)
            
            if score > best_score:
                best_score = score
                best_start = current
            
            current += stride
        
        return best_start, best_start + window_size, best_score
    
    def trim_clip(self,
                  clip_path: str,
                  target_duration: float,
                  prompt: str,
                  output_name: Optional[str] = None) -> TrimResult:
        """
        Trim clip to best segment.
        
        Args:
            clip_path: Path to source clip
            target_duration: Desired duration
            prompt: Text for relevance scoring
            output_name: Optional output filename
            
        Returns:
            TrimResult with trimmed clip info
        """
        clip_path = Path(clip_path)
        
        if not clip_path.exists():
            return TrimResult(
                success=False,
                original_path=str(clip_path),
                trimmed_path="",
                start_time=0,
                end_time=0,
                duration=0,
                relevance_score=0,
                error="Source clip not found"
            )
        
        # Find best segment
        start, end, score = self.find_best_segment(str(clip_path), target_duration, prompt)
        duration = end - start
        
        # Generate output path
        if output_name:
            output_path = self.output_dir / output_name
        else:
            output_path = self.output_dir / f"trimmed_{clip_path.stem}.mp4"
        
        # FFmpeg trim command
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(clip_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg trim failed: {result.stderr[:200]}")
                return TrimResult(
                    success=False,
                    original_path=str(clip_path),
                    trimmed_path="",
                    start_time=start,
                    end_time=end,
                    duration=duration,
                    relevance_score=score,
                    error=result.stderr[:200]
                )
            
            logger.info(f"Trimmed {clip_path.name}: {start:.1f}s - {end:.1f}s (score: {score:.1f})")
            
            return TrimResult(
                success=True,
                original_path=str(clip_path),
                trimmed_path=str(output_path),
                start_time=start,
                end_time=end,
                duration=duration,
                relevance_score=score
            )
            
        except subprocess.TimeoutExpired:
            return TrimResult(
                success=False,
                original_path=str(clip_path),
                trimmed_path="",
                start_time=start,
                end_time=end,
                duration=duration,
                relevance_score=score,
                error="FFmpeg timeout"
            )
        except Exception as e:
            return TrimResult(
                success=False,
                original_path=str(clip_path),
                trimmed_path="",
                start_time=start,
                end_time=end,
                duration=duration,
                relevance_score=score,
                error=str(e)
            )
    
    def extend_short_clip(self,
                          clip_path: str,
                          target_duration: float,
                          output_path: Optional[str] = None) -> str:
        """
        Extend a short clip using freeze-frame at end.
        
        Args:
            clip_path: Path to short clip
            target_duration: Desired duration
            output_path: Optional output path
            
        Returns:
            Path to extended clip
        """
        clip_duration = self._get_clip_duration(clip_path)
        
        if clip_duration >= target_duration:
            return clip_path
        
        if output_path is None:
            output_path = self.output_dir / f"extended_{Path(clip_path).stem}.mp4"
        
        extend_by = target_duration - clip_duration
        
        try:
            # Use tpad filter to add freeze frames
            cmd = [
                "ffmpeg", "-y",
                "-i", clip_path,
                "-vf", f"tpad=stop_mode=clone:stop_duration={extend_by}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"Clip extension failed: {e}")
            return clip_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def trim_to_best_segment(clip_path: str,
                         duration: float,
                         prompt: str) -> TrimResult:
    """Trim a clip to its best segment."""
    trimmer = ClipTrimmer()
    return trimmer.trim_clip(clip_path, duration, prompt)


def find_best_segment(clip_path: str,
                      duration: float,
                      prompt: str) -> Tuple[float, float, float]:
    """Find the best segment without trimming."""
    trimmer = ClipTrimmer()
    return trimmer.find_best_segment(clip_path, duration, prompt)
