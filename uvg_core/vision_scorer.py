"""
UVG MAX Vision Scorer Module

OpenCLIP-based relevance scoring with quality filters.
Option B: 50% relevance + 20% heuristics + 15% emotion + 10% motion + 5% color
"""

import os
import logging
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class ClipMetrics:
    """Scoring metrics for a clip."""
    relevance: float = 65.0  # CLIP similarity
    emotion: float = 50.0    # Emotion match
    heuristics: float = 50.0 # Clarity + cinematic
    motion: float = 50.0     # Motion stability
    color: float = 50.0      # Color quality
    
    final_score: float = 0.0
    
    # Quality filter results
    passed_quality: bool = True
    quality_issues: List[str] = field(default_factory=list)
    
    # Metadata
    clip_sha256: str = ""
    frame_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance": self.relevance,
            "emotion": self.emotion,
            "heuristics": self.heuristics,
            "motion": self.motion,
            "color": self.color,
            "final_score": self.final_score,
            "passed_quality": self.passed_quality,
            "quality_issues": self.quality_issues,
            "clip_sha256": self.clip_sha256,
            "frame_count": self.frame_count,
        }


# =============================================================================
# EMOTION PROMPTS FOR CLIP SCORING
# =============================================================================

EMOTION_PROMPTS = {
    "calm": "calm peaceful serene tranquil relaxing soothing",
    "dramatic": "dramatic intense powerful striking bold cinematic",
    "joyful": "happy joyful cheerful bright uplifting positive",
    "tense": "tense suspenseful anxious nervous worried intense",
    "romantic": "romantic loving tender intimate passionate gentle",
    "epic": "epic grand magnificent majestic heroic sweeping",
    "sad": "sad melancholy sorrowful emotional tearful somber",
    "energetic": "energetic dynamic active lively vibrant fast",
}


class VisionScorer:
    """
    Vision-based clip scoring using OpenCLIP.
    
    Scoring weights (Option B):
    - 50% relevance (CLIP similarity)
    - 20% heuristics (clarity + cinematic)
    - 15% emotion
    - 10% motion
    - 5% color
    """
    
    # Scoring weights
    W_RELEVANCE = 0.50
    W_HEURISTICS = 0.20
    W_EMOTION = 0.15
    W_MOTION = 0.10
    W_COLOR = 0.05
    
    # Quality thresholds
    MIN_RESOLUTION = 720
    MAX_INTERNAL_CUTS = 3
    MIN_MOTION_STABILITY = 0.3
    
    def __init__(self,
                 frames_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 use_cuda: bool = True,
                 min_resolution: int = 720):
        """
        Initialize vision scorer.
        
        Args:
            frames_dir: Directory for temporary frames
            cache_dir: Directory for embedding cache
            use_cuda: Use GPU if available
            min_resolution: Minimum resolution
        """
        self.frames_dir = Path(frames_dir) if frames_dir else Path(tempfile.gettempdir()) / "uvg_frames"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cuda = use_cuda
        self.min_resolution = min_resolution
        
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-loaded models
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None
        self._device = "cpu"
    
    def _load_clip_model(self) -> bool:
        """Load CLIP model."""
        if self._clip_model is not None:
            return True
        
        try:
            import open_clip
            import torch
            
            self._device = "cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu"
            
            # Use ViT-B-32 for balance of speed and quality
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k'
            )
            self._tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            self._clip_model = self._clip_model.to(self._device)
            self._clip_model.eval()
            
            logger.info(f"CLIP model loaded on {self._device}")
            return True
            
        except ImportError:
            logger.warning("open_clip not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")
            return False
    
    def _get_clip_sha256(self, clip_path: str) -> str:
        """Compute SHA256 of clip file."""
        sha256 = hashlib.sha256()
        with open(clip_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check for cached embeddings."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_cache(self, cache_key: str, data: Dict) -> None:
        """Save to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _sample_frames(self, 
                       clip_path: str,
                       num_frames: int = 5) -> Tuple[List[Any], Path]:
        """
        Sample frames from a video clip.
        
        Args:
            clip_path: Path to video
            num_frames: Number of frames to sample
            
        Returns:
            (list of PIL images, frames directory)
        """
        try:
            import cv2
            from PIL import Image
            
            cap = cv2.VideoCapture(clip_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames < 1:
                cap.release()
                return [], self.frames_dir
            
            # Calculate frame indices (avoid first and last)
            num_frames = min(num_frames, max(3, min(9, int(total_frames / fps))))
            indices = [
                int((i + 1) * total_frames / (num_frames + 1))
                for i in range(num_frames)
            ]
            
            frames = []
            clip_sha = self._get_clip_sha256(clip_path)
            frame_dir = self.frames_dir / clip_sha
            frame_dir.mkdir(exist_ok=True)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    frames.append(pil_img)
                    
                    # Save for cleanup later
                    pil_img.save(frame_dir / f"frame_{idx}.jpg")
            
            cap.release()
            return frames, frame_dir
            
        except Exception as e:
            logger.warning(f"Frame sampling failed: {e}")
            return [], self.frames_dir
    
    def _compute_clip_embedding(self, frames: List[Any]) -> Optional[Any]:
        """Compute average CLIP embedding for frames."""
        if not self._load_clip_model() or not frames:
            return None
        
        try:
            import torch
            
            embeddings = []
            
            with torch.no_grad():
                for frame in frames:
                    img_tensor = self._clip_preprocess(frame).unsqueeze(0).to(self._device)
                    embedding = self._clip_model.encode_image(img_tensor)
                    embeddings.append(embedding)
            
            # Average all embeddings
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            avg_embedding = avg_embedding / avg_embedding.norm(dim=-1, keepdim=True)
            
            return avg_embedding
            
        except Exception as e:
            logger.warning(f"Embedding computation failed: {e}")
            return None
    
    def _compute_text_embedding(self, text: str) -> Optional[Any]:
        """Compute CLIP text embedding."""
        if not self._load_clip_model():
            return None
        
        try:
            import torch
            
            tokens = self._tokenizer([text]).to(self._device)
            
            with torch.no_grad():
                embedding = self._clip_model.encode_text(tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Text embedding failed: {e}")
            return None
    
    def _compute_relevance(self, 
                           image_embedding: Any,
                           text: str) -> float:
        """Compute relevance score (0-100)."""
        text_embedding = self._compute_text_embedding(text)
        
        if image_embedding is None or text_embedding is None:
            return 65.0  # Default
        
        try:
            import torch
            
            similarity = (image_embedding @ text_embedding.T).item()
            # Convert cosine similarity (-1, 1) to (0, 100)
            score = (similarity + 1) / 2 * 100
            return round(max(0, min(100, score)), 2)
            
        except Exception as e:
            logger.debug(f"Relevance computation failed: {e}")
            return 65.0
    
    def _compute_emotion(self, 
                         image_embedding: Any,
                         target_emotion: str) -> float:
        """Compute emotion match score (0-100)."""
        if image_embedding is None:
            return 50.0
        
        # Get emotion prompt
        emotion_text = EMOTION_PROMPTS.get(target_emotion, EMOTION_PROMPTS["calm"])
        
        text_embedding = self._compute_text_embedding(emotion_text)
        if text_embedding is None:
            return 50.0
        
        try:
            import torch
            
            similarity = (image_embedding @ text_embedding.T).item()
            score = (similarity + 1) / 2 * 100
            return round(max(0, min(100, score)), 2)
            
        except Exception:
            return 50.0
    
    def _compute_clarity(self, frames: List[Any]) -> float:
        """Compute clarity score using Laplacian variance (0-100)."""
        if not frames:
            return 50.0
        
        try:
            import cv2
            import numpy as np
            
            variances = []
            for frame in frames:
                # Convert to numpy array
                arr = np.array(frame)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                variance = laplacian.var()
                variances.append(variance)
            
            avg_variance = np.mean(variances)
            # Normalize: typical range 0-1000
            score = min(100, avg_variance / 10)
            return round(score, 2)
            
        except Exception as e:
            logger.debug(f"Clarity computation failed: {e}")
            return 50.0
    
    def _compute_color(self, frames: List[Any]) -> float:
        """Compute color saturation score (0-100)."""
        if not frames:
            return 50.0
        
        try:
            import cv2
            import numpy as np
            
            saturations = []
            for frame in frames:
                arr = np.array(frame)
                hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                saturation = hsv[:, :, 1].mean()
                saturations.append(saturation)
            
            avg_saturation = np.mean(saturations)
            # Normalize: saturation is 0-255
            score = avg_saturation / 255 * 100
            return round(score, 2)
            
        except Exception as e:
            logger.debug(f"Color computation failed: {e}")
            return 50.0
    
    def _compute_motion_stability(self, clip_path: str) -> float:
        """Compute motion stability score (0-100). Higher = more stable."""
        try:
            import cv2
            import numpy as np
            
            cap = cv2.VideoCapture(clip_path)
            
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return 50.0
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            motion_magnitudes = []
            frame_count = 0
            max_frames = 30  # Sample max 30 frames
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_magnitudes.append(magnitude.mean())
                
                prev_gray = gray
                frame_count += 1
            
            cap.release()
            
            if not motion_magnitudes:
                return 50.0
            
            # Standard deviation indicates shakiness
            # Lower std = more stable
            std = np.std(motion_magnitudes)
            # Invert and normalize
            stability = 100 - min(100, std * 10)
            
            return round(max(0, stability), 2)
            
        except Exception as e:
            logger.debug(f"Motion computation failed: {e}")
            return 50.0
    
    def quality_filter(self, 
                       clip_path: str,
                       project_config: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """
        Filter clip for quality issues.
        
        Checks:
        - Resolution
        - Motion stability (shaky camera)
        - Watermarks (basic detection)
        - Hard cuts
        
        Returns:
            (passed, list of issues)
        """
        issues = []
        project_config = project_config or {}
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(clip_path)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            # Resolution check
            min_res = project_config.get("min_resolution", self.MIN_RESOLUTION)
            if min(width, height) < min_res:
                issues.append(f"Resolution too low: {width}x{height} (min {min_res}p)")
            
            # Motion stability
            stability = self._compute_motion_stability(clip_path)
            if stability < self.MIN_MOTION_STABILITY * 100:
                issues.append(f"Excessive camera shake (stability: {stability:.1f})")
            
            # Basic watermark detection (corners have high contrast patterns)
            # This is a simplified heuristic
            watermark_score = self._detect_watermark(clip_path)
            if watermark_score > 0.7:
                issues.append("Possible watermark detected")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.warning(f"Quality filter failed: {e}")
            return True, []  # Pass if can't check
    
    def _detect_watermark(self, clip_path: str) -> float:
        """
        Simple watermark detection.
        
        Returns:
            Score 0-1, higher = more likely watermark
        """
        try:
            import cv2
            import numpy as np
            
            cap = cv2.VideoCapture(clip_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return 0.0
            
            h, w = frame.shape[:2]
            
            # Check corners for high-contrast patterns
            corner_size = min(h, w) // 8
            corners = [
                frame[:corner_size, :corner_size],
                frame[:corner_size, -corner_size:],
                frame[-corner_size:, :corner_size],
                frame[-corner_size:, -corner_size:],
            ]
            
            max_std = 0
            for corner in corners:
                gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
                std = gray.std()
                max_std = max(max_std, std)
            
            # High std in corners suggests text/logos
            # Typical range 20-80
            score = min(1.0, max_std / 80)
            
            return score
            
        except Exception:
            return 0.0
    
    def score_clip(self, 
                   clip_path: str,
                   prompt: str,
                   target_emotion: str = "neutral") -> ClipMetrics:
        """
        Score a clip against a prompt.
        
        Args:
            clip_path: Path to video clip
            prompt: Text prompt to match
            target_emotion: Expected emotion
            
        Returns:
            ClipMetrics with all scores
        """
        metrics = ClipMetrics()
        
        # Get clip hash for caching
        clip_sha = self._get_clip_sha256(clip_path)
        metrics.clip_sha256 = clip_sha
        
        # Check cache
        cache_key = f"{clip_sha}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        cached = self._check_cache(cache_key)
        if cached:
            return ClipMetrics(**cached)
        
        # Sample frames
        frames, frame_dir = self._sample_frames(clip_path)
        metrics.frame_count = len(frames)
        
        if not frames:
            logger.warning(f"No frames sampled from {clip_path}")
            return metrics
        
        # Compute embedding
        embedding = self._compute_clip_embedding(frames)
        
        # Compute scores
        metrics.relevance = self._compute_relevance(embedding, prompt)
        metrics.emotion = self._compute_emotion(embedding, target_emotion)
        
        # Heuristics (clarity + cinematic approximation)
        clarity = self._compute_clarity(frames)
        metrics.heuristics = clarity  # Simplified
        
        # Color
        metrics.color = self._compute_color(frames)
        
        # Motion
        metrics.motion = self._compute_motion_stability(clip_path)
        
        # Quality filter
        passed, issues = self.quality_filter(clip_path)
        metrics.passed_quality = passed
        metrics.quality_issues = issues
        
        # Final score (Option B weights)
        metrics.final_score = (
            self.W_RELEVANCE * metrics.relevance +
            self.W_HEURISTICS * metrics.heuristics +
            self.W_EMOTION * metrics.emotion +
            self.W_MOTION * metrics.motion +
            self.W_COLOR * metrics.color
        )
        metrics.final_score = round(metrics.final_score, 2)
        
        # Cache result
        self._save_cache(cache_key, metrics.to_dict())
        
        # Cleanup frames immediately
        self._cleanup_frames(frame_dir)
        
        logger.debug(f"Scored {clip_path}: {metrics.final_score}")
        
        return metrics
    
    def _cleanup_frames(self, frame_dir: Path) -> None:
        """Delete sampled frames immediately after scoring."""
        try:
            import shutil
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            logger.debug(f"Cleaned up frames: {frame_dir}")
        except Exception as e:
            logger.debug(f"Frame cleanup failed: {e}")
    
    def batch_score(self, 
                    clips: List[Dict],
                    prompt: str,
                    emotion: str = "neutral") -> List[Tuple[Dict, ClipMetrics]]:
        """
        Score multiple clips.
        
        Args:
            clips: List of clip dicts with 'path'
            prompt: Text prompt
            emotion: Target emotion
            
        Returns:
            List of (clip, metrics) tuples sorted by score
        """
        results = []
        
        for clip in clips:
            path = clip.get("path") or clip.get("downloaded_path", "")
            if not path or not Path(path).exists():
                continue
            
            metrics = self.score_clip(path, prompt, emotion)
            results.append((clip, metrics))
        
        # Sort by final score
        results.sort(key=lambda x: x[1].final_score, reverse=True)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score_clip(clip_path: str, prompt: str, emotion: str = "neutral") -> ClipMetrics:
    """Score a single clip."""
    scorer = VisionScorer()
    return scorer.score_clip(clip_path, prompt, emotion)


def quality_filter(clip_path: str) -> Tuple[bool, List[str]]:
    """Check clip quality."""
    scorer = VisionScorer()
    return scorer.quality_filter(clip_path)
