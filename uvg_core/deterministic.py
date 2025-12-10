# uvg_core/deterministic.py
"""
Deterministic Mode Controller for UVG MAX.

When deterministic_mode=True, this module ensures reproducible outputs by:
- Using fixed frame sampling instead of random
- Sorting candidates lexicographically
- Seeding all random operations
- Disabling random VFX (film grain)
- Using fixed STFT parameters for beat detection

Usage:
    from uvg_core.deterministic import get_deterministic_context
    
    ctx = get_deterministic_context(enabled=True, seed=42)
    
    # Use in CLIP scoring
    frames = ctx.get_frame_indices(video_duration, count=10)
    
    # Use in clip selection
    candidates = ctx.sort_candidates(clip_paths)
    
    # Check if grain should be disabled
    if ctx.should_disable_grain():
        skip_grain_filter()
"""

import random
import logging
from dataclasses import dataclass
from typing import List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DeterministicContext:
    """
    Context for deterministic operations.
    
    When enabled=True, all random operations are made reproducible.
    """
    enabled: bool = False
    seed: int = 42
    
    def __post_init__(self):
        if self.enabled:
            logger.info(f"Deterministic mode ENABLED with seed={self.seed}")
            # Set global random seed
            random.seed(self.seed)
            
            # Set numpy seed if available
            try:
                import numpy as np
                np.random.seed(self.seed)
            except ImportError:
                pass
            
            # Set torch seed if available
            try:
                import torch
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                    # Ensure deterministic operations
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass
    
    def get_frame_indices(self, video_duration: float, count: int, fps: float = 30.0) -> List[int]:
        """
        Get frame indices for sampling.
        
        In deterministic mode: Returns evenly spaced frames (0, N, 2N, ...)
        In random mode: Returns random frame indices
        
        Args:
            video_duration: Video duration in seconds
            count: Number of frames to sample
            fps: Frames per second
            
        Returns:
            List of frame indices
        """
        total_frames = int(video_duration * fps)
        if total_frames <= 0:
            return [0]
        
        if self.enabled:
            # Fixed evenly-spaced frames
            if count >= total_frames:
                return list(range(total_frames))
            step = total_frames / count
            return [int(i * step) for i in range(count)]
        else:
            # Random sampling
            if count >= total_frames:
                return list(range(total_frames))
            return sorted(random.sample(range(total_frames), count))
    
    def sort_candidates(self, items: List[Any], key_func=None) -> List[Any]:
        """
        Sort candidates for deterministic selection.
        
        In deterministic mode: Always sorts lexicographically
        In random mode: Returns items as-is (preserves API order)
        
        Args:
            items: List of items (paths, dicts, etc.)
            key_func: Optional key function for sorting
            
        Returns:
            Sorted list if deterministic, original otherwise
        """
        if self.enabled:
            if key_func:
                return sorted(items, key=key_func)
            # Default: sort by string representation
            return sorted(items, key=lambda x: str(x))
        return items
    
    def get_tts_seed(self) -> Optional[int]:
        """
        Get seed for TTS temperature sampling.
        
        Returns:
            Seed if deterministic, None otherwise
        """
        return self.seed if self.enabled else None
    
    def should_disable_grain(self) -> bool:
        """
        Check if film grain VFX should be disabled.
        
        Film grain uses random noise which breaks determinism.
        
        Returns:
            True if grain should be disabled
        """
        return self.enabled
    
    def get_noise_seed(self) -> Optional[int]:
        """
        Get seed for FFmpeg noise filter.
        
        Can be used with: -vf "noise=c0s=10:c0f=t+u:seed={seed}"
        
        Returns:
            Seed if deterministic, None otherwise
        """
        return self.seed if self.enabled else None
    
    def get_stft_params(self) -> dict:
        """
        Get STFT parameters for beat detection.
        
        Returns fixed parameters in deterministic mode for reproducibility.
        
        Returns:
            Dict with n_fft, hop_length, win_length
        """
        # These are standard librosa defaults, but we fix them explicitly
        return {
            "n_fft": 2048,
            "hop_length": 512,
            "win_length": 2048,
        }
    
    def select_best_on_tie(self, items: List[Any], scores: List[float]) -> Any:
        """
        Select best item when multiple have same score.
        
        In deterministic mode: Returns first in sorted order
        In random mode: Returns random among tied items
        
        Args:
            items: List of items
            scores: Corresponding scores
            
        Returns:
            Best item
        """
        if not items:
            return None
        
        max_score = max(scores)
        tied = [item for item, score in zip(items, scores) if score == max_score]
        
        if len(tied) == 1:
            return tied[0]
        
        if self.enabled:
            # Return first in sorted order
            return self.sort_candidates(tied)[0]
        else:
            # Random choice
            return random.choice(tied)
    
    def shuffle(self, items: List[Any]) -> List[Any]:
        """
        Shuffle items (only in non-deterministic mode).
        
        Args:
            items: List to shuffle
            
        Returns:
            Shuffled list (or original if deterministic)
        """
        if self.enabled:
            # Don't shuffle in deterministic mode
            return items
        
        result = items.copy()
        random.shuffle(result)
        return result


# =============================================================================
# GLOBAL CONTEXT MANAGEMENT
# =============================================================================

_context: Optional[DeterministicContext] = None


def init_deterministic_mode(enabled: bool = False, seed: int = 42) -> DeterministicContext:
    """
    Initialize deterministic mode globally.
    
    Call this at the start of video generation.
    
    Args:
        enabled: Whether to enable deterministic mode
        seed: Random seed to use
        
    Returns:
        DeterministicContext instance
    """
    global _context
    _context = DeterministicContext(enabled=enabled, seed=seed)
    return _context


def get_deterministic_context() -> DeterministicContext:
    """
    Get current deterministic context.
    
    Returns:
        Current context (creates disabled context if not initialized)
    """
    global _context
    if _context is None:
        _context = DeterministicContext(enabled=False)
    return _context


def is_deterministic() -> bool:
    """Check if deterministic mode is enabled."""
    return get_deterministic_context().enabled


def get_seed() -> Optional[int]:
    """Get current seed (None if not deterministic)."""
    ctx = get_deterministic_context()
    return ctx.seed if ctx.enabled else None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def deterministic_frame_sample(video_duration: float, count: int) -> List[int]:
    """Get frame indices respecting deterministic mode."""
    return get_deterministic_context().get_frame_indices(video_duration, count)


def deterministic_sort(items: List[Any]) -> List[Any]:
    """Sort items respecting deterministic mode."""
    return get_deterministic_context().sort_candidates(items)


def should_disable_grain() -> bool:
    """Check if film grain should be disabled."""
    return get_deterministic_context().should_disable_grain()
