# uvg_selector/aesthetic.py
"""
Aesthetic scoring using LAION linear probe (or deterministic fallback).
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path for aesthetic weights
DEFAULT_WEIGHTS_PATH = "models/aesthetic_weights.npz"


class AestheticScorer:
    """
    Aesthetic scorer using LAION linear probe MLP.
    
    Maps CLIP embeddings to aesthetic scores (0-10).
    """
    
    def __init__(self, weights_path: str = None):
        """
        Initialize aesthetic scorer.
        
        Args:
            weights_path: Path to weights file (npz or json)
        """
        self.weights = None
        self.bias = None
        self.fallback_mode = False
        
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH
        
        weights_file = Path(weights_path)
        
        if weights_file.exists():
            try:
                self._load_weights(weights_file)
                logger.info(f"Loaded aesthetic weights from {weights_file}")
            except Exception as e:
                logger.warning(f"Failed to load aesthetic weights: {e}")
                self._init_fallback()
        else:
            logger.debug(f"Aesthetic weights not found: {weights_file}")
            logger.debug("Using deterministic fallback. To get weights:")
            logger.debug("  https://github.com/LAION-AI/aesthetic-predictor")
            self._init_fallback()
    
    def _load_weights(self, weights_file: Path):
        """Load weights from file."""
        if weights_file.suffix == ".npz":
            data = np.load(str(weights_file))
            self.weights = data["weights"]
            self.bias = data.get("bias", 0.0)
        else:
            # JSON format
            import json
            with open(weights_file) as f:
                data = json.load(f)
            self.weights = np.array(data["weights"], dtype=np.float32)
            self.bias = data.get("bias", 0.0)
    
    def _init_fallback(self):
        """Initialize deterministic fallback weights."""
        self.fallback_mode = True
        
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(12345)
        
        # Create simple linear mapping (512 -> 1)
        self.weights = rng.normal(scale=0.1, size=(512,)).astype(np.float32)
        self.bias = 5.0  # Center around 5
        
        logger.debug("Initialized fallback aesthetic weights (seed=12345)")
    
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute aesthetic scores for embeddings.
        
        Args:
            embeddings: CLIP embeddings (N x 512)
            
        Returns:
            Scores array (N,) in range [0, 10]
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        
        # Handle dimension mismatch
        if embeddings.shape[1] != len(self.weights):
            # Pad or truncate
            if embeddings.shape[1] < len(self.weights):
                pad_width = len(self.weights) - embeddings.shape[1]
                embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)))
            else:
                embeddings = embeddings[:, :len(self.weights)]
        
        # Linear projection
        raw_scores = embeddings.dot(self.weights) + self.bias
        
        # Normalize to [0, 10]
        if len(raw_scores) > 1:
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s - min_s > 1e-9:
                raw_scores = (raw_scores - min_s) / (max_s - min_s) * 10.0
            else:
                raw_scores = np.full_like(raw_scores, 5.0)
        else:
            # Single sample: use sigmoid-like mapping
            raw_scores = 10.0 / (1.0 + np.exp(-raw_scores))
        
        return np.clip(raw_scores, 0, 10)


# Module-level scorer instance
_scorer = None


def get_scorer(weights_path: str = None) -> AestheticScorer:
    """Get or create aesthetic scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = AestheticScorer(weights_path)
    return _scorer


def score_aesthetic(embeddings: np.ndarray) -> np.ndarray:
    """
    Score aesthetic quality of CLIP embeddings.
    
    Args:
        embeddings: CLIP embeddings (N x D) or (D,)
        
    Returns:
        Aesthetic scores (N,) in range [0, 10]
    """
    scorer = get_scorer()
    return scorer.score(embeddings)
