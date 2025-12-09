# uvg_selector/clip_selector.py
"""
Main clip selection orchestrator.
Ranks clips by semantic similarity, aesthetics, motion, and more.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .decord_loader import sample_frames, get_video_info
from .onnx_clip import ONNXCLIP
from .aesthetic import score_aesthetic
from .mv_extractor import get_motion_stats
from .cache import make_key, get_cache, set_cache

logger = logging.getLogger(__name__)

# Default configuration
DEFAULTS = {
    "frame_sample_fps": 1.0,
    "max_sample_frames": 16,
    "top_k": 5,
    "weights": {
        "semantic": 0.30,
        "motion": 0.15,
        "objects": 0.15,
        "emotion": 0.15,
        "aesthetic": 0.15,
        "temporal": 0.10
    },
    "clip_model": "models/clip-vit-b-32.onnx"
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def normalize_list(values: List[float]) -> List[float]:
    """Normalize list of values to [0, 1] range."""
    if not values:
        return []
    
    mi, ma = min(values), max(values)
    if ma - mi < 1e-9:
        return [0.5] * len(values)
    
    return [(v - mi) / (ma - mi) for v in values]


def rank_clips(
    prompt: str,
    clip_paths: List[str],
    top_k: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Rank video clips by relevance to prompt.
    
    Args:
        prompt: Text prompt describing desired content
        clip_paths: List of paths to video files
        top_k: Number of top clips to return
        config: Override configuration dict
        use_cache: Whether to use cached results
        
    Returns:
        List of dicts with keys: path, final_score, signals, cached
    """
    start_time = time.time()
    
    # Merge config with defaults
    cfg = DEFAULTS.copy()
    if config:
        cfg.update(config)
    
    if top_k is None:
        top_k = cfg["top_k"]
    
    # Check cache
    cache_key = make_key("rank", prompt, tuple(sorted(clip_paths)))
    if use_cache:
        cached = get_cache(cache_key)
        if cached:
            logger.info(f"Cache hit for prompt: {prompt[:50]}...")
            for item in cached:
                item["cached"] = True
            return cached
    
    logger.info(f"Ranking {len(clip_paths)} clips for: {prompt[:50]}...")
    
    # Load CLIP model
    try:
        clip = ONNXCLIP(cfg["clip_model"])
    except Exception as e:
        logger.warning(f"CLIP model load failed: {e}")
        clip = ONNXCLIP()  # Use fallback mode
    
    # Compute text embedding once
    text_emb = clip.embed_text(prompt)
    
    # Process each clip
    results = []
    
    for path in clip_paths:
        try:
            clip_result = _process_clip(
                path, text_emb, clip, cfg
            )
            results.append(clip_result)
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
            results.append({
                "path": path,
                "signals": {
                    "semantic": 0.0,
                    "aesthetic": 0.0,
                    "motion": 0.0,
                    "objects": 0.5,
                    "emotion": 0.5,
                    "temporal": 0.5
                },
                "error": str(e)
            })
    
    # Extract signal arrays for normalization
    if results:
        sem_scores = [r["signals"]["semantic"] for r in results]
        aes_scores = [r["signals"]["aesthetic"] for r in results]
        mot_scores = [r["signals"]["motion"] for r in results]
        obj_scores = [r["signals"]["objects"] for r in results]
        
        # Normalize
        sem_norm = normalize_list(sem_scores)
        aes_norm = normalize_list(aes_scores)
        mot_norm = normalize_list(mot_scores)
        obj_norm = normalize_list(obj_scores)
        
        # Compute final scores
        w = cfg["weights"]
        for i, r in enumerate(results):
            final = (
                w["semantic"] * sem_norm[i] +
                w["aesthetic"] * aes_norm[i] +
                w["motion"] * mot_norm[i] +
                w["objects"] * obj_norm[i] +
                w["emotion"] * 0.5 +  # Placeholder
                w["temporal"] * 0.5   # Placeholder
            )
            r["final_score"] = float(final)
            r["cached"] = False
    
    # Sort by final score
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    # Limit to top_k
    top_results = results[:top_k]
    
    # Cache results
    if use_cache:
        set_cache(cache_key, top_results)
    
    elapsed = time.time() - start_time
    logger.info(f"Ranked {len(clip_paths)} clips in {elapsed:.2f}s")
    
    return top_results


def _process_clip(
    path: str,
    text_emb: np.ndarray,
    clip: ONNXCLIP,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process single clip and compute all signals.
    """
    # Sample frames
    frames = sample_frames(
        path,
        fps=cfg["frame_sample_fps"],
        max_frames=cfg["max_sample_frames"]
    )
    
    if not frames:
        return {
            "path": path,
            "signals": {
                "semantic": 0.0,
                "aesthetic": 5.0,
                "motion": 0.5,
                "objects": 0.5,
                "emotion": 0.5,
                "temporal": 0.5
            },
            "frame_count": 0
        }
    
    # Compute frame embeddings
    emb_batch = clip.embed_batch(frames)
    
    # Semantic similarity
    similarities = [cosine_sim(text_emb, emb) for emb in emb_batch]
    sem_mean = float(np.mean(similarities))
    sem_max = float(np.max(similarities))
    
    # Aesthetic scores
    aes_scores = score_aesthetic(emb_batch)
    aes_mean = float(np.mean(aes_scores))
    
    # Motion stats
    motion_stats = get_motion_stats(path, sample_rate=cfg["frame_sample_fps"])
    motion_score = motion_stats.get("mean_motion", 0.5)
    
    # Normalize motion to 0-1 range (cap at reasonable values)
    motion_norm = min(1.0, motion_score / 2.0) if motion_score > 0 else 0.5
    
    # Object detection placeholder (would use YOLO if available)
    object_score = 0.5
    
    # Get video info for temporal signal
    info = get_video_info(path)
    duration = info.get("duration", 0)
    
    # Temporal: prefer clips in 3-10s range
    if 3 <= duration <= 10:
        temporal_score = 1.0
    elif duration < 3:
        temporal_score = duration / 3.0
    else:
        temporal_score = max(0.5, 1.0 - (duration - 10) / 20.0)
    
    return {
        "path": path,
        "signals": {
            "semantic": sem_mean,
            "semantic_max": sem_max,
            "aesthetic": aes_mean,
            "motion": motion_norm,
            "objects": object_score,
            "emotion": 0.5,  # Placeholder
            "temporal": temporal_score
        },
        "frame_count": len(frames),
        "duration": duration
    }


def select_best_clip(
    prompt: str,
    clip_paths: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Select the single best clip for a prompt.
    
    Args:
        prompt: Text prompt
        clip_paths: List of candidate clip paths
        config: Optional configuration override
        
    Returns:
        Dict with best clip info
    """
    results = rank_clips(prompt, clip_paths, top_k=1, config=config)
    if results:
        return results[0]
    return {"path": clip_paths[0] if clip_paths else "", "final_score": 0.0}
