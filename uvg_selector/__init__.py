# uvg_selector - Local Clip Selection + Smart Trimming Module
"""
UVG Selector: Production-ready local clip selection and trimming.

Features:
- ONNX CLIP-based semantic matching
- Aesthetic scoring via LAION linear probe
- Motion vector analysis
- Smart segment trimming (3-6s)
- MD5-based caching

Usage:
    from uvg_selector.clip_selector import rank_clips
    candidates = rank_clips(prompt, clip_paths, top_k=5)
    best_clip = candidates[0]["path"]
"""

from .clip_selector import rank_clips
from .trim_segment import find_best_trim, extract_ffmpeg_segment

__version__ = "1.0.0"
__all__ = ["rank_clips", "find_best_trim", "extract_ffmpeg_segment"]
