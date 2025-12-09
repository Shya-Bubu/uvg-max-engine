"""
Unit tests for uvg_selector module.
"""

import pytest
import os
import tempfile
from pathlib import Path


class TestCacheModule:
    """Test cache.py functions."""
    
    def test_make_key_deterministic(self):
        from uvg_selector.cache import make_key
        
        key1 = make_key("test", "prompt", ["a", "b"])
        key2 = make_key("test", "prompt", ["a", "b"])
        
        assert key1 == key2
        assert len(key1) == 32  # MD5 hex length
    
    def test_make_key_different_inputs(self):
        from uvg_selector.cache import make_key
        
        key1 = make_key("prompt1")
        key2 = make_key("prompt2")
        
        assert key1 != key2
    
    def test_cache_roundtrip(self):
        from uvg_selector.cache import make_key, get_cache, set_cache
        
        key = make_key("test_roundtrip", str(os.getpid()))
        value = {"test": "data", "score": 0.95}
        
        set_cache(key, value)
        result = get_cache(key)
        
        assert result is not None
        assert result["test"] == "data"
        assert result["score"] == 0.95
    
    def test_cache_miss(self):
        from uvg_selector.cache import get_cache
        
        result = get_cache("nonexistent_key_12345")
        assert result is None


class TestAestheticScorer:
    """Test aesthetic.py functions."""
    
    def test_score_aesthetic_single(self):
        from uvg_selector.aesthetic import score_aesthetic
        import numpy as np
        
        # Single embedding
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        score = score_aesthetic(emb)
        
        assert score.shape == (1,)
        assert 0 <= score[0] <= 10
    
    def test_score_aesthetic_batch(self):
        from uvg_selector.aesthetic import score_aesthetic
        import numpy as np
        
        # Batch of embeddings
        embs = np.random.randn(5, 512).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        
        scores = score_aesthetic(embs)
        
        assert scores.shape == (5,)
        assert all(0 <= s <= 10 for s in scores)
    
    def test_scorer_fallback_mode(self):
        from uvg_selector.aesthetic import AestheticScorer
        
        # Should initialize in fallback mode since weights don't exist
        scorer = AestheticScorer("nonexistent_weights.npz")
        
        assert scorer.fallback_mode is True


class TestONNXCLIP:
    """Test onnx_clip.py functions."""
    
    def test_clip_fallback_mode(self):
        from uvg_selector.onnx_clip import ONNXCLIP
        
        # Should initialize in fallback mode since model doesn't exist
        clip = ONNXCLIP("nonexistent_model.onnx")
        
        assert clip.fallback_mode is True
    
    def test_fallback_text_embedding(self):
        from uvg_selector.onnx_clip import ONNXCLIP
        import numpy as np
        
        clip = ONNXCLIP("nonexistent_model.onnx")
        
        emb = clip.embed_text("test prompt")
        
        assert emb.shape == (512,)
        # Check L2 normalized (approximately)
        assert abs(np.linalg.norm(emb) - 1.0) < 0.01
    
    def test_fallback_text_deterministic(self):
        from uvg_selector.onnx_clip import ONNXCLIP
        import numpy as np
        
        clip = ONNXCLIP("nonexistent_model.onnx")
        
        emb1 = clip.embed_text("same prompt")
        emb2 = clip.embed_text("same prompt")
        
        assert np.allclose(emb1, emb2)


class TestDecordLoader:
    """Test decord_loader.py functions."""
    
    def test_get_video_info_invalid(self):
        from uvg_selector.decord_loader import get_video_info
        
        info = get_video_info("nonexistent_video.mp4")
        
        assert info["duration"] == 0
        assert info["fps"] == 0
    
    def test_sample_frames_invalid(self):
        from uvg_selector.decord_loader import sample_frames
        
        frames = sample_frames("nonexistent_video.mp4")
        
        assert frames == []


class TestClipSelector:
    """Test clip_selector.py main functions."""
    
    def test_cosine_sim(self):
        from uvg_selector.clip_selector import cosine_sim
        import numpy as np
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        
        sim = cosine_sim(a, b)
        assert abs(sim - 1.0) < 0.001
        
        c = np.array([0.0, 1.0, 0.0])
        sim2 = cosine_sim(a, c)
        assert abs(sim2) < 0.001
    
    def test_normalize_list(self):
        from uvg_selector.clip_selector import normalize_list
        
        values = [1.0, 5.0, 10.0]
        normalized = normalize_list(values)
        
        assert normalized[0] == 0.0
        assert normalized[2] == 1.0
        assert 0 <= normalized[1] <= 1
    
    def test_normalize_list_equal_values(self):
        from uvg_selector.clip_selector import normalize_list
        
        values = [5.0, 5.0, 5.0]
        normalized = normalize_list(values)
        
        assert all(v == 0.5 for v in normalized)
    
    def test_rank_clips_empty(self):
        from uvg_selector.clip_selector import rank_clips
        
        result = rank_clips("test prompt", [], use_cache=False)
        
        assert result == []


class TestTrimSegment:
    """Test trim_segment.py functions."""
    
    def test_compute_stability(self):
        from uvg_selector.trim_segment import _compute_stability
        import numpy as np
        
        # Stable signal
        stable = np.ones(20)
        stability = _compute_stability(stable)
        
        assert len(stability) == 20
        assert all(s > 0.5 for s in stability)  # High stability
    
    def test_find_best_trim_invalid_video(self):
        from uvg_selector.trim_segment import find_best_trim
        
        result = find_best_trim("nonexistent.mp4", "test prompt")
        
        assert "error" in result or result["score"] == 0.0


class TestMotionExtractor:
    """Test mv_extractor.py functions."""
    
    def test_get_motion_stats_invalid(self):
        from uvg_selector.mv_extractor import get_motion_stats
        
        stats = get_motion_stats("nonexistent.mp4")
        
        assert "mean_motion" in stats
        assert stats["mean_motion"] >= 0
    
    def test_compute_motion_match(self):
        from uvg_selector.mv_extractor import compute_motion_match
        
        # Medium motion should match "medium" target well
        stats = {"mean_motion": 0.5}
        score = compute_motion_match(stats, "medium")
        
        assert 0.5 <= score <= 1.0
