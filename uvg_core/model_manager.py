# uvg_core/model_manager.py
"""
Centralized Model Manager for UVG MAX.

Singleton pattern to load AI models once and reuse them.
Handles CLIP, aesthetic, YOLO with GPU/CPU fallback.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Default model paths
MODELS_DIR = Path("models")
CLIP_VISUAL_PATH = MODELS_DIR / "clip-vit-b-32-visual.onnx"
CLIP_TEXT_PATH = MODELS_DIR / "clip-vit-b-32-text.onnx"
AESTHETIC_PATH = MODELS_DIR / "aesthetic_weights.npz"


class ModelManager:
    """
    Singleton model manager for all AI models.
    
    Usage:
        manager = ModelManager.instance()
        clip = manager.get_clip_session()
    """
    
    _instance: Optional["ModelManager"] = None
    
    @classmethod
    def instance(cls) -> "ModelManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            cls._instance.release_models()
        cls._instance = None
    
    def __init__(self):
        """Initialize model manager (use instance() instead)."""
        self._clip_visual = None
        self._clip_text = None
        self._aesthetic = None
        self._yolo = None
        self._device = self._detect_device()
        self._providers = self._get_providers()
        
        logger.info(f"ModelManager initialized (device={self._device})")
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        # Check for CUDA via torch
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        # Check for CUDA via onnxruntime
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except ImportError:
            pass
        
        return "cpu"
    
    def _get_providers(self) -> list:
        """Get ONNX execution providers."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            
            if self._device == "cuda" and "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        except ImportError:
            return []
    
    @property
    def device(self) -> str:
        """Current device (cuda or cpu)."""
        return self._device
    
    def get_clip_session(self) -> Any:
        """
        Get cached CLIP visual encoder session.
        
        Returns:
            ONNXCLIP instance (or creates one)
        """
        if self._clip_visual is None:
            try:
                from uvg_selector.onnx_clip import ONNXCLIP
                self._clip_visual = ONNXCLIP(
                    str(CLIP_VISUAL_PATH),
                    provider_preference=self._providers
                )
                logger.info("Loaded CLIP visual encoder")
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")
                # Return fallback
                from uvg_selector.onnx_clip import ONNXCLIP
                self._clip_visual = ONNXCLIP()  # Fallback mode
        
        return self._clip_visual
    
    def get_text_encoder(self) -> Any:
        """
        Get cached CLIP text encoder session.
        
        Returns:
            ONNX InferenceSession or None
        """
        if self._clip_text is None:
            if not CLIP_TEXT_PATH.exists():
                logger.debug("CLIP text encoder not found, using visual encoder's text method")
                return None
            
            try:
                import onnxruntime as ort
                self._clip_text = ort.InferenceSession(
                    str(CLIP_TEXT_PATH),
                    providers=self._providers
                )
                logger.info("Loaded CLIP text encoder")
            except Exception as e:
                logger.warning(f"Failed to load text encoder: {e}")
        
        return self._clip_text
    
    def get_aesthetic_model(self) -> Any:
        """
        Get cached aesthetic scorer.
        
        Returns:
            AestheticScorer instance
        """
        if self._aesthetic is None:
            try:
                from uvg_selector.aesthetic import AestheticScorer
                self._aesthetic = AestheticScorer(str(AESTHETIC_PATH))
                logger.info("Loaded aesthetic model")
            except Exception as e:
                logger.warning(f"Failed to load aesthetic model: {e}")
                from uvg_selector.aesthetic import AestheticScorer
                self._aesthetic = AestheticScorer()  # Fallback
        
        return self._aesthetic
    
    def get_object_detector(self) -> Any:
        """
        Get cached YOLO detector (optional).
        
        Returns:
            YOLO instance or None if not available
        """
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
                logger.info("Loaded YOLOv8n detector")
            except ImportError:
                logger.debug("ultralytics not installed, YOLO disabled")
                self._yolo = False  # Mark as unavailable
            except Exception as e:
                logger.debug(f"YOLO load failed: {e}")
                self._yolo = False
        
        return self._yolo if self._yolo is not False else None
    
    def release_models(self) -> None:
        """Release all models to free memory."""
        self._clip_visual = None
        self._clip_text = None
        self._aesthetic = None
        self._yolo = None
        
        # Try to clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Released all models")
    
    def get_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                return allocated / (1024 * 1024)
        except ImportError:
            pass
        return 0.0
    
    def check_models_available(self) -> dict:
        """Check which models are available."""
        return {
            "clip_visual": CLIP_VISUAL_PATH.exists(),
            "clip_text": CLIP_TEXT_PATH.exists(),
            "aesthetic": AESTHETIC_PATH.exists(),
            "yolo": self.get_object_detector() is not None,
            "device": self._device,
        }


# Convenience function
def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager.instance()
