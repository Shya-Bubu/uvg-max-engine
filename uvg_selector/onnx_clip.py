# uvg_selector/onnx_clip.py
"""
ONNX-based CLIP wrapper for text and image embeddings.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False
    logger.warning("onnxruntime not available - using fallback embeddings")

# Try to import OpenCV for preprocessing
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

DEFAULT_MODEL = "models/clip-vit-b-32.onnx"

# CLIP normalization constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class ONNXCLIP:
    """
    ONNX CLIP model wrapper for embeddings.
    
    Supports:
    - Image embedding (single and batch)
    - Text embedding
    - GPU execution when available
    - Fallback mode when model not found
    """
    
    def __init__(self, model_path: str = None, provider_preference: list = None):
        """
        Initialize ONNX CLIP model.
        
        Args:
            model_path: Path to ONNX model file
            provider_preference: List of execution providers
        """
        if model_path is None:
            model_path = DEFAULT_MODEL
        
        self.model_path = Path(model_path)
        self.sess = None
        self.fallback_mode = False
        
        if not HAVE_ORT:
            logger.warning("ONNX Runtime not available - using fallback mode")
            self.fallback_mode = True
            return
        
        if not self.model_path.exists():
            logger.warning(f"ONNX CLIP model not found: {self.model_path}")
            logger.warning("Using fallback embedding mode. To get the model:")
            logger.warning("  1. Download from: https://huggingface.co/openai/clip-vit-base-patch32")
            logger.warning("  2. Export to ONNX or use: pip install open_clip_torch")
            self.fallback_mode = True
            return
        
        # Initialize ONNX session
        try:
            providers = provider_preference or self._get_providers()
            self.sess = ort.InferenceSession(str(self.model_path), providers=providers)
            logger.info(f"Loaded CLIP model: {self.model_path}")
            logger.info(f"Using providers: {self.sess.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.fallback_mode = True
    
    def _get_providers(self) -> list:
        """Get available execution providers, preferring GPU."""
        available = ort.get_available_providers()
        preferred = []
        
        if "CUDAExecutionProvider" in available:
            preferred.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            preferred.append("CPUExecutionProvider")
        
        return preferred if preferred else available
    
    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CLIP.
        
        Args:
            img_bgr: BGR image (HxWx3)
            
        Returns:
            Preprocessed tensor (1x3x224x224)
        """
        if not HAVE_CV2:
            # Simple fallback without cv2
            return np.zeros((1, 3, 224, 224), dtype=np.float32)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply CLIP normalization
        img = (img - CLIP_MEAN) / CLIP_STD
        
        # HWC to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        return img[None, ...].astype(np.float32)
    
    def embed_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Compute embedding for single frame.
        
        Args:
            img_bgr: BGR image
            
        Returns:
            L2-normalized embedding vector
        """
        if self.fallback_mode:
            return self._fallback_image_embedding(img_bgr)
        
        try:
            inp = self.preprocess_image(img_bgr)
            
            # Get input name from model
            input_name = self.sess.get_inputs()[0].name
            out = self.sess.run(None, {input_name: inp})[0]
            
            # L2 normalize
            out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
            return out[0]
            
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, using fallback")
            return self._fallback_image_embedding(img_bgr)
    
    def embed_batch(self, imgs_bgr: list) -> np.ndarray:
        """
        Compute embeddings for batch of frames.
        
        Args:
            imgs_bgr: List of BGR images
            
        Returns:
            Array of L2-normalized embeddings (Nx512)
        """
        if not imgs_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        
        if self.fallback_mode:
            return np.array([self._fallback_image_embedding(img) for img in imgs_bgr])
        
        try:
            # Preprocess all images
            batch = np.concatenate([self.preprocess_image(img) for img in imgs_bgr], axis=0)
            
            # Get input name from model
            input_name = self.sess.get_inputs()[0].name
            out = self.sess.run(None, {input_name: batch})[0]
            
            # L2 normalize
            out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
            return out
            
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}, using fallback")
            return np.array([self._fallback_image_embedding(img) for img in imgs_bgr])
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Compute embedding for text prompt.
        
        Args:
            text: Text string
            
        Returns:
            L2-normalized embedding vector
        """
        if self.fallback_mode:
            return self._fallback_text_embedding(text)
        
        try:
            # Try to find text input in model
            input_names = [inp.name for inp in self.sess.get_inputs()]
            
            # Common text input names
            text_input_name = None
            for name in ["input_text", "text", "input_ids", "text_input"]:
                if name in input_names:
                    text_input_name = name
                    break
            
            if text_input_name:
                out = self.sess.run(None, {text_input_name: np.array([text])})[0]
                out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
                return out[0]
            else:
                # Model may be vision-only, use fallback
                logger.debug("Text encoder not found in ONNX model, using fallback")
                return self._fallback_text_embedding(text)
                
        except Exception as e:
            logger.debug(f"Text embedding failed: {e}, using fallback")
            return self._fallback_text_embedding(text)
    
    def _fallback_image_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Fallback image embedding when model not available.
        Uses deterministic hash of resized image.
        """
        try:
            if HAVE_CV2:
                # Resize to small size and flatten
                small = cv2.resize(img_bgr, (16, 16))
                flat = small.flatten().astype(np.float32)
            else:
                flat = np.zeros(768, dtype=np.float32)
            
            # Pad/truncate to 512 dimensions
            if len(flat) < 512:
                flat = np.pad(flat, (0, 512 - len(flat)))
            else:
                flat = flat[:512]
            
            # Normalize
            flat = flat / (np.linalg.norm(flat) + 1e-12)
            return flat
            
        except Exception:
            return np.zeros(512, dtype=np.float32)
    
    def _fallback_text_embedding(self, text: str) -> np.ndarray:
        """
        Fallback text embedding when model not available.
        Uses deterministic hash of text.
        """
        # Create deterministic vector from text bytes
        text_bytes = text.encode('utf-8')[:512].ljust(512, b'\0')
        v = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.float32)
        
        # Add some structure using hash
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        h_arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        h_arr = np.tile(h_arr, 32)[:512]  # Repeat to 512 dims
        
        # Combine
        v = v * 0.5 + h_arr * 0.5
        v = v / (np.linalg.norm(v) + 1e-12)
        return v


# Convenience functions
def get_clip_model(model_path: str = None) -> ONNXCLIP:
    """Get new ONNX CLIP model instance."""
    return ONNXCLIP(model_path)


def get_managed_clip() -> ONNXCLIP:
    """Get CLIP model from ModelManager (singleton, reused)."""
    try:
        from uvg_core.model_manager import get_model_manager
        return get_model_manager().get_clip_session()
    except ImportError:
        # Fallback if model_manager not available
        return ONNXCLIP()
