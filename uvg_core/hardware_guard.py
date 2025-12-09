# uvg_core/hardware_guard.py
"""
UVG MAX Hardware Guard.

Detects GPU availability, manages fallbacks, prevents OOM crashes.
"""

import os
import subprocess
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def get_available_device() -> str:
    """
    Detect best available compute device.
    
    Returns:
        "cuda" or "cpu"
    """
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.debug("CUDA available via PyTorch")
            return "cuda"
    except ImportError:
        pass
    
    # Check ONNX Runtime CUDA
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            logger.debug("CUDA available via ONNX Runtime")
            return "cuda"
    except ImportError:
        pass
    
    logger.debug("Using CPU (no CUDA detected)")
    return "cpu"


def get_gpu_info() -> dict:
    """
    Get GPU information.
    
    Returns:
        Dict with name, memory_total_mb, memory_free_mb
    """
    result = {
        "available": False,
        "name": "None",
        "memory_total_mb": 0,
        "memory_free_mb": 0,
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["name"] = torch.cuda.get_device_name(0)
            
            free, total = torch.cuda.mem_get_info()
            result["memory_total_mb"] = total // (1024 * 1024)
            result["memory_free_mb"] = free // (1024 * 1024)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU info error: {e}")
    
    return result


def get_vram_mb() -> int:
    """
    Get available VRAM in MB.
    
    Returns:
        Available VRAM in MB, or 0 if no GPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except:
        pass
    return 0


def check_vram_sufficient(required_mb: int = 500) -> bool:
    """
    Check if sufficient VRAM is available.
    
    Args:
        required_mb: Required VRAM in MB
        
    Returns:
        True if sufficient VRAM available
    """
    available = get_vram_mb()
    return available >= required_mb


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared GPU memory cache")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU memory clear error: {e}")


def check_ffmpeg_available() -> bool:
    """
    Check if FFmpeg is available on PATH.
    
    Returns:
        True if FFmpeg is available
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def check_ffmpeg_nvenc() -> bool:
    """
    Check if FFmpeg has NVENC support.
    
    Returns:
        True if NVENC encoding is available
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "h264_nvenc" in result.stdout
    except:
        return False


def get_ffmpeg_encoder() -> Tuple[str, list]:
    """
    Get best available FFmpeg encoder.
    
    Returns:
        Tuple of (encoder_name, encoder_args)
    """
    if check_ffmpeg_nvenc() and get_available_device() == "cuda":
        return "h264_nvenc", ["-preset", "p4", "-rc", "vbr"]
    
    return "libx264", ["-preset", "fast", "-crf", "23"]


def get_onnx_providers() -> list:
    """
    Get ONNX Runtime execution providers with fallback.
    
    Returns:
        List of provider names
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        
        # Check if GPU has enough memory
        if "CUDAExecutionProvider" in available and check_vram_sufficient(500):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        return ["CPUExecutionProvider"]
    except ImportError:
        return ["CPUExecutionProvider"]


def get_system_info() -> dict:
    """
    Get complete system information.
    
    Returns:
        Dict with device, GPU, FFmpeg info
    """
    gpu_info = get_gpu_info()
    
    return {
        "device": get_available_device(),
        "gpu_available": gpu_info["available"],
        "gpu_name": gpu_info["name"],
        "vram_total_mb": gpu_info["memory_total_mb"],
        "vram_free_mb": gpu_info["memory_free_mb"],
        "ffmpeg_available": check_ffmpeg_available(),
        "nvenc_available": check_ffmpeg_nvenc(),
        "onnx_providers": get_onnx_providers(),
    }


class HardwareGuard:
    """
    Hardware guard for managing device resources.
    
    Usage:
        guard = HardwareGuard()
        if guard.can_use_gpu():
            # Use GPU
        else:
            # Fallback to CPU
    """
    
    def __init__(self, min_vram_mb: int = 500):
        self.min_vram_mb = min_vram_mb
        self._device = None
        self._refresh()
    
    def _refresh(self) -> None:
        """Refresh device detection."""
        self._device = get_available_device()
    
    @property
    def device(self) -> str:
        """Current device."""
        return self._device
    
    def can_use_gpu(self) -> bool:
        """Check if GPU can be used."""
        return self._device == "cuda" and check_vram_sufficient(self.min_vram_mb)
    
    def get_onnx_providers(self) -> list:
        """Get ONNX providers based on current state."""
        return get_onnx_providers()
    
    def get_ffmpeg_encoder(self) -> Tuple[str, list]:
        """Get FFmpeg encoder based on current state."""
        return get_ffmpeg_encoder()
    
    def release_memory(self) -> None:
        """Release GPU memory and refresh state."""
        clear_gpu_memory()
        self._refresh()
