# uvg_core/colab_resource_manager.py
"""
Colab Resource Manager for UVG MAX.

Manages resources on Colab T4 environment:
- 10GB RAM
- 15GB VRAM (T4)
- 90GB storage

Ensures models are loaded/unloaded properly to prevent crashes.
"""

import gc
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Callable, Any, List
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ColabResources:
    """Current Colab resource status."""
    ram_total_gb: float = 10.0
    ram_used_gb: float = 0.0
    ram_free_gb: float = 10.0
    
    vram_total_gb: float = 15.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 15.0
    
    storage_total_gb: float = 90.0
    storage_used_gb: float = 0.0
    storage_free_gb: float = 90.0
    
    gpu_name: str = "Unknown"
    is_t4: bool = False


class ColabResourceManager:
    """
    Manages Colab resources to prevent OOM and crashes.
    
    Key features:
    - Sequential model loading (one at a time)
    - Automatic cleanup between stages
    - Memory monitoring
    - Storage management
    
    Usage:
        manager = ColabResourceManager()
        
        with manager.load_model("fish_speech"):
            # Use Fish-Speech
            pass
        # Model automatically unloaded
        
        with manager.load_model("whisper"):
            # Use Whisper
            pass
    """
    
    # Models that can be loaded (only one at a time!)
    MODELS = {
        "fish_speech": {
            "vram_gb": 4.0,
            "ram_gb": 2.0,
            "description": "Fish-Speech S1 TTS"
        },
        "whisper": {
            "vram_gb": 2.0,
            "ram_gb": 1.5,
            "description": "Whisper ASR (base)"
        },
        "whisper_small": {
            "vram_gb": 3.0,
            "ram_gb": 2.0,
            "description": "Whisper ASR (small)"
        },
        "clip": {
            "vram_gb": 4.0,
            "ram_gb": 2.0,
            "description": "CLIP ViT-B/32"
        },
        "depth_anything": {
            "vram_gb": 4.0,
            "ram_gb": 2.0,
            "description": "DepthAnything V2"
        },
    }
    
    def __init__(self, output_dir: Path = None):
        """Initialize resource manager."""
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._loaded_models: List[str] = []
        self._torch_available = False
        self._init_torch()
    
    def _init_torch(self):
        """Initialize PyTorch for GPU monitoring."""
        try:
            import torch
            self._torch_available = True
            self._torch = torch
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
            self._torch_available = False
    
    def get_status(self) -> ColabResources:
        """Get current resource status."""
        status = ColabResources()
        
        # RAM status
        try:
            import psutil
            mem = psutil.virtual_memory()
            status.ram_total_gb = mem.total / (1024**3)
            status.ram_used_gb = mem.used / (1024**3)
            status.ram_free_gb = mem.available / (1024**3)
        except ImportError:
            pass
        
        # VRAM status
        if self._torch_available and self._torch.cuda.is_available():
            try:
                props = self._torch.cuda.get_device_properties(0)
                status.gpu_name = props.name
                status.is_t4 = "T4" in props.name
                status.vram_total_gb = props.total_memory / (1024**3)
                status.vram_used_gb = self._torch.cuda.memory_allocated(0) / (1024**3)
                status.vram_free_gb = status.vram_total_gb - status.vram_used_gb
            except Exception as e:
                logger.warning(f"GPU query failed: {e}")
        
        # Storage status
        try:
            disk = shutil.disk_usage(str(self.output_dir))
            status.storage_total_gb = disk.total / (1024**3)
            status.storage_used_gb = disk.used / (1024**3)
            status.storage_free_gb = disk.free / (1024**3)
        except Exception:
            pass
        
        return status
    
    def print_status(self):
        """Print current resource status."""
        status = self.get_status()
        print(f"""
+----------------------------------------------------------+
|                   COLAB RESOURCES                        |
+----------------------------------------------------------+
| GPU: {status.gpu_name:20} {'(T4)' if status.is_t4 else '':>6}               |
| VRAM: {status.vram_used_gb:.1f}GB / {status.vram_total_gb:.1f}GB ({status.vram_free_gb:.1f}GB free)                 |
| RAM:  {status.ram_used_gb:.1f}GB / {status.ram_total_gb:.1f}GB ({status.ram_free_gb:.1f}GB free)                  |
| Storage: {status.storage_used_gb:.1f}GB / {status.storage_total_gb:.1f}GB ({status.storage_free_gb:.1f}GB free)         |
| Loaded Models: {', '.join(self._loaded_models) or 'None':40} |
+----------------------------------------------------------+
""")
    
    def clear_all(self):
        """Clear all GPU memory and loaded models."""
        logger.info("Clearing all GPU memory...")
        
        self._loaded_models.clear()
        
        if self._torch_available:
            self._torch.cuda.empty_cache()
        
        gc.collect()
        
        logger.info("Memory cleared")
    
    def cleanup_stage(self, stage_name: str):
        """
        Cleanup after a pipeline stage.
        
        Call this between major stages to free memory.
        """
        logger.info(f"Cleaning up after stage: {stage_name}")
        self.clear_all()
        
        status = self.get_status()
        logger.info(f"After cleanup: {status.vram_free_gb:.1f}GB VRAM, {status.ram_free_gb:.1f}GB RAM free")
    
    def can_load_model(self, model_name: str) -> bool:
        """Check if we have enough resources to load a model."""
        if model_name not in self.MODELS:
            return True  # Unknown model, try anyway
        
        model_info = self.MODELS[model_name]
        status = self.get_status()
        
        vram_needed = model_info["vram_gb"]
        ram_needed = model_info["ram_gb"]
        
        vram_ok = status.vram_free_gb >= vram_needed
        ram_ok = status.ram_free_gb >= ram_needed
        
        if not vram_ok:
            logger.warning(f"Not enough VRAM for {model_name}: need {vram_needed}GB, have {status.vram_free_gb:.1f}GB")
        if not ram_ok:
            logger.warning(f"Not enough RAM for {model_name}: need {ram_needed}GB, have {status.ram_free_gb:.1f}GB")
        
        return vram_ok and ram_ok
    
    @contextmanager
    def load_model(self, model_name: str, force_cleanup: bool = True):
        """
        Context manager for loading a model.
        
        Ensures only one model is loaded at a time and cleans up after.
        
        Usage:
            with manager.load_model("fish_speech"):
                # Model loaded, use it here
                result = fish_speech.synthesize(text)
            # Model automatically unloaded
        """
        if force_cleanup and self._loaded_models:
            logger.info(f"Unloading existing models before loading {model_name}")
            self.clear_all()
        
        if not self.can_load_model(model_name):
            logger.warning(f"Low resources for {model_name}, attempting anyway...")
        
        self._loaded_models.append(model_name)
        logger.info(f"Loading model: {model_name}")
        
        try:
            yield
        finally:
            logger.info(f"Unloading model: {model_name}")
            if model_name in self._loaded_models:
                self._loaded_models.remove(model_name)
            self.clear_all()
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than max_age_hours."""
        import time
        
        temp_dirs = [
            self.output_dir / "clips",
            self.output_dir / "audio",
            self.output_dir / "temp",
        ]
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        freed_mb = 0
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_size = file_path.stat().st_size / (1024 * 1024)
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            freed_mb += file_size
                        except Exception:
                            pass
        
        logger.info(f"Cleaned up {deleted_count} files, freed {freed_mb:.1f}MB")
    
    def ensure_storage(self, needed_gb: float = 5.0) -> bool:
        """Ensure we have enough storage space."""
        status = self.get_status()
        
        if status.storage_free_gb >= needed_gb:
            return True
        
        logger.warning(f"Low storage: {status.storage_free_gb:.1f}GB, need {needed_gb}GB")
        
        # Try cleanup
        self.cleanup_temp_files(max_age_hours=1)
        
        status = self.get_status()
        return status.storage_free_gb >= needed_gb


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_manager: Optional[ColabResourceManager] = None


def get_resource_manager() -> ColabResourceManager:
    """Get global resource manager."""
    global _manager
    if _manager is None:
        _manager = ColabResourceManager()
    return _manager


def print_resources():
    """Print current resource status."""
    get_resource_manager().print_status()


def cleanup_stage(stage_name: str):
    """Cleanup after a pipeline stage."""
    get_resource_manager().cleanup_stage(stage_name)


def load_model(model_name: str):
    """Context manager for loading a model."""
    return get_resource_manager().load_model(model_name)
