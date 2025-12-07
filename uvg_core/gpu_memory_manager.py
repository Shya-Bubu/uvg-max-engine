"""
UVG MAX GPU Memory Manager Module

Monitors GPU memory usage, auto-switches to CPU, manages batch sizes.
Prevents CUDA OOM errors on Colab T4 and other environments.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Any
import gc

logger = logging.getLogger(__name__)


@dataclass
class MemoryStatus:
    """Current GPU memory status."""
    available: bool = False
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    utilization_percent: float = 0.0


class GPUMemoryManager:
    """
    Manages GPU memory to prevent OOM errors.
    
    Features:
    - Monitor VRAM usage
    - Auto-switch to CPU when memory high
    - Dynamic batch size adjustment
    - Cache clearing
    """
    
    # Thresholds
    HIGH_USAGE_THRESHOLD = 0.85  # 85% - consider switching to CPU
    CRITICAL_USAGE_THRESHOLD = 0.95  # 95% - force switch to CPU
    SAFE_USAGE_THRESHOLD = 0.70  # 70% - safe to use GPU
    
    def __init__(self, auto_manage: bool = True):
        """
        Initialize GPU memory manager.
        
        Args:
            auto_manage: If True, automatically manage memory
        """
        self.auto_manage = auto_manage
        self._torch_available = False
        self._torch = None
        self._forced_cpu = False
        
        # Try to import torch
        try:
            import torch
            self._torch = torch
            self._torch_available = torch.cuda.is_available()
        except ImportError:
            logger.debug("PyTorch not available, GPU management disabled")
    
    def get_status(self) -> MemoryStatus:
        """
        Get current GPU memory status.
        
        Returns:
            MemoryStatus with current usage
        """
        status = MemoryStatus()
        
        if not self._torch_available or self._torch is None:
            return status
        
        try:
            status.available = self._torch.cuda.is_available()
            if status.available:
                status.total_gb = self._torch.cuda.get_device_properties(0).total_memory / (1024**3)
                status.used_gb = self._torch.cuda.memory_allocated(0) / (1024**3)
                status.free_gb = status.total_gb - status.used_gb
                status.utilization_percent = (status.used_gb / status.total_gb) * 100 if status.total_gb > 0 else 0
        except Exception as e:
            logger.debug(f"Failed to get GPU memory status: {e}")
        
        return status
    
    def monitor_usage(self) -> float:
        """
        Get current GPU memory utilization as percentage.
        
        Returns:
            Utilization 0.0-1.0 (or 0.0 if no GPU)
        """
        status = self.get_status()
        return status.utilization_percent / 100.0
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        return self.monitor_usage() >= self.CRITICAL_USAGE_THRESHOLD
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is high."""
        return self.monitor_usage() >= self.HIGH_USAGE_THRESHOLD
    
    def is_memory_safe(self) -> bool:
        """Check if memory usage is safe."""
        return self.monitor_usage() <= self.SAFE_USAGE_THRESHOLD
    
    def should_use_cpu(self) -> bool:
        """
        Determine if operations should use CPU instead of GPU.
        
        Returns:
            True if CPU should be used
        """
        if self._forced_cpu:
            return True
        
        if not self._torch_available:
            return True
        
        if self.auto_manage and self.is_memory_critical():
            logger.warning("GPU memory critical, switching to CPU")
            return True
        
        return False
    
    def auto_switch_to_cpu(self) -> bool:
        """
        Auto-switch to CPU if memory is too high.
        
        Returns:
            True if switched to CPU
        """
        if self.is_memory_high():
            logger.warning(f"GPU memory at {self.monitor_usage()*100:.1f}%, switching to CPU")
            self._forced_cpu = True
            self.clear_cache()
            return True
        return False
    
    def force_cpu_mode(self) -> None:
        """Force CPU mode."""
        self._forced_cpu = True
        self.clear_cache()
        logger.info("Forced CPU mode enabled")
    
    def reset_to_gpu(self) -> bool:
        """
        Attempt to reset to GPU mode.
        
        Returns:
            True if successfully reset to GPU
        """
        if not self._torch_available:
            return False
        
        self.clear_cache()
        
        if self.is_memory_safe():
            self._forced_cpu = False
            logger.info("Reset to GPU mode")
            return True
        
        logger.warning("Cannot reset to GPU, memory still high")
        return False
    
    def dynamic_batch_size(self, base_batch: int, 
                           memory_per_item_gb: float = 0.5) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            base_batch: Desired batch size
            memory_per_item_gb: Estimated memory per batch item
            
        Returns:
            Adjusted batch size
        """
        if not self._torch_available or self.should_use_cpu():
            return max(1, base_batch // 4)  # Reduce for CPU
        
        status = self.get_status()
        
        # Calculate how many items can fit in free memory (with safety margin)
        safe_free = status.free_gb * 0.7  # Use only 70% of free memory
        max_items = int(safe_free / memory_per_item_gb) if memory_per_item_gb > 0 else base_batch
        
        # Clamp to reasonable range
        adjusted = max(1, min(base_batch, max_items))
        
        if adjusted < base_batch:
            logger.debug(f"Reduced batch size from {base_batch} to {adjusted} due to memory")
        
        return adjusted
    
    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        
        if self._torch_available and self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
                self._torch.cuda.synchronize()
                logger.debug("GPU cache cleared")
            except Exception as e:
                logger.debug(f"Failed to clear GPU cache: {e}")
    
    def run_with_memory_management(self, 
                                    func: Callable[..., Any],
                                    *args,
                                    fallback_to_cpu: bool = True,
                                    **kwargs) -> Any:
        """
        Run a function with automatic memory management.
        
        Args:
            func: Function to run
            *args: Positional arguments
            fallback_to_cpu: If True, retry on CPU if OOM
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and fallback_to_cpu:
                logger.warning(f"OOM error, clearing cache and retrying: {e}")
                self.clear_cache()
                
                try:
                    return func(*args, **kwargs)
                except RuntimeError:
                    if fallback_to_cpu:
                        logger.warning("Retrying on CPU")
                        self.force_cpu_mode()
                        return func(*args, **kwargs)
            raise
    
    def get_device(self) -> str:
        """
        Get recommended device string.
        
        Returns:
            "cuda" or "cpu"
        """
        if self.should_use_cpu():
            return "cpu"
        return "cuda" if self._torch_available else "cpu"
    
    def print_status(self) -> None:
        """Print current memory status."""
        status = self.get_status()
        print(f"\n{'='*40}")
        print("GPU Memory Status")
        print(f"{'='*40}")
        print(f"Available: {status.available}")
        if status.available:
            print(f"Total: {status.total_gb:.2f} GB")
            print(f"Used: {status.used_gb:.2f} GB")
            print(f"Free: {status.free_gb:.2f} GB")
            print(f"Utilization: {status.utilization_percent:.1f}%")
        print(f"Forced CPU: {self._forced_cpu}")
        print(f"{'='*40}\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_manager: Optional[GPUMemoryManager] = None


def get_memory_manager() -> GPUMemoryManager:
    """Get global GPU memory manager."""
    global _manager
    if _manager is None:
        _manager = GPUMemoryManager()
    return _manager


def get_device() -> str:
    """Get recommended device ("cuda" or "cpu")."""
    return get_memory_manager().get_device()


def clear_gpu_cache() -> None:
    """Clear GPU cache."""
    get_memory_manager().clear_cache()


def check_memory() -> MemoryStatus:
    """Check current GPU memory status."""
    return get_memory_manager().get_status()


def dynamic_batch_size(base: int, memory_per_item: float = 0.5) -> int:
    """Get dynamic batch size based on available memory."""
    return get_memory_manager().dynamic_batch_size(base, memory_per_item)
