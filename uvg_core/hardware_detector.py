"""
UVG MAX Hardware Detector Module

Detects GPU/VRAM/CPU and optimizes settings accordingly.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware profile with detected capabilities."""
    
    # GPU
    gpu_available: bool = False
    gpu_type: str = "None"
    gpu_name: str = "Unknown"
    vram_gb: float = 0.0
    cuda_version: str = ""
    
    # CPU
    cpu_name: str = "Unknown"
    cpu_threads: int = 4
    
    # Memory
    ram_gb: float = 8.0
    
    # Platform
    os_name: str = "Unknown"
    python_version: str = ""
    
    def is_high_end(self) -> bool:
        """Check if hardware is high-end (A100, V100, etc.)."""
        high_end_gpus = ["A100", "V100", "A10", "RTX 4090", "RTX 4080", "RTX 3090"]
        return any(gpu in self.gpu_name for gpu in high_end_gpus)
    
    def is_colab_t4(self) -> bool:
        """Check if running on Colab T4."""
        return "T4" in self.gpu_name
    
    def is_cpu_only(self) -> bool:
        """Check if running CPU only."""
        return not self.gpu_available


class HardwareDetector:
    """
    Detects hardware capabilities and provides optimized settings.
    
    Supports:
    - NVIDIA GPUs via torch/nvidia-smi
    - CPU thread detection
    - RAM detection
    - Platform detection
    """
    
    def __init__(self):
        self._profile: Optional[HardwareProfile] = None
    
    def detect(self) -> HardwareProfile:
        """
        Detect hardware and return profile.
        
        Returns:
            HardwareProfile with detected capabilities
        """
        if self._profile is not None:
            return self._profile
        
        profile = HardwareProfile()
        
        # Platform info
        profile.os_name = platform.system()
        profile.python_version = platform.python_version()
        
        # CPU info
        profile.cpu_threads = os.cpu_count() or 4
        profile.cpu_name = self._detect_cpu_name()
        
        # RAM
        profile.ram_gb = self._detect_ram()
        
        # GPU
        gpu_info = self._detect_gpu()
        profile.gpu_available = gpu_info["available"]
        profile.gpu_type = gpu_info["type"]
        profile.gpu_name = gpu_info["name"]
        profile.vram_gb = gpu_info["vram_gb"]
        profile.cuda_version = gpu_info["cuda_version"]
        
        self._profile = profile
        logger.info(f"Hardware detected: {profile}")
        
        return profile
    
    def _detect_cpu_name(self) -> str:
        """Detect CPU name."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"CPU detection failed: {e}")
        
        return platform.processor() or "Unknown"
    
    def _detect_ram(self) -> float:
        """Detect total RAM in GB."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "OS", "get", "TotalVisibleMemorySize"],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    kb = int(lines[1].strip())
                    return round(kb / (1024 * 1024), 1)
            elif platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            kb = int(line.split()[1])
                            return round(kb / (1024 * 1024), 1)
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                bytes_ram = int(result.stdout.strip())
                return round(bytes_ram / (1024**3), 1)
        except Exception as e:
            logger.debug(f"RAM detection failed: {e}")
        
        return 8.0  # Default
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU capabilities."""
        result = {
            "available": False,
            "type": "None",
            "name": "None",
            "vram_gb": 0.0,
            "cuda_version": "",
        }
        
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                result["available"] = True
                result["type"] = "CUDA"
                result["name"] = torch.cuda.get_device_name(0)
                result["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                )
                result["cuda_version"] = torch.version.cuda or ""
                return result
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")
        
        # Try nvidia-smi
        try:
            nvidia_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if nvidia_result.returncode == 0:
                line = nvidia_result.stdout.strip().split('\n')[0]
                parts = line.split(',')
                if len(parts) >= 2:
                    result["available"] = True
                    result["type"] = "CUDA"
                    result["name"] = parts[0].strip()
                    result["vram_gb"] = round(float(parts[1].strip()) / 1024, 1)
                    return result
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")
        
        return result
    
    def get_optimal_settings(self, profile: Optional[HardwareProfile] = None) -> Dict[str, Any]:
        """
        Get optimal pipeline settings based on hardware.
        
        Args:
            profile: Hardware profile (detects if not provided)
            
        Returns:
            Dict with optimized settings
        """
        if profile is None:
            profile = self.detect()
        
        settings = {
            "use_cuda": profile.gpu_available,
            "use_nvenc": profile.gpu_available,
            "clip_batch_size": 4,
            "motion_quality": "high",
            "parallel_downloads": 6,
            "vfx_intensity": 1.0,
            "max_resolution": 1920,
            "frame_batch_size": 8,
        }
        
        # High-end GPU (A100, V100, etc.)
        if profile.is_high_end():
            settings.update({
                "clip_batch_size": 16,
                "motion_quality": "ultra",
                "parallel_downloads": 12,
                "frame_batch_size": 32,
            })
        
        # Colab T4 (15GB VRAM)
        elif profile.is_colab_t4():
            settings.update({
                "clip_batch_size": 8,
                "motion_quality": "high",
                "parallel_downloads": 6,
                "frame_batch_size": 16,
            })
        
        # Mid-range GPU (8-12GB VRAM)
        elif profile.gpu_available and 8 <= profile.vram_gb <= 12:
            settings.update({
                "clip_batch_size": 6,
                "motion_quality": "high",
                "parallel_downloads": 6,
                "frame_batch_size": 12,
            })
        
        # Low-end GPU (<8GB VRAM)
        elif profile.gpu_available and profile.vram_gb < 8:
            settings.update({
                "clip_batch_size": 2,
                "motion_quality": "medium",
                "parallel_downloads": 4,
                "frame_batch_size": 4,
            })
        
        # CPU only
        elif profile.is_cpu_only():
            settings.update({
                "use_cuda": False,
                "use_nvenc": False,
                "clip_batch_size": 1,
                "motion_quality": "low",
                "parallel_downloads": min(4, profile.cpu_threads),
                "frame_batch_size": 2,
                "max_resolution": 1080,
            })
        
        # Adjust for RAM
        if profile.ram_gb < 8:
            settings["clip_batch_size"] = min(2, settings["clip_batch_size"])
            settings["frame_batch_size"] = min(4, settings["frame_batch_size"])
            settings["parallel_downloads"] = min(2, settings["parallel_downloads"])
        
        return settings
    
    def print_summary(self) -> None:
        """Print hardware summary."""
        profile = self.detect()
        print(f"\n{'='*50}")
        print("UVG MAX Hardware Detection")
        print(f"{'='*50}")
        print(f"Platform: {profile.os_name} (Python {profile.python_version})")
        print(f"CPU: {profile.cpu_name}")
        print(f"CPU Threads: {profile.cpu_threads}")
        print(f"RAM: {profile.ram_gb} GB")
        print(f"GPU Available: {profile.gpu_available}")
        if profile.gpu_available:
            print(f"GPU Name: {profile.gpu_name}")
            print(f"VRAM: {profile.vram_gb} GB")
            print(f"CUDA Version: {profile.cuda_version}")
        print(f"{'='*50}\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_detector: Optional[HardwareDetector] = None


def get_hardware_profile() -> HardwareProfile:
    """Get cached hardware profile."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector.detect()


def get_optimal_settings() -> Dict[str, Any]:
    """Get optimal settings for current hardware."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector.get_optimal_settings()


def print_hardware_summary() -> None:
    """Print hardware summary."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    _detector.print_summary()
