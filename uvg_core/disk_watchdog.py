"""
UVG MAX Disk Watchdog Module

Strict cleanup enforcement, threshold monitoring, and automatic deletion.
Prevents disk overflow during video generation.
"""

import os
import shutil
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DiskStatus:
    """Current disk status."""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    project_size_gb: float = 0.0
    utilization_percent: float = 0.0


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    files_deleted: int = 0
    bytes_freed: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def gb_freed(self) -> float:
        return self.bytes_freed / (1024**3)


class DiskWatchdog:
    """
    Monitors disk usage and enforces strict cleanup.
    
    Features:
    - Delete frames immediately after scoring
    - Delete unselected candidates after selection
    - Delete original after trim validation
    - Auto-cleanup when threshold exceeded
    - Cleanup logging to metrics
    """
    
    def __init__(self, 
                 base_dir: Path,
                 threshold_gb: int = 70,
                 cleanup_target_gb: int = 50,
                 max_project_gb: int = 80):
        """
        Initialize disk watchdog.
        
        Args:
            base_dir: Base project directory to monitor
            threshold_gb: Trigger cleanup when disk usage exceeds this
            cleanup_target_gb: Target to clean down to
            max_project_gb: Maximum project size allowed
        """
        self.base_dir = Path(base_dir)
        self.threshold_gb = threshold_gb
        self.cleanup_target_gb = cleanup_target_gb
        self.max_project_gb = max_project_gb
        
        # Track files scheduled for deletion
        self._pending_deletion: Set[Path] = set()
        
        # Cleanup history
        self._cleanup_history: List[Dict[str, Any]] = []
        
        # Ensure base dir exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_disk_status(self) -> DiskStatus:
        """
        Get current disk status.
        
        Returns:
            DiskStatus with usage information
        """
        status = DiskStatus()
        
        try:
            disk_usage = shutil.disk_usage(self.base_dir)
            status.total_gb = disk_usage.total / (1024**3)
            status.used_gb = disk_usage.used / (1024**3)
            status.free_gb = disk_usage.free / (1024**3)
            status.utilization_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Calculate project size
            status.project_size_gb = self._get_directory_size(self.base_dir) / (1024**3)
            
        except Exception as e:
            logger.error(f"Failed to get disk status: {e}")
        
        return status
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of a directory in bytes."""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except (OSError, PermissionError):
                        pass
        except Exception as e:
            logger.debug(f"Error getting directory size: {e}")
        return total
    
    def needs_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        status = self.get_disk_status()
        return (status.free_gb < self.threshold_gb or 
                status.project_size_gb > self.max_project_gb)
    
    def delete_file(self, path: Path, reason: str = "") -> bool:
        """
        Delete a single file immediately.
        
        Args:
            path: Path to file
            reason: Reason for deletion (for logging)
            
        Returns:
            True if deleted successfully
        """
        path = Path(path)
        if not path.exists():
            return True
        
        try:
            size = path.stat().st_size if path.is_file() else 0
            
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            
            logger.debug(f"Deleted {path} ({size} bytes): {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False
    
    def delete_files(self, paths: List[Path], reason: str = "") -> CleanupResult:
        """
        Delete multiple files.
        
        Args:
            paths: List of paths to delete
            reason: Reason for deletion
            
        Returns:
            CleanupResult with statistics
        """
        result = CleanupResult()
        
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            
            try:
                size = path.stat().st_size if path.is_file() else self._get_directory_size(path)
                
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                
                result.files_deleted += 1
                result.bytes_freed += size
                
            except Exception as e:
                result.errors.append(f"{path}: {e}")
        
        if result.files_deleted > 0:
            logger.info(f"Deleted {result.files_deleted} files, freed {result.gb_freed:.2f} GB: {reason}")
        
        return result
    
    def cleanup_frames_after_scoring(self, frames_dir: Path) -> CleanupResult:
        """
        Delete frame samples immediately after scoring.
        
        Args:
            frames_dir: Directory containing frame samples
            
        Returns:
            CleanupResult
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            return CleanupResult()
        
        files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        result = self.delete_files(files, "frames after scoring")
        
        # Also remove empty directory
        if frames_dir.exists() and not any(frames_dir.iterdir()):
            try:
                frames_dir.rmdir()
            except Exception:
                pass
        
        return result
    
    def cleanup_unselected_candidates(self, 
                                       all_candidates: List[Path],
                                       selected: List[Path]) -> CleanupResult:
        """
        Delete unselected clip candidates.
        
        Args:
            all_candidates: All candidate clip paths
            selected: Selected clip paths to keep
            
        Returns:
            CleanupResult
        """
        selected_set = {Path(p).resolve() for p in selected}
        to_delete = [
            Path(p) for p in all_candidates 
            if Path(p).resolve() not in selected_set
        ]
        
        return self.delete_files(to_delete, "unselected candidates")
    
    def cleanup_original_after_trim(self, 
                                     original: Path, 
                                     trimmed: Path,
                                     validate: bool = True) -> bool:
        """
        Delete original clip after trimmed version is validated.
        
        Args:
            original: Original clip path
            trimmed: Trimmed clip path
            validate: Whether to validate trimmed exists first
            
        Returns:
            True if original was deleted
        """
        original = Path(original)
        trimmed = Path(trimmed)
        
        if validate and not trimmed.exists():
            logger.warning(f"Cannot delete original, trimmed not found: {trimmed}")
            return False
        
        if validate:
            # Basic validation - check trimmed has reasonable size
            try:
                if trimmed.stat().st_size < 1000:  # Less than 1KB
                    logger.warning(f"Trimmed file too small, keeping original")
                    return False
            except Exception:
                return False
        
        return self.delete_file(original, "original after validated trim")
    
    def cleanup_temp_directories(self) -> CleanupResult:
        """
        Clean up temporary directories.
        
        Returns:
            CleanupResult
        """
        temp_patterns = [
            "tmp_*",
            "temp_*", 
            "*.tmp",
            "__pycache__",
            ".cache",
        ]
        
        to_delete = []
        for pattern in temp_patterns:
            to_delete.extend(self.base_dir.rglob(pattern))
        
        return self.delete_files(to_delete, "temporary directories")
    
    def auto_cleanup(self) -> CleanupResult:
        """
        Automatic cleanup to meet target.
        
        Deletes oldest temp files first until target is reached.
        
        Returns:
            CleanupResult
        """
        result = CleanupResult()
        status = self.get_disk_status()
        
        if status.free_gb >= self.cleanup_target_gb:
            return result
        
        bytes_to_free = (self.cleanup_target_gb - status.free_gb) * (1024**3)
        bytes_freed = 0
        
        logger.info(f"Auto-cleanup starting, need to free {bytes_to_free / (1024**3):.2f} GB")
        
        # Priority order for deletion
        cleanup_dirs = [
            self.base_dir / "frames",
            self.base_dir / "cache",
            self.base_dir / "clips",
            self.base_dir / "trimmed",
        ]
        
        for cleanup_dir in cleanup_dirs:
            if bytes_freed >= bytes_to_free:
                break
            
            if not cleanup_dir.exists():
                continue
            
            # Get files sorted by modification time (oldest first)
            try:
                files = sorted(
                    cleanup_dir.rglob('*'),
                    key=lambda f: f.stat().st_mtime if f.is_file() else 0
                )
            except Exception:
                continue
            
            for file_path in files:
                if bytes_freed >= bytes_to_free:
                    break
                
                if not file_path.is_file():
                    continue
                
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    bytes_freed += size
                    result.files_deleted += 1
                except Exception as e:
                    result.errors.append(str(e))
        
        result.bytes_freed = bytes_freed
        
        # Log cleanup
        self._cleanup_history.append({
            "timestamp": datetime.now().isoformat(),
            "files_deleted": result.files_deleted,
            "bytes_freed": result.bytes_freed,
            "trigger": "auto",
        })
        
        logger.info(f"Auto-cleanup complete: deleted {result.files_deleted} files, "
                   f"freed {result.gb_freed:.2f} GB")
        
        return result
    
    def schedule_deletion(self, path: Path) -> None:
        """
        Schedule a file for deletion (deleted on next flush).
        
        Args:
            path: Path to schedule for deletion
        """
        self._pending_deletion.add(Path(path))
    
    def flush_pending(self) -> CleanupResult:
        """
        Delete all pending files.
        
        Returns:
            CleanupResult
        """
        paths = list(self._pending_deletion)
        self._pending_deletion.clear()
        return self.delete_files(paths, "pending deletion flush")
    
    def run_cleanup_cycle(self) -> CleanupResult:
        """
        Run a full cleanup cycle.
        
        1. Flush pending deletions
        2. Clean temp directories
        3. Auto-cleanup if needed
        
        Returns:
            Combined CleanupResult
        """
        total_result = CleanupResult()
        
        # 1. Flush pending
        result = self.flush_pending()
        total_result.files_deleted += result.files_deleted
        total_result.bytes_freed += result.bytes_freed
        total_result.errors.extend(result.errors)
        
        # 2. Clean temps
        result = self.cleanup_temp_directories()
        total_result.files_deleted += result.files_deleted
        total_result.bytes_freed += result.bytes_freed
        total_result.errors.extend(result.errors)
        
        # 3. Auto-cleanup if needed
        if self.needs_cleanup():
            result = self.auto_cleanup()
            total_result.files_deleted += result.files_deleted
            total_result.bytes_freed += result.bytes_freed
            total_result.errors.extend(result.errors)
        
        return total_result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cleanup metrics for logging.
        
        Returns:
            Dictionary with cleanup statistics
        """
        status = self.get_disk_status()
        
        total_freed = sum(h.get("bytes_freed", 0) for h in self._cleanup_history)
        total_deleted = sum(h.get("files_deleted", 0) for h in self._cleanup_history)
        
        return {
            "disk_status": {
                "total_gb": round(status.total_gb, 2),
                "used_gb": round(status.used_gb, 2),
                "free_gb": round(status.free_gb, 2),
                "project_size_gb": round(status.project_size_gb, 2),
            },
            "cleanup_stats": {
                "total_files_deleted": total_deleted,
                "total_bytes_freed": total_freed,
                "total_gb_freed": round(total_freed / (1024**3), 2),
                "cleanup_count": len(self._cleanup_history),
            },
            "thresholds": {
                "threshold_gb": self.threshold_gb,
                "cleanup_target_gb": self.cleanup_target_gb,
                "max_project_gb": self.max_project_gb,
            },
        }
    
    def save_metrics(self, path: Optional[Path] = None) -> None:
        """Save cleanup metrics to JSON file."""
        if path is None:
            path = self.base_dir / "cleanup_metrics.json"
        
        metrics = self.get_metrics()
        metrics["history"] = self._cleanup_history
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def print_status(self) -> None:
        """Print current disk status."""
        status = self.get_disk_status()
        print(f"\n{'='*50}")
        print("Disk Watchdog Status")
        print(f"{'='*50}")
        print(f"Total: {status.total_gb:.1f} GB")
        print(f"Used: {status.used_gb:.1f} GB ({status.utilization_percent:.1f}%)")
        print(f"Free: {status.free_gb:.1f} GB")
        print(f"Project Size: {status.project_size_gb:.2f} GB")
        print(f"Threshold: {self.threshold_gb} GB free")
        print(f"Needs Cleanup: {self.needs_cleanup()}")
        print(f"Pending Deletions: {len(self._pending_deletion)}")
        print(f"{'='*50}\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_watchdog: Optional[DiskWatchdog] = None


def get_watchdog(base_dir: Optional[Path] = None) -> DiskWatchdog:
    """Get global disk watchdog instance."""
    global _watchdog
    if _watchdog is None:
        if base_dir is None:
            base_dir = Path("./uvg_output")
        _watchdog = DiskWatchdog(base_dir)
    return _watchdog


def cleanup_frames(frames_dir: Path) -> CleanupResult:
    """Convenience function to cleanup frames after scoring."""
    return get_watchdog().cleanup_frames_after_scoring(frames_dir)


def cleanup_unselected(all_clips: List[Path], selected: List[Path]) -> CleanupResult:
    """Convenience function to cleanup unselected candidates."""
    return get_watchdog().cleanup_unselected_candidates(all_clips, selected)


def cleanup_original(original: Path, trimmed: Path) -> bool:
    """Convenience function to cleanup original after trim."""
    return get_watchdog().cleanup_original_after_trim(original, trimmed)


def run_cleanup() -> CleanupResult:
    """Run a full cleanup cycle."""
    return get_watchdog().run_cleanup_cycle()
