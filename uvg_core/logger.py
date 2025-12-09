# uvg_core/logger.py
"""
UVG MAX Logging System.

Provides dual-level logging:
- INFO: User-friendly progress messages
- DEBUG: Developer analysis logs
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Create loggers
_user_logger = logging.getLogger("uvg.user")
_debug_logger = logging.getLogger("uvg.debug")

# Track if setup has been done
_setup_done = False


def setup_logging(
    debug_mode: bool = False,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Set up logging system.
    
    Args:
        debug_mode: Enable debug-level logging
        log_file: Path to debug log file (default: uvg_debug.log)
        console_output: Enable console output
    """
    global _setup_done
    
    if _setup_done:
        return
    
    # User logger (always INFO level, clean format)
    _user_logger.setLevel(logging.INFO)
    _user_logger.propagate = False
    
    if console_output:
        user_handler = logging.StreamHandler(sys.stdout)
        user_handler.setFormatter(logging.Formatter("%(message)s"))
        _user_logger.addHandler(user_handler)
    
    # Debug logger
    _debug_logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
    _debug_logger.propagate = False
    
    if debug_mode:
        log_path = log_file or "uvg_debug.log"
        try:
            debug_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            debug_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S"
            ))
            _debug_logger.addHandler(debug_handler)
        except Exception as e:
            print(f"Warning: Could not create debug log file: {e}")
    
    _setup_done = True


def log_info(msg: str) -> None:
    """
    Log user-friendly info message.
    
    Args:
        msg: Message to log (will be prefixed with ✅)
    """
    if not _setup_done:
        setup_logging()
    _user_logger.info(f"✅ {msg}")


def log_step(msg: str) -> None:
    """
    Log a pipeline step.
    
    Args:
        msg: Step description (will be prefixed with 🔄)
    """
    if not _setup_done:
        setup_logging()
    _user_logger.info(f"🔄 {msg}")


def log_progress(current: int, total: int, msg: str = "") -> None:
    """
    Log progress update.
    
    Args:
        current: Current item number
        total: Total items
        msg: Optional message
    """
    if not _setup_done:
        setup_logging()
    pct = int(current / total * 100) if total > 0 else 0
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    _user_logger.info(f"  [{bar}] {pct}% {msg}")


def log_success(msg: str) -> None:
    """
    Log success message.
    
    Args:
        msg: Success message (will be prefixed with 🎉)
    """
    if not _setup_done:
        setup_logging()
    _user_logger.info(f"🎉 {msg}")


def log_warn(msg: str) -> None:
    """
    Log warning message.
    
    Args:
        msg: Warning message (will be prefixed with ⚠️)
    """
    if not _setup_done:
        setup_logging()
    _user_logger.warning(f"⚠️ {msg}")


def log_error(msg: str) -> None:
    """
    Log error message.
    
    Args:
        msg: Error message (will be prefixed with ❌)
    """
    if not _setup_done:
        setup_logging()
    _user_logger.error(f"❌ {msg}")


def log_debug(msg: str) -> None:
    """
    Log debug message (only visible in debug mode).
    
    Args:
        msg: Debug message for developers
    """
    if not _setup_done:
        setup_logging()
    _debug_logger.debug(msg)


def log_debug_data(label: str, data: dict) -> None:
    """
    Log structured debug data.
    
    Args:
        label: Data label
        data: Dictionary of debug values
    """
    if not _setup_done:
        setup_logging()
    
    lines = [f"[{label}]"]
    for k, v in data.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    _debug_logger.debug("\n".join(lines))


def log_timing(operation: str, duration_sec: float) -> None:
    """
    Log operation timing.
    
    Args:
        operation: Operation name
        duration_sec: Duration in seconds
    """
    if not _setup_done:
        setup_logging()
    _debug_logger.debug(f"TIMING: {operation} took {duration_sec:.2f}s")


class LogContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, log_user: bool = False):
        self.operation = operation
        self.log_user = log_user
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        if self.log_user:
            log_step(f"{self.operation}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        log_timing(self.operation, duration)
        if self.log_user and exc_type is None:
            log_info(f"{self.operation} complete ({duration:.1f}s)")
        return False


# Auto-setup with defaults
def _auto_setup():
    """Auto-setup based on environment."""
    debug = os.environ.get("UVG_DEBUG", "").lower() in ("1", "true", "yes")
    setup_logging(debug_mode=debug)


# Run auto-setup on import
_auto_setup()
