# uvg_selector/cache.py
"""
MD5-based file caching for embeddings and metadata.
"""

import os
import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path("uvg_selector_cache")
CACHE_DIR.mkdir(exist_ok=True)


def make_key(*args) -> str:
    """
    Create MD5 hash key from arguments.
    
    Args:
        *args: Values to hash (strings, lists, tuples)
        
    Returns:
        MD5 hex digest string
    """
    m = hashlib.md5()
    for a in args:
        if isinstance(a, (list, tuple)):
            a = "|".join(map(str, a))
        m.update(str(a).encode('utf-8'))
    return m.hexdigest()


def get_cache(key: str) -> dict | None:
    """
    Retrieve cached value by key.
    
    Args:
        key: Cache key (MD5 hash)
        
    Returns:
        Cached dict or None if not found
    """
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f"Cache read failed for {key}: {e}")
        return None


def set_cache(key: str, value: dict) -> None:
    """
    Store value in cache.
    
    Args:
        key: Cache key (MD5 hash)
        value: Dict to cache (must be JSON serializable)
    """
    try:
        p = CACHE_DIR / f"{key}.json"
        p.write_text(json.dumps(value, indent=2), encoding='utf-8')
        logger.debug(f"Cached: {key}")
    except Exception as e:
        logger.warning(f"Cache write failed for {key}: {e}")


def clear_cache() -> int:
    """
    Clear all cached files.
    
    Returns:
        Number of files deleted
    """
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    logger.info(f"Cleared {count} cache files")
    return count


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        Dict with count and total_size_mb
    """
    files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }
