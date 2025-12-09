#!/usr/bin/env python
"""
UVG MAX Model Downloader.

Downloads required AI models:
- CLIP ViT-B/32 (ONNX)
- Aesthetic predictor weights
- YOLOv8n (optional)
"""

import os
import sys
import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Model URLs and hashes
MODELS = {
    "clip-visual": {
        "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/model.onnx",
        "path": "models/clip-vit-b-32-visual.onnx",
        "size_mb": 350,
        "description": "CLIP ViT-B/32 visual encoder (ONNX)",
        "optional": False,
    },
    "aesthetic": {
        "url": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
        "path": "models/aesthetic_weights.pth",
        "size_mb": 3,
        "description": "LAION aesthetic predictor weights",
        "optional": False,
    },
    "yolo": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "path": "models/yolov8n.pt",
        "size_mb": 6,
        "description": "YOLOv8n object detector",
        "optional": True,
    },
}


def get_project_root() -> Path:
    """Get project root directory."""
    # Try to find project root by looking for uvg_core
    current = Path(__file__).parent
    for _ in range(5):  # Max 5 levels up
        if (current / "uvg_core").exists():
            return current
        current = current.parent
    return Path.cwd()


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    """
    Download a file with progress.
    
    Args:
        url: Download URL
        dest: Destination path
        desc: Description for logging
        
    Returns:
        True if successful
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"⬇️  Downloading {desc}...")
        logger.info(f"   URL: {url}")
        logger.info(f"   Destination: {dest}")
        
        # Simple download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(block_num * block_size * 100 / total_size))
                print(f"\r   Progress: {pct}%", end="", flush=True)
        
        urlretrieve(url, str(dest), reporthook=report_progress)
        print()  # New line after progress
        
        logger.info(f"✅ Downloaded: {dest}")
        return True
        
    except URLError as e:
        logger.error(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def check_model_exists(model_key: str, root: Path) -> bool:
    """Check if model file exists."""
    model = MODELS.get(model_key, {})
    path = root / model.get("path", "")
    return path.exists()


def download_model(model_key: str, root: Path, force: bool = False) -> bool:
    """
    Download a specific model.
    
    Args:
        model_key: Model key from MODELS dict
        root: Project root path
        force: Force re-download
        
    Returns:
        True if successful
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    model = MODELS[model_key]
    dest = root / model["path"]
    
    if dest.exists() and not force:
        logger.info(f"✓ {model['description']} already exists")
        return True
    
    return download_file(model["url"], dest, model["description"])


def download_all_models(root: Path, include_optional: bool = False, force: bool = False) -> dict:
    """
    Download all required models.
    
    Args:
        root: Project root path
        include_optional: Include optional models
        force: Force re-download
        
    Returns:
        Dict of model -> success
    """
    results = {}
    
    for key, model in MODELS.items():
        if model.get("optional") and not include_optional:
            logger.info(f"⏭️  Skipping optional: {model['description']}")
            continue
        
        results[key] = download_model(key, root, force)
    
    return results


def check_all_models(root: Path) -> dict:
    """
    Check which models are available.
    
    Args:
        root: Project root path
        
    Returns:
        Dict of model -> exists
    """
    status = {}
    
    for key, model in MODELS.items():
        path = root / model["path"]
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0
        
        status[key] = {
            "exists": exists,
            "path": str(path),
            "size_mb": round(size_mb, 1),
            "description": model["description"],
            "optional": model.get("optional", False),
        }
    
    return status


def print_status(root: Path):
    """Print model status."""
    status = check_all_models(root)
    
    print("\n📦 Model Status:")
    print("-" * 60)
    
    for key, info in status.items():
        icon = "✅" if info["exists"] else "❌"
        opt = " (optional)" if info["optional"] else ""
        size = f"({info['size_mb']} MB)" if info["exists"] else f"(need ~{MODELS[key]['size_mb']} MB)"
        
        print(f"  {icon} {info['description']}{opt}")
        print(f"     {info['path']} {size}")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download UVG MAX models")
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all models including optional"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        help="Download specific model"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show model status only"
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Get root directory
    root = Path(args.root) if args.root else get_project_root()
    logger.info(f"📁 Project root: {root}")
    
    # Status only
    if args.status:
        print_status(root)
        return 0
    
    # Download specific model
    if args.model:
        success = download_model(args.model, root, args.force)
        return 0 if success else 1
    
    # Download all
    if args.all:
        results = download_all_models(root, include_optional=True, force=args.force)
    else:
        results = download_all_models(root, include_optional=False, force=args.force)
    
    # Print results
    print_status(root)
    
    success_count = sum(1 for v in results.values() if v)
    total = len(results)
    
    if success_count == total:
        logger.info(f"\n✅ All {total} models ready!")
        return 0
    else:
        logger.warning(f"\n⚠️  {success_count}/{total} models downloaded")
        return 1


if __name__ == "__main__":
    sys.exit(main())
