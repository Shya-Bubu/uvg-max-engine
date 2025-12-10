# uvg_core/license_metadata.py
"""
License Metadata Tracking for UVG MAX.

Tracks license information for all downloaded media assets (clips, music, SFX).
Generates attribution text for YouTube descriptions and legal compliance.

Supported Providers:
- Pexels (free, requires attribution)
- Pixabay (free, no attribution required)
- Unsplash (free, attribution appreciated)
- Freesound (various licenses)

Usage:
    from uvg_core.license_metadata import LicenseTracker, ClipLicense
    
    tracker = LicenseTracker()
    
    # Add license when downloading clip
    license_info = ClipLicense(
        provider="pexels",
        asset_id="12345",
        license_type="pexels",
        author="John Doe",
        original_url="https://pexels.com/video/12345"
    )
    tracker.add_clip_license(scene_id=1, license_info=license_info)
    
    # Generate attribution file
    tracker.generate_attribution_file("uvg_output/attribution.txt")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# LICENSE TYPES
# =============================================================================

LICENSE_INFO = {
    "pexels": {
        "name": "Pexels License",
        "requires_attribution": False,  # Not required but appreciated
        "commercial_use": True,
        "modification": True,
        "attribution_template": "Video by {author} from Pexels",
        "license_url": "https://www.pexels.com/license/",
    },
    "pixabay": {
        "name": "Pixabay License",
        "requires_attribution": False,
        "commercial_use": True,
        "modification": True,
        "attribution_template": "Video from Pixabay",
        "license_url": "https://pixabay.com/service/license/",
    },
    "unsplash": {
        "name": "Unsplash License",
        "requires_attribution": False,  # Not required but appreciated
        "commercial_use": True,
        "modification": True,
        "attribution_template": "Photo by {author} on Unsplash",
        "license_url": "https://unsplash.com/license",
    },
    "freesound_cc0": {
        "name": "Creative Commons Zero",
        "requires_attribution": False,
        "commercial_use": True,
        "modification": True,
        "attribution_template": "Sound from Freesound.org (CC0)",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    },
    "freesound_ccby": {
        "name": "Creative Commons Attribution",
        "requires_attribution": True,
        "commercial_use": True,
        "modification": True,
        "attribution_template": "Sound by {author} from Freesound.org (CC BY)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    },
    "freesound_ccbync": {
        "name": "Creative Commons Attribution-NonCommercial",
        "requires_attribution": True,
        "commercial_use": False,
        "modification": True,
        "attribution_template": "Sound by {author} from Freesound.org (CC BY-NC)",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    },
    "custom": {
        "name": "Custom License",
        "requires_attribution": True,
        "commercial_use": False,  # Assume restrictive
        "modification": False,
        "attribution_template": "{author}",
        "license_url": "",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClipLicense:
    """License information for a single clip/asset."""
    provider: str  # pexels, pixabay, unsplash, freesound
    asset_id: str  # Provider's ID for the asset
    license_type: str  # License key from LICENSE_INFO
    author: str = ""  # Creator's name
    original_url: str = ""  # Link to original asset
    downloaded_path: str = ""  # Local path where downloaded
    download_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_attribution(self) -> str:
        """Generate attribution text for this asset."""
        license_info = LICENSE_INFO.get(self.license_type, LICENSE_INFO["custom"])
        template = license_info["attribution_template"]
        return template.format(author=self.author or "Unknown")
    
    def requires_attribution(self) -> bool:
        """Check if this asset requires attribution."""
        license_info = LICENSE_INFO.get(self.license_type, LICENSE_INFO["custom"])
        return license_info["requires_attribution"]
    
    def allows_commercial(self) -> bool:
        """Check if this asset allows commercial use."""
        license_info = LICENSE_INFO.get(self.license_type, LICENSE_INFO["custom"])
        return license_info["commercial_use"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipLicense":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MusicLicense:
    """License information for music track."""
    provider: str
    asset_id: str
    license_type: str
    title: str = ""
    artist: str = ""
    original_url: str = ""
    downloaded_path: str = ""
    download_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_attribution(self) -> str:
        """Generate attribution text for music."""
        if self.title and self.artist:
            return f'"{self.title}" by {self.artist}'
        elif self.title:
            return f'"{self.title}"'
        elif self.artist:
            return f"Music by {self.artist}"
        return "Background music"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# LICENSE TRACKER
# =============================================================================

class LicenseTracker:
    """
    Tracks all license information for a video project.
    
    Attributes:
        clip_licenses: Dict mapping scene_id -> ClipLicense
        music_license: Optional MusicLicense for background music
        sfx_licenses: List of ClipLicense for sound effects
    """
    
    def __init__(self):
        self.clip_licenses: Dict[int, ClipLicense] = {}
        self.music_license: Optional[MusicLicense] = None
        self.sfx_licenses: List[ClipLicense] = []
        self.project_title: str = ""
    
    def add_clip_license(self, scene_id: int, license_info: ClipLicense):
        """Add license info for a scene's clip."""
        self.clip_licenses[scene_id] = license_info
        logger.debug(f"Added license for scene {scene_id}: {license_info.provider}")
    
    def set_music_license(self, license_info: MusicLicense):
        """Set license info for background music."""
        self.music_license = license_info
        logger.debug(f"Set music license: {license_info.provider}")
    
    def add_sfx_license(self, license_info: ClipLicense):
        """Add license info for a sound effect."""
        self.sfx_licenses.append(license_info)
    
    def has_attribution_required(self) -> bool:
        """Check if any asset requires attribution."""
        for license_info in self.clip_licenses.values():
            if license_info.requires_attribution():
                return True
        
        if self.music_license:
            music_info = LICENSE_INFO.get(self.music_license.license_type, {})
            if music_info.get("requires_attribution", True):
                return True
        
        for sfx in self.sfx_licenses:
            if sfx.requires_attribution():
                return True
        
        return False
    
    def has_commercial_restriction(self) -> bool:
        """Check if any asset restricts commercial use."""
        for license_info in self.clip_licenses.values():
            if not license_info.allows_commercial():
                return True
        
        if self.music_license:
            music_info = LICENSE_INFO.get(self.music_license.license_type, {})
            if not music_info.get("commercial_use", False):
                return True
        
        return False
    
    def generate_attribution_text(self) -> str:
        """Generate full attribution text for video description."""
        lines = []
        
        lines.append("=" * 50)
        lines.append("ATTRIBUTION / CREDITS")
        lines.append("=" * 50)
        lines.append("")
        
        # Video clips
        if self.clip_licenses:
            lines.append("[VIDEO CLIPS]")
            for scene_id, license_info in sorted(self.clip_licenses.items()):
                attribution = license_info.get_attribution()
                if license_info.original_url:
                    lines.append(f"  Scene {scene_id}: {attribution}")
                    lines.append(f"    -> {license_info.original_url}")
                else:
                    lines.append(f"  Scene {scene_id}: {attribution}")
            lines.append("")
        
        # Music
        if self.music_license:
            lines.append("[MUSIC]")
            lines.append(f"  {self.music_license.get_attribution()}")
            if self.music_license.original_url:
                lines.append(f"  -> {self.music_license.original_url}")
            lines.append("")
        
        # Sound effects
        if self.sfx_licenses:
            lines.append("[SOUND EFFECTS]")
            seen = set()
            for sfx in self.sfx_licenses:
                key = (sfx.provider, sfx.asset_id)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  {sfx.get_attribution()}")
            lines.append("")
        
        # License summary
        lines.append("[LICENSES]")
        licenses_used = set()
        for license_info in self.clip_licenses.values():
            licenses_used.add(license_info.license_type)
        if self.music_license:
            licenses_used.add(self.music_license.license_type)
        
        for license_type in sorted(licenses_used):
            info = LICENSE_INFO.get(license_type, {})
            name = info.get("name", license_type)
            url = info.get("license_url", "")
            lines.append(f"  • {name}")
            if url:
                lines.append(f"    {url}")
        
        lines.append("")
        lines.append("Generated by UVG MAX")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        return "\n".join(lines)
    
    def generate_attribution_file(self, output_path: str):
        """Save attribution to a text file."""
        text = self.generate_attribution_text()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info(f"Attribution saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all license data as dictionary."""
        return {
            "project_title": self.project_title,
            "clip_licenses": {
                str(k): v.to_dict() for k, v in self.clip_licenses.items()
            },
            "music_license": self.music_license.to_dict() if self.music_license else None,
            "sfx_licenses": [s.to_dict() for s in self.sfx_licenses],
            "has_attribution_required": self.has_attribution_required(),
            "has_commercial_restriction": self.has_commercial_restriction(),
        }
    
    def save_json(self, output_path: str):
        """Save license data as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"License data saved to: {output_path}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_pexels_license(
    asset_id: str,
    author: str = "",
    original_url: str = "",
    downloaded_path: str = ""
) -> ClipLicense:
    """Create license for Pexels asset."""
    return ClipLicense(
        provider="pexels",
        asset_id=asset_id,
        license_type="pexels",
        author=author,
        original_url=original_url or f"https://www.pexels.com/video/{asset_id}/",
        downloaded_path=downloaded_path,
    )


def create_pixabay_license(
    asset_id: str,
    author: str = "",
    original_url: str = "",
    downloaded_path: str = ""
) -> ClipLicense:
    """Create license for Pixabay asset."""
    return ClipLicense(
        provider="pixabay",
        asset_id=asset_id,
        license_type="pixabay",
        author=author,
        original_url=original_url or f"https://pixabay.com/videos/id-{asset_id}/",
        downloaded_path=downloaded_path,
    )


def create_unsplash_license(
    asset_id: str,
    author: str = "",
    original_url: str = "",
    downloaded_path: str = ""
) -> ClipLicense:
    """Create license for Unsplash asset."""
    return ClipLicense(
        provider="unsplash",
        asset_id=asset_id,
        license_type="unsplash",
        author=author,
        original_url=original_url or f"https://unsplash.com/photos/{asset_id}",
        downloaded_path=downloaded_path,
    )


# =============================================================================
# GLOBAL TRACKER
# =============================================================================

_tracker: Optional[LicenseTracker] = None


def get_license_tracker() -> LicenseTracker:
    """Get global license tracker."""
    global _tracker
    if _tracker is None:
        _tracker = LicenseTracker()
    return _tracker


def reset_license_tracker():
    """Reset global license tracker (for new video)."""
    global _tracker
    _tracker = LicenseTracker()
