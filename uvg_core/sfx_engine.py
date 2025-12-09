# uvg_core/sfx_engine.py
"""
SFX Engine for UVG MAX.

Automatic sound effects:
- Trigger-based selection
- Freesound API integration
- Local library fallback
"""

import logging
import os
import hashlib
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import requests
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False


@dataclass
class SFXResult:
    """SFX operation result."""
    success: bool
    output_path: str
    sfx_count: int
    source: str  # "local" or "freesound"
    error: str = ""


# SFX trigger mappings
SFX_TRIGGERS = {
    # Transitions
    "transition:fade": "fade_soft",
    "transition:whip_pan": "whoosh_fast",
    "transition:zoom": "zoom_swoosh",
    "transition:flash": "flash_impact",
    
    # Emotions
    "emotion:tense": "tension_riser",
    "emotion:happy": "bright_ding",
    "emotion:sad": "sad_tone",
    "emotion:epic": "epic_hit",
    "emotion:peaceful": None,  # No SFX
    
    # Scene events
    "scene:start": "subtle_whoosh",
    "scene:climax": "impact_deep",
    "scene:end": "fade_out_tone",
    
    # Text events
    "text:appear": "pop_soft",
    "text:highlight": "click_soft",
}

# Local SFX library (embedded simple sounds)
LOCAL_SFX = {
    "whoosh_fast": "whoosh_fast.wav",
    "whoosh_slow": "whoosh_slow.wav",
    "impact_deep": "impact_deep.wav",
    "tension_riser": "tension_riser.wav",
    "bright_ding": "bright_ding.wav",
    "pop_soft": "pop_soft.wav",
    "fade_soft": "fade_soft.wav",
    "click_soft": "click_soft.wav",
}


class SFXEngine:
    """
    Automatic sound effects engine.
    
    Features:
    - Trigger-based SFX selection
    - Freesound API integration
    - Local library fallback
    - Caching
    """
    
    FREESOUND_API = "https://freesound.org/apiv2"
    
    def __init__(
        self,
        sfx_dir: Path = None,
        cache_dir: Path = None,
        freesound_key: str = None
    ):
        """
        Initialize SFX engine.
        
        Args:
            sfx_dir: Local SFX directory
            cache_dir: Cache directory
            freesound_key: Freesound API key
        """
        self.sfx_dir = Path(sfx_dir) if sfx_dir else Path("sfx")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("sfx_cache")
        self.freesound_key = freesound_key or os.getenv("FREESOUND_API_KEY", "")
        
        self.sfx_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sfx(self, trigger: str) -> Optional[str]:
        """
        Get SFX file for trigger.
        
        Args:
            trigger: Trigger string (e.g., "transition:whip_pan")
            
        Returns:
            Path to SFX file or None
        """
        # Look up trigger
        sfx_name = SFX_TRIGGERS.get(trigger)
        if sfx_name is None:
            return None
        
        # Check local library first
        local_path = self._get_local_sfx(sfx_name)
        if local_path:
            return local_path
        
        # Try Freesound
        if self.freesound_key and HAVE_REQUESTS:
            freesound_path = self._get_freesound_sfx(sfx_name)
            if freesound_path:
                return freesound_path
        
        return None
    
    def _get_local_sfx(self, sfx_name: str) -> Optional[str]:
        """Get SFX from local library."""
        # Check direct file
        if sfx_name in LOCAL_SFX:
            path = self.sfx_dir / LOCAL_SFX[sfx_name]
            if path.exists():
                return str(path)
        
        # Check by name pattern
        for ext in [".wav", ".mp3", ".ogg"]:
            path = self.sfx_dir / f"{sfx_name}{ext}"
            if path.exists():
                return str(path)
        
        return None
    
    def _get_freesound_sfx(self, query: str) -> Optional[str]:
        """Download SFX from Freesound API."""
        cache_key = hashlib.md5(query.encode()).hexdigest()[:8]
        cache_path = self.cache_dir / f"{cache_key}.wav"
        
        # Check cache
        if cache_path.exists():
            return str(cache_path)
        
        try:
            # Search Freesound
            search_url = f"{self.FREESOUND_API}/search/text/"
            params = {
                "query": query,
                "token": self.freesound_key,
                "fields": "id,name,previews",
                "page_size": 1,
                "filter": "duration:[0.5 TO 5]"  # Short sounds only
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            results = response.json().get("results", [])
            if not results:
                return None
            
            # Get preview URL
            preview_url = results[0].get("previews", {}).get("preview-hq-mp3")
            if not preview_url:
                return None
            
            # Download preview
            audio_response = requests.get(preview_url, timeout=30)
            if audio_response.status_code == 200:
                # Save as MP3, convert to WAV
                mp3_path = self.cache_dir / f"{cache_key}.mp3"
                with open(mp3_path, 'wb') as f:
                    f.write(audio_response.content)
                
                # Convert to WAV
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(mp3_path),
                    "-ar", "44100", "-ac", "2",
                    str(cache_path)
                ], capture_output=True, timeout=30)
                
                if cache_path.exists():
                    return str(cache_path)
            
        except Exception as e:
            logger.debug(f"Freesound download failed: {e}")
        
        return None
    
    def generate_sfx_track(
        self,
        scenes: List[Dict],
        total_duration: float,
        output_path: str = None
    ) -> SFXResult:
        """
        Generate complete SFX track for video.
        
        Args:
            scenes: Scene list with timing and emotion
            total_duration: Total duration in seconds
            output_path: Output audio path
            
        Returns:
            SFXResult
        """
        if output_path is None:
            output_path = str(self.cache_dir / "sfx_track.wav")
        
        sfx_events = []
        
        # Collect SFX events
        current_time = 0
        for scene in scenes:
            duration = scene.get("duration", 4.0)
            emotion = scene.get("emotion", "neutral")
            transition = scene.get("transition", "fade")
            
            # Transition SFX at scene start
            trigger = f"transition:{transition}"
            sfx_path = self.get_sfx(trigger)
            if sfx_path:
                sfx_events.append({
                    "path": sfx_path,
                    "time": current_time,
                    "volume": -12  # dB
                })
            
            # Emotion SFX
            if emotion in ["tense", "epic", "happy"]:
                trigger = f"emotion:{emotion}"
                sfx_path = self.get_sfx(trigger)
                if sfx_path:
                    sfx_events.append({
                        "path": sfx_path,
                        "time": current_time + duration * 0.3,
                        "volume": -15
                    })
            
            current_time += duration
        
        if not sfx_events:
            # Create silent track
            self._create_silent_audio(output_path, total_duration)
            return SFXResult(
                success=True,
                output_path=output_path,
                sfx_count=0,
                source="none"
            )
        
        # Mix SFX events
        result = self._mix_sfx_events(sfx_events, total_duration, output_path)
        return result
    
    def _mix_sfx_events(
        self,
        events: List[Dict],
        total_duration: float,
        output_path: str
    ) -> SFXResult:
        """Mix multiple SFX events into one track."""
        
        # Create silent base
        silent_path = self.cache_dir / "silent_base.wav"
        self._create_silent_audio(str(silent_path), total_duration)
        
        # Build FFmpeg filter for mixing
        inputs = ["-i", str(silent_path)]
        filter_parts = ["[0:a]volume=0dB[base]"]
        mix_inputs = "[base]"
        
        for i, event in enumerate(events):
            inputs.extend(["-i", event["path"]])
            delay_ms = int(event["time"] * 1000)
            volume = event.get("volume", -12)
            
            filter_parts.append(
                f"[{i+1}:a]volume={volume}dB,adelay={delay_ms}|{delay_ms}[sfx{i}]"
            )
            mix_inputs += f"[sfx{i}]"
        
        # Mix all
        n_inputs = len(events) + 1
        filter_parts.append(f"{mix_inputs}amix=inputs={n_inputs}:duration=first[out]")
        
        filter_complex = ";".join(filter_parts)
        
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                return SFXResult(
                    success=True,
                    output_path=output_path,
                    sfx_count=len(events),
                    source="mixed"
                )
        except Exception:
            pass
        
        # Fallback to first SFX only
        if events:
            return SFXResult(
                success=True,
                output_path=events[0]["path"],
                sfx_count=1,
                source="fallback"
            )
        
        return SFXResult(
            success=False,
            output_path="",
            sfx_count=0,
            source="",
            error="Mix failed"
        )
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """Create silent audio file."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
            "-c:a", "pcm_s16le",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_sfx_for_trigger(trigger: str) -> Optional[str]:
    """Get SFX path for trigger."""
    engine = SFXEngine()
    return engine.get_sfx(trigger)


def generate_scene_sfx(scenes: List[Dict], duration: float) -> str:
    """Generate SFX track for scenes."""
    engine = SFXEngine()
    result = engine.generate_sfx_track(scenes, duration)
    return result.output_path if result.success else ""
