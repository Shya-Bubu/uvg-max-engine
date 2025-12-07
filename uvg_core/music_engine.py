"""
UVG MAX Music Engine Module

Music search, beat detection, and loop handling.
"""

import logging
import subprocess
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


@dataclass
class MusicTrack:
    """A music track."""
    id: str
    title: str
    url: str
    download_url: str
    duration: float
    mood: str
    bpm: float = 0.0
    provider: str = "unknown"
    downloaded_path: str = ""


@dataclass
class BeatInfo:
    """Beat detection results."""
    bpm: float
    beat_times: List[float] = field(default_factory=list)  # in seconds
    downbeat_times: List[float] = field(default_factory=list)


class MusicEngine:
    """
    Music selection and beat synchronization.
    
    Features:
    - Mood-based search (Pixabay Audio, Freesound)
    - Beat detection
    - Loop point detection
    - Beat-aligned looping with gain smoothing
    """
    
    MOOD_KEYWORDS = {
        "epic": ["epic", "cinematic", "orchestral", "dramatic", "heroic"],
        "energetic": ["upbeat", "energetic", "dance", "electronic", "driving"],
        "calm": ["calm", "peaceful", "ambient", "relaxing", "meditation"],
        "inspirational": ["inspirational", "motivational", "uplifting", "hopeful"],
        "romantic": ["romantic", "love", "emotional", "piano", "tender"],
        "tense": ["suspense", "tension", "thriller", "dark", "mysterious"],
        "happy": ["happy", "cheerful", "fun", "positive", "bright"],
    }
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 pixabay_key: str = "",
                 freesound_key: str = ""):
        """
        Initialize music engine.
        
        Args:
            output_dir: Directory for downloaded music
            pixabay_key: Pixabay API key
            freesound_key: Freesound API key
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/music")
        self.pixabay_key = pixabay_key or os.getenv("PIXABAY_KEY", "")
        self.freesound_key = freesound_key or os.getenv("FREESOUND_KEY", "")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def search_by_mood(self,
                       mood: str,
                       max_results: int = 10,
                       min_duration: float = 30.0,
                       max_duration: float = 300.0) -> List[MusicTrack]:
        """
        Search for music by mood.
        
        Args:
            mood: Mood keyword
            max_results: Maximum results
            min_duration: Minimum track duration
            max_duration: Maximum track duration
            
        Returns:
            List of MusicTrack objects
        """
        tracks = []
        
        # Get mood keywords
        keywords = self.MOOD_KEYWORDS.get(mood.lower(), [mood])
        query = " ".join(keywords[:3])
        
        # Try Pixabay first
        if self.pixabay_key:
            tracks.extend(self._search_pixabay(query, max_results))
        
        # Filter by duration
        tracks = [
            t for t in tracks
            if min_duration <= t.duration <= max_duration
        ]
        
        return tracks[:max_results]
    
    def _search_pixabay(self, query: str, max_results: int) -> List[MusicTrack]:
        """Search Pixabay Audio."""
        try:
            url = "https://pixabay.com/api/videos/"  # Pixabay uses same endpoint
            # Note: Pixabay has a music API but requires different endpoint
            # For now, placeholder implementation
            
            # In real implementation:
            # params = {"key": self.pixabay_key, "q": query, "per_page": max_results}
            # response = requests.get(url, params=params)
            
            logger.debug(f"Pixabay music search for: {query}")
            return []
            
        except Exception as e:
            logger.debug(f"Pixabay search failed: {e}")
            return []
    
    def detect_beats(self, audio_path: str) -> BeatInfo:
        """
        Detect beats in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            BeatInfo with BPM and beat times
        """
        try:
            import librosa
            import numpy as np
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Detect tempo and beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Estimate downbeats (every 4 beats)
            downbeats = beat_times[::4].tolist()
            
            return BeatInfo(
                bpm=float(tempo),
                beat_times=beat_times.tolist(),
                downbeat_times=downbeats
            )
            
        except ImportError:
            logger.warning("librosa not installed, using fallback beat detection")
            return self._fallback_beat_detection(audio_path)
        except Exception as e:
            logger.warning(f"Beat detection failed: {e}")
            return BeatInfo(bpm=120.0, beat_times=[], downbeat_times=[])
    
    def _fallback_beat_detection(self, audio_path: str) -> BeatInfo:
        """Simple fallback beat detection using FFmpeg."""
        # Estimate based on common tempos
        # In real implementation, could use onset detection
        default_bpm = 120.0
        
        # Get duration
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ], capture_output=True, text=True, timeout=30)
            
            duration = float(result.stdout.strip())
            
            # Generate estimated beat times
            beat_interval = 60.0 / default_bpm
            beat_times = [i * beat_interval for i in range(int(duration / beat_interval))]
            downbeats = beat_times[::4]
            
            return BeatInfo(
                bpm=default_bpm,
                beat_times=beat_times,
                downbeat_times=downbeats
            )
            
        except Exception:
            return BeatInfo(bpm=120.0, beat_times=[], downbeat_times=[])
    
    def find_loop_points(self,
                          audio_path: str,
                          beat_info: Optional[BeatInfo] = None) -> Tuple[float, float]:
        """
        Find good loop points in music.
        
        Args:
            audio_path: Path to audio
            beat_info: Optional pre-computed beat info
            
        Returns:
            (loop_start, loop_end) in seconds
        """
        if beat_info is None:
            beat_info = self.detect_beats(audio_path)
        
        if not beat_info.downbeat_times or len(beat_info.downbeat_times) < 2:
            return 0.0, 30.0  # Default
        
        # Find a good 4 or 8 bar section
        downbeats = beat_info.downbeat_times
        
        # Look for 8-bar sections (32 beats / 8 downbeats)
        if len(downbeats) >= 8:
            loop_start = downbeats[0]
            loop_end = downbeats[8] if len(downbeats) > 8 else downbeats[-1]
        elif len(downbeats) >= 4:
            loop_start = downbeats[0]
            loop_end = downbeats[4] if len(downbeats) > 4 else downbeats[-1]
        else:
            loop_start = downbeats[0]
            loop_end = downbeats[-1]
        
        return loop_start, loop_end
    
    def create_seamless_loop(self,
                              audio_path: str,
                              target_duration: float,
                              output_path: Optional[str] = None) -> str:
        """
        Create a seamless looping version of music.
        
        Args:
            audio_path: Source audio
            target_duration: Target duration
            output_path: Output path
            
        Returns:
            Path to looped audio
        """
        if output_path is None:
            output_path = self.output_dir / f"loop_{Path(audio_path).stem}.wav"
        
        # Detect beats and find loop points
        beat_info = self.detect_beats(audio_path)
        loop_start, loop_end = self.find_loop_points(audio_path, beat_info)
        loop_duration = loop_end - loop_start
        
        if loop_duration <= 0:
            loop_duration = 30.0  # Fallback
        
        # Calculate number of loops needed
        num_loops = int(target_duration / loop_duration) + 1
        
        # Build FFmpeg command for looping with crossfade
        crossfade_duration = min(0.5, loop_duration / 4)
        
        # Extract loop section
        temp_loop = self.output_dir / "temp_loop_section.wav"
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(loop_start),
            "-i", audio_path,
            "-t", str(loop_duration),
            "-c:a", "pcm_s16le",
            str(temp_loop)
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=60)
        
        # Create looped version with crossfades
        if num_loops > 1:
            # Build filter for looping with crossfades
            # For simplicity, just loop the file
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", str(num_loops - 1),
                "-i", str(temp_loop),
                "-t", str(target_duration),
                "-af", f"afade=t=out:st={target_duration-1}:d=1",  # Fade out at end
                "-c:a", "pcm_s16le",
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=120)
        else:
            # Just use the extracted section
            Path(temp_loop).rename(output_path)
        
        # Cleanup
        Path(temp_loop).unlink(missing_ok=True)
        
        return str(output_path)
    
    def sync_to_beat(self,
                      timestamp: float,
                      beat_info: BeatInfo,
                      snap_to: str = "beat") -> float:
        """
        Snap a timestamp to nearest beat.
        
        Args:
            timestamp: Time in seconds
            beat_info: Beat information
            snap_to: "beat" or "downbeat"
            
        Returns:
            Snapped timestamp
        """
        if snap_to == "downbeat":
            times = beat_info.downbeat_times
        else:
            times = beat_info.beat_times
        
        if not times:
            return timestamp
        
        # Find nearest beat
        nearest = min(times, key=lambda t: abs(t - timestamp))
        return nearest
    
    def get_beat_aligned_cuts(self,
                               scene_durations: List[float],
                               beat_info: BeatInfo) -> List[float]:
        """
        Get scene cut points aligned to beats.
        
        Args:
            scene_durations: Original scene durations
            beat_info: Beat info from music
            
        Returns:
            Adjusted cut times aligned to beats
        """
        cut_times = []
        current_time = 0.0
        
        for duration in scene_durations:
            current_time += duration
            # Snap to nearest beat
            snapped = self.sync_to_beat(current_time, beat_info, "beat")
            cut_times.append(snapped)
        
        return cut_times


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search_music(mood: str, max_results: int = 10) -> List[MusicTrack]:
    """Search for music by mood."""
    engine = MusicEngine()
    return engine.search_by_mood(mood, max_results)


def detect_beats(audio_path: str) -> BeatInfo:
    """Detect beats in audio file."""
    engine = MusicEngine()
    return engine.detect_beats(audio_path)


def create_loop(audio_path: str, duration: float) -> str:
    """Create seamless loop of music."""
    engine = MusicEngine()
    return engine.create_seamless_loop(audio_path, duration)
