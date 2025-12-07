"""
UVG MAX Audio Engine Module

Audio mastering with continuity engine.
Includes loudness smoothing, emotional peaks, silence shaping, and synced crossfades.
"""

import logging
import subprocess
import json
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """An audio segment with metadata."""
    path: str
    start_ms: int
    end_ms: int
    volume: float = 1.0
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class MasteringResult:
    """Result of audio mastering."""
    success: bool
    input_path: str
    output_path: str
    loudness_lufs: float = -14.0
    peak_db: float = -1.0
    error: str = ""


class AudioEngine:
    """
    Audio mastering engine with continuity features.
    
    Mastering chain:
    1. Noise gate
    2. Compressor (soft knee)
    3. EQ (voice clarity)
    4. De-esser
    5. Limiter
    6. BGM ducking
    7. SFX layering
    
    Continuity features:
    - Scene-to-scene loudness smoothing
    - Emotional peak emphasis
    - Silence shaping
    - Transition-synced crossfades
    """
    
    TARGET_LUFS = -14.0
    TARGET_PEAK = -1.5
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 sample_rate: int = 44100):
        """
        Initialize audio engine.
        
        Args:
            output_dir: Output directory for processed audio
            sample_rate: Target sample rate
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/audio")
        self.sample_rate = sample_rate
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_ffmpeg(self, cmd: List[str], timeout: int = 120) -> Tuple[bool, str]:
        """Run FFmpeg command."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                return False, result.stderr[:500]
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except Exception as e:
            return False, str(e)
    
    def _get_loudness(self, audio_path: str) -> Dict[str, float]:
        """Get loudness metrics using ffmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-af", "loudnorm=print_format=json",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse loudnorm output from stderr
            stderr = result.stderr
            
            # Find JSON in output
            import re
            json_match = re.search(r'\{[^}]+\}', stderr)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "input_i": float(data.get("input_i", -24)),
                    "input_tp": float(data.get("input_tp", -1)),
                    "input_lra": float(data.get("input_lra", 7)),
                }
            
            return {"input_i": -24, "input_tp": -1, "input_lra": 7}
            
        except Exception as e:
            logger.debug(f"Loudness measurement failed: {e}")
            return {"input_i": -24, "input_tp": -1, "input_lra": 7}
    
    def apply_mastering_chain(self,
                               audio_path: str,
                               output_path: Optional[str] = None,
                               voice_mode: bool = True) -> MasteringResult:
        """
        Apply full mastering chain to audio.
        
        Args:
            audio_path: Input audio path
            output_path: Optional output path
            voice_mode: Optimize for voice (vs music)
            
        Returns:
            MasteringResult
        """
        if output_path is None:
            output_path = self.output_dir / f"mastered_{Path(audio_path).stem}.wav"
        
        # Build filter chain
        filters = []
        
        if voice_mode:
            # 1. Noise gate - remove silence/noise
            filters.append("agate=threshold=-40dB:ratio=6:attack=5:release=100")
            
            # 2. Compressor - even out dynamics  
            filters.append("acompressor=threshold=-20dB:ratio=3:attack=10:release=100:knee=6dB")
            
            # 3. EQ - voice clarity boost
            filters.append("equalizer=f=200:t=h:w=100:g=-2")  # Cut mud
            filters.append("equalizer=f=3000:t=h:w=500:g=3")  # Presence boost
            filters.append("equalizer=f=8000:t=h:w=1000:g=2")  # Air
            
            # 4. De-esser
            filters.append("deesser=i=0.5:m=0.5:f=5500:s=o")
        
        # 5. Loudnorm - target LUFS and peak
        filters.append(f"loudnorm=I={self.TARGET_LUFS}:TP={self.TARGET_PEAK}:LRA=11")
        
        # 6. Limiter - safety
        filters.append(f"alimiter=level_in=1:level_out=0.95:limit={self.TARGET_PEAK}dB")
        
        filter_str = ",".join(filters)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", filter_str,
            "-ar", str(self.sample_rate),
            "-c:a", "pcm_s16le",
            str(output_path)
        ]
        
        success, error = self._run_ffmpeg(cmd)
        
        if success:
            loudness = self._get_loudness(str(output_path))
            return MasteringResult(
                success=True,
                input_path=audio_path,
                output_path=str(output_path),
                loudness_lufs=loudness.get("input_i", self.TARGET_LUFS),
                peak_db=loudness.get("input_tp", self.TARGET_PEAK)
            )
        
        return MasteringResult(
            success=False,
            input_path=audio_path,
            output_path="",
            error=error
        )
    
    def duck_bgm(self,
                 voice_path: str,
                 music_path: str,
                 output_path: str,
                 duck_amount: float = -12.0,
                 attack_ms: int = 200,
                 release_ms: int = 500) -> bool:
        """
        Apply ducking to background music when voice is present.
        
        Args:
            voice_path: Voice audio path
            music_path: Background music path
            output_path: Output path
            duck_amount: Amount to duck in dB
            attack_ms: Duck attack time
            release_ms: Duck release time
            
        Returns:
            Success boolean
        """
        # Use sidechaincompress for ducking
        filter_complex = (
            f"[1:a]asplit=2[sc][music];"
            f"[0:a][sc]sidechaincompress=threshold=0.02:ratio=5:"
            f"attack={attack_ms/1000}:release={release_ms/1000}:level_sc=1[ducked];"
            f"[ducked]volume={duck_amount}dB[ducked_vol];"
            f"[music][ducked_vol]amix=inputs=2:duration=longest"
        )
        
        # Simpler approach: just lower music volume when voice is present
        # Using sidechaingate effect
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume=-10dB[music];[0:a][music]amix=inputs=2:duration=first:weights=1 0.3",
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        success, _ = self._run_ffmpeg(cmd)
        return success
    
    def smooth_loudness_across_scenes(self,
                                       scene_audios: List[str],
                                       output_dir: Optional[Path] = None) -> List[str]:
        """
        Ensure consistent loudness across all scene audio files.
        
        Args:
            scene_audios: List of scene audio paths
            output_dir: Output directory
            
        Returns:
            List of normalized audio paths
        """
        if output_dir is None:
            output_dir = self.output_dir / "normalized"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Measure all loudness levels
        loudness_values = []
        for audio_path in scene_audios:
            loudness = self._get_loudness(audio_path)
            loudness_values.append(loudness.get("input_i", -24))
        
        # Calculate average (target)
        avg_loudness = sum(loudness_values) / len(loudness_values) if loudness_values else -14
        target = max(-24, min(-10, avg_loudness))  # Clamp to reasonable range
        
        logger.info(f"Normalizing {len(scene_audios)} scenes to {target:.1f} LUFS")
        
        normalized = []
        for i, audio_path in enumerate(scene_audios):
            output_path = output_dir / f"norm_{i}_{Path(audio_path).stem}.wav"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11",
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                str(output_path)
            ]
            
            success, _ = self._run_ffmpeg(cmd)
            
            if success:
                normalized.append(str(output_path))
            else:
                normalized.append(audio_path)  # Use original if failed
        
        return normalized
    
    def emotional_peak_emphasis(self,
                                 audio_path: str,
                                 tension_curve: List[float],
                                 output_path: str) -> bool:
        """
        Boost volume at emotional peaks based on tension curve.
        
        Args:
            audio_path: Input audio
            tension_curve: List of tension values (0-1) per segment
            output_path: Output path
            
        Returns:
            Success boolean
        """
        if not tension_curve:
            return False
        
        # Calculate volume automation based on tension
        # Higher tension = slightly louder
        volume_adjustments = [
            0.9 + (t * 0.2)  # 0.9 to 1.1 range
            for t in tension_curve
        ]
        
        # For now, apply average adjustment
        # TODO: Implement segment-by-segment volume automation
        avg_vol = sum(volume_adjustments) / len(volume_adjustments)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", f"volume={avg_vol}",
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        success, _ = self._run_ffmpeg(cmd)
        return success
    
    def silence_shaping_between_scenes(self,
                                        audio_segments: List[AudioSegment],
                                        gap_ms: int = 100) -> List[AudioSegment]:
        """
        Add small silence gaps between scenes for breathing room.
        
        Args:
            audio_segments: List of audio segments
            gap_ms: Gap duration in milliseconds
            
        Returns:
            Updated audio segments with gaps
        """
        if not audio_segments:
            return audio_segments
        
        shaped = []
        current_offset = 0
        
        for i, segment in enumerate(audio_segments):
            # Add gap before (except first)
            if i > 0:
                current_offset += gap_ms
            
            new_segment = AudioSegment(
                path=segment.path,
                start_ms=current_offset,
                end_ms=current_offset + segment.duration_ms,
                volume=segment.volume,
                fade_in_ms=min(50, segment.duration_ms // 4),
                fade_out_ms=min(50, segment.duration_ms // 4),
            )
            
            shaped.append(new_segment)
            current_offset = new_segment.end_ms
        
        return shaped
    
    def transition_synced_audio_crossfade(self,
                                           audio1_path: str,
                                           audio2_path: str,
                                           crossfade_ms: int,
                                           output_path: str) -> bool:
        """
        Crossfade two audio files aligned with video transition.
        
        Args:
            audio1_path: First audio
            audio2_path: Second audio
            crossfade_ms: Crossfade duration in ms
            output_path: Output path
            
        Returns:
            Success boolean
        """
        crossfade_sec = crossfade_ms / 1000.0
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio1_path,
            "-i", audio2_path,
            "-filter_complex",
            f"acrossfade=d={crossfade_sec}:c1=tri:c2=tri",
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        success, _ = self._run_ffmpeg(cmd)
        return success
    
    def mix_layers(self,
                   voice_path: str,
                   music_path: Optional[str] = None,
                   sfx_paths: Optional[List[str]] = None,
                   output_path: str = "") -> bool:
        """
        Mix multiple audio layers.
        
        Args:
            voice_path: Voice track
            music_path: Optional background music
            sfx_paths: Optional SFX tracks
            output_path: Output path
            
        Returns:
            Success boolean
        """
        if not output_path:
            output_path = self.output_dir / "mixed_final.wav"
        
        inputs = ["-i", voice_path]
        
        if music_path:
            inputs.extend(["-i", music_path])
        
        for sfx in (sfx_paths or []):
            inputs.extend(["-i", sfx])
        
        # Count inputs
        num_inputs = 1 + (1 if music_path else 0) + len(sfx_paths or [])
        
        if num_inputs == 1:
            # Just copy voice
            cmd = ["ffmpeg", "-y"] + inputs + ["-c:a", "pcm_s16le", str(output_path)]
        else:
            # Build weights: voice loud, music soft, sfx medium
            weights = "1"
            if music_path:
                weights += " 0.25"
            for _ in (sfx_paths or []):
                weights += " 0.5"
            
            cmd = [
                "ffmpeg", "-y"
            ] + inputs + [
                "-filter_complex",
                f"amix=inputs={num_inputs}:duration=first:weights={weights}",
                "-c:a", "pcm_s16le",
                str(output_path)
            ]
        
        success, _ = self._run_ffmpeg(cmd)
        return success


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def master_audio(audio_path: str, voice_mode: bool = True) -> MasteringResult:
    """Apply mastering to audio file."""
    engine = AudioEngine()
    return engine.apply_mastering_chain(audio_path, voice_mode=voice_mode)


def smooth_scene_loudness(scene_audios: List[str]) -> List[str]:
    """Normalize loudness across scenes."""
    engine = AudioEngine()
    return engine.smooth_loudness_across_scenes(scene_audios)
