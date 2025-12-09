# uvg_core/audio_mixer.py
"""
UVG MAX Audio Mixer.

Professional audio mixing with:
- Sidechain ducking
- Loudness normalization (-14 LUFS)
- Fade in/out
- Noise gate
- De-esser
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MixResult:
    """Result of audio mixing operation."""
    success: bool
    output_path: str
    error: str = ""


class AudioMixer:
    """
    Professional audio mixer for video production.
    
    Features:
    - Loudness normalization to broadcast standards
    - Sidechain compression for ducking
    - Fade effects
    - Noise gate
    """
    
    LUFS_TARGET = -14.0  # Broadcast standard
    TRUE_PEAK = -1.5
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize audio mixer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_ffmpeg(self, cmd: List[str], timeout: int = 120) -> bool:
        """Run FFmpeg command."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                logger.debug(f"FFmpeg error: {result.stderr[:500]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return False
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            return False
    
    def normalize_loudness(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        target_lufs: float = None
    ) -> MixResult:
        """
        Normalize audio to target loudness.
        
        Args:
            audio_path: Input audio path
            output_path: Output path (auto-generated if None)
            target_lufs: Target loudness (default: -14 LUFS)
            
        Returns:
            MixResult
        """
        if target_lufs is None:
            target_lufs = self.LUFS_TARGET
        
        if output_path is None:
            output_path = str(self.output_dir / "normalized.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", f"loudnorm=I={target_lufs}:TP={self.TRUE_PEAK}:LRA=11",
            "-ar", "44100",
            output_path
        ]
        
        success = self._run_ffmpeg(cmd)
        return MixResult(
            success=success,
            output_path=output_path if success else "",
            error="" if success else "normalization_failed"
        )
    
    def apply_ducking(
        self,
        voice_path: str,
        music_path: str,
        output_path: Optional[str] = None,
        duck_db: float = -12.0,
        attack_ms: int = 50,
        release_ms: int = 500
    ) -> MixResult:
        """
        Apply sidechain ducking - lower music when voice is present.
        
        Args:
            voice_path: Voice/narration audio
            music_path: Background music
            output_path: Output path
            duck_db: Amount to duck in dB (negative)
            attack_ms: Attack time in ms
            release_ms: Release time in ms
            
        Returns:
            MixResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "ducked.wav")
        
        # Sidechain compress: use voice to control music volume
        filter_complex = (
            f"[1:a]asplit=2[sc][music];"
            f"[0:a][sc]sidechaincompress="
            f"threshold=0.02:ratio=10:attack={attack_ms}:release={release_ms}[voice];"
            f"[voice][music]amix=inputs=2:duration=first:weights=1 0.3[out]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "44100",
            output_path
        ]
        
        success = self._run_ffmpeg(cmd)
        return MixResult(
            success=success,
            output_path=output_path if success else "",
            error="" if success else "ducking_failed"
        )
    
    def apply_fade(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        fade_in_sec: float = 1.0,
        fade_out_sec: float = 2.0
    ) -> MixResult:
        """
        Apply fade in and fade out effects.
        
        Args:
            audio_path: Input audio
            output_path: Output path
            fade_in_sec: Fade in duration
            fade_out_sec: Fade out duration
            
        Returns:
            MixResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "faded.wav")
        
        # Get audio duration first
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip())
        except:
            duration = 60.0  # Fallback
        
        fade_out_start = max(0, duration - fade_out_sec)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", f"afade=t=in:d={fade_in_sec},afade=t=out:st={fade_out_start}:d={fade_out_sec}",
            output_path
        ]
        
        success = self._run_ffmpeg(cmd)
        return MixResult(
            success=success,
            output_path=output_path if success else "",
            error="" if success else "fade_failed"
        )
    
    def apply_noise_gate(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        threshold: float = 0.01,
        attack_ms: int = 5,
        release_ms: int = 50
    ) -> MixResult:
        """
        Apply noise gate to remove background noise.
        
        Args:
            audio_path: Input audio
            output_path: Output path
            threshold: Gate threshold (0-1)
            attack_ms: Attack time
            release_ms: Release time
            
        Returns:
            MixResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "gated.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", f"agate=threshold={threshold}:attack={attack_ms}:release={release_ms}",
            output_path
        ]
        
        success = self._run_ffmpeg(cmd)
        return MixResult(
            success=success,
            output_path=output_path if success else "",
            error="" if success else "noise_gate_failed"
        )
    
    def apply_deesser(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        frequency: int = 6000
    ) -> MixResult:
        """
        Apply de-esser to reduce sibilance.
        
        Args:
            audio_path: Input audio
            output_path: Output path
            frequency: Target frequency for de-essing
            
        Returns:
            MixResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "deessed.wav")
        
        # Simple de-esser using highpass and compression
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", f"highpass=f=50,lowpass=f=15000,compand=attacks=0:points=-80/-80|-6/-6|0/-3:gain=0",
            output_path
        ]
        
        success = self._run_ffmpeg(cmd)
        return MixResult(
            success=success,
            output_path=output_path if success else "",
            error="" if success else "deesser_failed"
        )
    
    def mix_voice_and_music(
        self,
        voice_path: str,
        music_path: str,
        output_path: Optional[str] = None,
        music_volume: float = 0.15,
        apply_ducking: bool = True,
        fade_music: bool = True
    ) -> MixResult:
        """
        Mix voice and background music with full processing chain.
        
        Args:
            voice_path: Voice/narration track
            music_path: Background music track
            output_path: Output path
            music_volume: Music volume (0-1)
            apply_ducking: Apply sidechain ducking
            fade_music: Apply fade in/out to music
            
        Returns:
            MixResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "final_mix.wav")
        
        # Step 1: Normalize voice
        voice_normalized = self.normalize_loudness(
            voice_path,
            str(self.output_dir / "voice_norm.wav")
        )
        
        if not voice_normalized.success:
            return voice_normalized
        
        # Step 2: Process music (fade if requested)
        music_processed = music_path
        if fade_music:
            fade_result = self.apply_fade(
                music_path,
                str(self.output_dir / "music_faded.wav")
            )
            if fade_result.success:
                music_processed = fade_result.output_path
        
        # Step 3: Mix with or without ducking
        if apply_ducking:
            return self.apply_ducking(
                voice_normalized.output_path,
                music_processed,
                output_path
            )
        else:
            # Simple mix without ducking
            filter_complex = f"[0:a][1:a]amix=inputs=2:duration=first:weights=1 {music_volume}[out]"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", voice_normalized.output_path,
                "-i", music_processed,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", "44100",
                output_path
            ]
            
            success = self._run_ffmpeg(cmd)
            return MixResult(
                success=success,
                output_path=output_path if success else "",
                error="" if success else "mix_failed"
            )
    
    def process_voice(
        self,
        voice_path: str,
        output_path: Optional[str] = None,
        normalize: bool = True,
        noise_gate: bool = True,
        deess: bool = False
    ) -> MixResult:
        """
        Full voice processing chain.
        
        Args:
            voice_path: Input voice audio
            output_path: Output path
            normalize: Apply loudness normalization
            noise_gate: Apply noise gate
            deess: Apply de-esser
            
        Returns:
            MixResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "voice_processed.wav")
        
        current_path = voice_path
        
        # Processing chain
        if noise_gate:
            result = self.apply_noise_gate(current_path)
            if result.success:
                current_path = result.output_path
        
        if deess:
            result = self.apply_deesser(current_path)
            if result.success:
                current_path = result.output_path
        
        if normalize:
            result = self.normalize_loudness(current_path, output_path)
            return result
        
        return MixResult(success=True, output_path=current_path)


# Convenience functions
def mix_audio(
    voice_path: str,
    music_path: str = None,
    output_path: str = None
) -> str:
    """Mix voice with optional music."""
    mixer = AudioMixer()
    
    if music_path:
        result = mixer.mix_voice_and_music(voice_path, music_path, output_path)
    else:
        result = mixer.process_voice(voice_path, output_path)
    
    return result.output_path if result.success else ""


def normalize_audio(audio_path: str, output_path: str = None) -> str:
    """Normalize audio to -14 LUFS."""
    mixer = AudioMixer()
    result = mixer.normalize_loudness(audio_path, output_path)
    return result.output_path if result.success else ""
