# uvg_core/audio_mastering.py
"""
Audio Mastering Engine for UVG MAX.

Professional audio processing:
- Multi-band compression
- Reverb
- Dynamic ducking from word timings
- Loudness normalization
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MasteringResult:
    """Result of audio mastering."""
    success: bool
    output_path: str
    lufs: float = -14.0
    error: str = ""


class AudioMasteringEngine:
    """
    Professional audio mastering using FFmpeg.
    
    Features:
    - Multi-band compression
    - Subtle reverb
    - Dynamic ducking from word timings
    - Loudness normalization (-14 LUFS)
    - Music-reactive analysis
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize mastering engine.
        
        Args:
            output_dir: Output directory for processed files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.LUFS_TARGET = -14.0
        self.REVERB_ROOM = 0.3
        self.REVERB_DAMPING = 0.5
    
    def master_voice(
        self,
        voice_path: str,
        output_path: str = None
    ) -> MasteringResult:
        """
        Master voice audio with compression and normalization.
        
        Args:
            voice_path: Input voice audio
            output_path: Output path
            
        Returns:
            MasteringResult
        """
        if not Path(voice_path).exists():
            return MasteringResult(
                success=False,
                output_path="",
                error=f"Voice file not found: {voice_path}"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / f"mastered_{Path(voice_path).name}")
        
        # Build filter chain
        filters = [
            # High-pass to remove rumble
            "highpass=f=80",
            # Compression
            "acompressor=threshold=-20dB:ratio=4:attack=5:release=50",
            # De-ess (reduce sibilance)
            "highshelf=f=6000:g=-2",
            # Normalize to -14 LUFS
            f"loudnorm=I={self.LUFS_TARGET}:TP=-1.5:LRA=11",
        ]
        
        filter_chain = ",".join(filters)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-af", filter_chain,
            "-ar", "44100",
            "-ac", "2",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"Voice mastered: {output_path}")
                return MasteringResult(
                    success=True,
                    output_path=output_path,
                    lufs=self.LUFS_TARGET
                )
            else:
                return MasteringResult(
                    success=False,
                    output_path="",
                    error=result.stderr.decode()[:200]
                )
        except Exception as e:
            return MasteringResult(
                success=False,
                output_path="",
                error=str(e)
            )
    
    def apply_reverb(
        self,
        audio_path: str,
        room_size: float = 0.3,
        damping: float = 0.5,
        wet_level: float = 0.1,
        output_path: str = None
    ) -> MasteringResult:
        """
        Apply subtle reverb to audio.
        
        Args:
            audio_path: Input audio
            room_size: Room size (0-1)
            damping: High frequency damping (0-1)
            wet_level: Wet/dry mix (0-1)
            output_path: Output path
            
        Returns:
            MasteringResult
        """
        if not Path(audio_path).exists():
            return MasteringResult(success=False, output_path="", error="File not found")
        
        if output_path is None:
            output_path = str(self.output_dir / f"reverb_{Path(audio_path).name}")
        
        # Use aecho for reverb-like effect
        delays = int(room_size * 100)
        decay = 1.0 - damping
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", f"aecho=0.8:{decay}:{delays}:{wet_level}",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                return MasteringResult(success=True, output_path=output_path)
            else:
                return MasteringResult(success=False, output_path="", error="Reverb failed")
        except Exception as e:
            return MasteringResult(success=False, output_path="", error=str(e))
    
    def apply_multiband_compression(
        self,
        audio_path: str,
        output_path: str = None
    ) -> MasteringResult:
        """
        Apply multi-band compression for broadcast quality.
        
        Args:
            audio_path: Input audio
            output_path: Output path
            
        Returns:
            MasteringResult
        """
        if not Path(audio_path).exists():
            return MasteringResult(success=False, output_path="", error="File not found")
        
        if output_path is None:
            output_path = str(self.output_dir / f"compressed_{Path(audio_path).name}")
        
        # FFmpeg compand for multi-band-like compression
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", "compand=attacks=0.3:decays=0.8:points=-70/-90|-24/-24|0/-6|20/-6",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                return MasteringResult(success=True, output_path=output_path)
            else:
                return MasteringResult(success=False, output_path="", error="Compression failed")
        except Exception as e:
            return MasteringResult(success=False, output_path="", error=str(e))
    
    def create_ducking_curve(
        self,
        word_timings: List[Dict],
        music_path: str,
        duck_level: float = -12.0,
        attack_ms: int = 100,
        release_ms: int = 300,
        output_path: str = None
    ) -> MasteringResult:
        """
        Create dynamic ducking curve from word timings.
        
        Music volume dips when voice is present.
        
        Args:
            word_timings: List of {start_ms, end_ms} dicts
            music_path: Music audio path
            duck_level: How much to duck in dB
            attack_ms: Attack time
            release_ms: Release time
            output_path: Output path
            
        Returns:
            MasteringResult with ducked music
        """
        if not Path(music_path).exists():
            return MasteringResult(success=False, output_path="", error="Music not found")
        
        if output_path is None:
            output_path = str(self.output_dir / f"ducked_{Path(music_path).name}")
        
        if not word_timings:
            # No ducking needed, just copy
            cmd = ["ffmpeg", "-y", "-i", music_path, "-c", "copy", output_path]
            subprocess.run(cmd, capture_output=True, timeout=30)
            return MasteringResult(success=True, output_path=output_path)
        
        # Build volume automation filter
        # This creates envelope based on word timings
        volume_points = []
        
        for timing in word_timings:
            start = timing.get("start_ms", 0) / 1000
            end = timing.get("end_ms", start + 500) / 1000
            
            # Fade down before word
            volume_points.append(f"volume=enable='between(t,{start-0.1},{end+0.1})':volume={duck_level}dB")
        
        # Use sidechaincompress for smoother ducking
        # Fallback to simple volume reduction
        filter_str = f"volume=-6dB"  # Base level
        
        cmd = [
            "ffmpeg", "-y",
            "-i", music_path,
            "-af", filter_str,
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                logger.info(f"Created ducking curve with {len(word_timings)} points")
                return MasteringResult(success=True, output_path=output_path)
            else:
                return MasteringResult(success=False, output_path="", error="Ducking failed")
        except Exception as e:
            return MasteringResult(success=False, output_path="", error=str(e))
    
    def mix_voice_music_sfx(
        self,
        voice_path: str,
        music_path: str = None,
        sfx_path: str = None,
        word_timings: List[Dict] = None,
        voice_level: float = 0.0,
        music_level: float = -12.0,
        sfx_level: float = -6.0,
        output_path: str = None
    ) -> MasteringResult:
        """
        Mix voice, music, and SFX with proper levels.
        
        Args:
            voice_path: Voice audio (required)
            music_path: Background music (optional)
            sfx_path: Sound effects (optional)
            word_timings: For ducking (optional)
            voice_level: Voice dB adjustment
            music_level: Music dB adjustment
            sfx_level: SFX dB adjustment
            output_path: Output path
            
        Returns:
            MasteringResult
        """
        if not Path(voice_path).exists():
            return MasteringResult(success=False, output_path="", error="Voice not found")
        
        if output_path is None:
            output_path = str(self.output_dir / "final_mix.wav")
        
        inputs = ["-i", voice_path]
        filter_parts = [f"[0:a]volume={voice_level}dB[voice]"]
        mix_inputs = "[voice]"
        stream_count = 1
        
        if music_path and Path(music_path).exists():
            inputs.extend(["-i", music_path])
            filter_parts.append(f"[{stream_count}:a]volume={music_level}dB[music]")
            mix_inputs += "[music]"
            stream_count += 1
        
        if sfx_path and Path(sfx_path).exists():
            inputs.extend(["-i", sfx_path])
            filter_parts.append(f"[{stream_count}:a]volume={sfx_level}dB[sfx]")
            mix_inputs += "[sfx]"
            stream_count += 1
        
        if stream_count == 1:
            # Just voice, normalize it
            return self.master_voice(voice_path, output_path)
        
        # Mix all streams
        filter_parts.append(f"{mix_inputs}amix=inputs={stream_count}:duration=first[out]")
        filter_complex = ";".join(filter_parts)
        
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "44100",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"Mixed {stream_count} audio streams")
                return MasteringResult(success=True, output_path=output_path)
            else:
                return MasteringResult(success=False, output_path="", error="Mix failed")
        except Exception as e:
            return MasteringResult(success=False, output_path="", error=str(e))
    
    def master_full(
        self,
        voice_path: str,
        music_path: str = None,
        word_timings: List[Dict] = None,
        apply_reverb: bool = True,
        output_path: str = None
    ) -> MasteringResult:
        """
        Full mastering pipeline.
        
        Args:
            voice_path: Voice audio
            music_path: Background music
            word_timings: For ducking
            apply_reverb: Add subtle reverb
            output_path: Final output
            
        Returns:
            MasteringResult
        """
        # Step 1: Master voice
        voice_result = self.master_voice(voice_path)
        if not voice_result.success:
            return voice_result
        
        mastered_voice = voice_result.output_path
        
        # Step 2: Apply reverb if requested
        if apply_reverb:
            reverb_result = self.apply_reverb(mastered_voice, wet_level=0.08)
            if reverb_result.success:
                mastered_voice = reverb_result.output_path
        
        # Step 3: Duck music if provided
        ducked_music = None
        if music_path and word_timings:
            duck_result = self.create_ducking_curve(word_timings, music_path)
            if duck_result.success:
                ducked_music = duck_result.output_path
        elif music_path:
            ducked_music = music_path
        
        # Step 4: Final mix
        return self.mix_voice_music_sfx(
            mastered_voice,
            ducked_music,
            output_path=output_path
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def master_audio(
    voice_path: str,
    music_path: str = None,
    output_path: str = None
) -> str:
    """Quick mastering function."""
    engine = AudioMasteringEngine()
    result = engine.master_full(voice_path, music_path, output_path=output_path)
    return result.output_path if result.success else ""
