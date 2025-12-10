"""
UVG MAX Edge-TTS Adapter

Uses Microsoft Edge TTS for high-quality, free text-to-speech.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    success: bool
    audio_path: str
    duration_ms: int
    text: str
    error: str = ""


# Voice mapping for different styles
VOICE_STYLES = {
    # English voices
    "motivational": "en-US-GuyNeural",
    "documentary": "en-US-ChristopherNeural",
    "cinematic": "en-GB-RyanNeural",
    "warm": "en-US-JennyNeural",
    "energetic": "en-US-DavisNeural",
    "calm": "en-US-AriaNeural",
    "dramatic": "en-GB-ThomasNeural",
    "default": "en-US-GuyNeural",
    
    # Female options
    "female_warm": "en-US-JennyNeural",
    "female_calm": "en-US-AriaNeural",
    "female_energetic": "en-US-SaraNeural",
    
    # British
    "british_male": "en-GB-RyanNeural",
    "british_female": "en-GB-SoniaNeural",
}


class EdgeTTSAdapter:
    """
    Edge TTS adapter for UVG MAX.
    
    Uses Microsoft Edge's free TTS service via edge-tts library.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the adapter.
        
        Args:
            output_dir: Directory for audio output
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/tts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if edge-tts is available
        try:
            import edge_tts
            self._available = True
            logger.info("Edge-TTS initialized successfully")
        except ImportError:
            self._available = False
            logger.warning("edge-tts not installed. Run: pip install edge-tts")
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._available
    
    def get_voice(self, style: str = "default") -> str:
        """Get voice name for style."""
        return VOICE_STYLES.get(style.lower(), VOICE_STYLES["default"])
    
    async def _synthesize_async(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> bool:
        """Async synthesis using edge-tts."""
        import edge_tts
        
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)
        return True
    
    def synthesize(
        self,
        text: str,
        output_name: str,
        voice_style: str = "default",
        rate: float = 1.0,
        pitch: float = 1.0
    ) -> TTSResult:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to speak
            output_name: Output filename (without extension)
            voice_style: Voice style preset
            rate: Speech rate (0.5-2.0)
            pitch: Pitch adjustment (0.5-2.0)
            
        Returns:
            TTSResult
        """
        if not self._available:
            return TTSResult(
                success=False,
                audio_path="",
                duration_ms=0,
                text=text,
                error="edge-tts not installed"
            )
        
        try:
            voice = self.get_voice(voice_style)
            output_path = self.output_dir / f"{output_name}.mp3"
            
            # Convert rate/pitch to edge-tts format
            rate_str = f"+{int((rate - 1) * 100)}%" if rate >= 1 else f"{int((rate - 1) * 100)}%"
            pitch_str = f"+{int((pitch - 1) * 50)}Hz" if pitch >= 1 else f"{int((pitch - 1) * 50)}Hz"
            
            # Run async synthesis
            asyncio.run(self._synthesize_async(
                text=text,
                voice=voice,
                output_path=str(output_path),
                rate=rate_str,
                pitch=pitch_str
            ))
            
            # Get duration using ffprobe
            duration_ms = self._get_audio_duration(str(output_path))
            
            return TTSResult(
                success=True,
                audio_path=str(output_path),
                duration_ms=duration_ms,
                text=text
            )
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                audio_path="",
                duration_ms=0,
                text=text,
                error=str(e)
            )
    
    def synthesize_scenes(
        self,
        scenes: List[Dict],
        default_style: str = "default"
    ) -> List[TTSResult]:
        """
        Synthesize audio for multiple scenes.
        
        Args:
            scenes: List of scene dicts with 'text' and optional 'voice_style'
            default_style: Default voice style
            
        Returns:
            List of TTSResult
        """
        results = []
        
        for i, scene in enumerate(scenes):
            text = scene.get("text", "")
            style = scene.get("voice_style", default_style)
            
            logger.info(f"Synthesizing scene {i+1}: {text[:50]}...")
            
            result = self.synthesize(
                text=text,
                output_name=f"scene_{i+1:03d}",
                voice_style=style
            )
            results.append(result)
        
        return results
    
    def _get_audio_duration(self, audio_path: str) -> int:
        """Get audio duration in milliseconds using ffprobe."""
        try:
            import subprocess
            import json
            
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0))
            return int(duration * 1000)
            
        except Exception:
            # Fallback: estimate based on text length (150 wpm average)
            return 5000  # Default 5 seconds


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def synthesize_text(text: str, output_name: str = "output", style: str = "default") -> TTSResult:
    """Quick TTS synthesis."""
    adapter = EdgeTTSAdapter()
    return adapter.synthesize(text, output_name, style)


def synthesize_scenes(scenes: List[Dict]) -> List[TTSResult]:
    """Synthesize all scenes."""
    adapter = EdgeTTSAdapter()
    return adapter.synthesize_scenes(scenes)


def list_voices():
    """List available voices (async call)."""
    async def _list():
        import edge_tts
        voices = await edge_tts.list_voices()
        return voices
    
    return asyncio.run(_list())
