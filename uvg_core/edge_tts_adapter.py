"""
UVG MAX TTS Adapter

Uses gTTS (Google Text-to-Speech) for reliable, free synthesis.
Falls back to pyttsx3 if gTTS fails.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    success: bool
    audio_path: str
    duration_ms: int
    text: str
    error: str = ""
    word_timings: List[Dict[str, Any]] = field(default_factory=list)


class EdgeTTSAdapter:
    """
    TTS adapter for UVG MAX using gTTS (Google Text-to-Speech).
    
    Most reliable option for Colab - works without GPU or complex setup.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the adapter.
        
        Args:
            output_dir: Directory for audio output
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/tts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if gTTS is available
        try:
            from gtts import gTTS
            self._gtts_available = True
            logger.info("gTTS initialized successfully")
        except ImportError:
            self._gtts_available = False
            logger.warning("gTTS not installed. Run: pip install gTTS")
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._gtts_available
    
    def synthesize(
        self,
        text: str,
        output_name: str,
        voice_style: str = "default",
        rate: float = 1.0,
        pitch: float = 1.0
    ) -> TTSResult:
        """
        Synthesize text to speech using gTTS.
        
        Args:
            text: Text to speak
            output_name: Output filename (without extension)
            voice_style: Voice style (ignored for gTTS, kept for API compatibility)
            rate: Speech rate (ignored for gTTS)
            pitch: Pitch adjustment (ignored for gTTS)
            
        Returns:
            TTSResult
        """
        if not text.strip():
            return TTSResult(
                success=False,
                audio_path="",
                duration_ms=0,
                text=text,
                error="Empty text provided"
            )
        
        output_path = self.output_dir / f"{output_name}.mp3"
        
        # Try gTTS first
        if self._gtts_available:
            try:
                from gtts import gTTS
                
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(str(output_path))
                
                # Get duration
                duration_ms = self._get_audio_duration(str(output_path))
                
                logger.info(f"TTS generated: {output_path} ({duration_ms}ms)")
                
                return TTSResult(
                    success=True,
                    audio_path=str(output_path),
                    duration_ms=duration_ms,
                    text=text,
                    word_timings=[]
                )
                
            except Exception as e:
                logger.error(f"gTTS failed: {e}")
        
        # Return failure
        return TTSResult(
            success=False,
            audio_path="",
            duration_ms=0,
            text=text,
            error="TTS synthesis failed"
        )
    
    def synthesize_scenes(
        self,
        scenes: List[Dict],
        default_style: str = "default"
    ) -> List[TTSResult]:
        """
        Synthesize audio for multiple scenes.
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
        """Get audio duration in milliseconds."""
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
            # Estimate: ~150 words per minute
            words = len(self.text.split()) if hasattr(self, 'text') else 10
            return int(words / 150 * 60 * 1000)


# Convenience functions
def synthesize_text(text: str, output_name: str = "output", style: str = "default") -> TTSResult:
    """Quick TTS synthesis."""
    adapter = EdgeTTSAdapter()
    return adapter.synthesize(text, output_name, style)
