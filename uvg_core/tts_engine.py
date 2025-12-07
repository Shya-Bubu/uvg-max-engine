"""
UVG MAX TTS Engine Module

Azure TTS with word-level timing, voice styles, and retries.
Normalizes to -14 LUFS for consistent loudness.
"""

import os
import logging
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Word-level timing from TTS."""
    word: str
    start_ms: int
    end_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    success: bool
    text: str
    audio_path: str
    duration_ms: int
    word_timings: List[WordTiming] = field(default_factory=list)
    voice_style: str = "calm"
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "audio_path": self.audio_path,
            "duration_ms": self.duration_ms,
            "word_timings": [w.to_dict() for w in self.word_timings],
            "voice_style": self.voice_style,
            "error": self.error,
        }


# =============================================================================
# AZURE VOICE CONFIGURATIONS
# =============================================================================

AZURE_VOICES = {
    "en-US": {
        "male": "en-US-GuyNeural",
        "female": "en-US-JennyNeural",
        "narrator": "en-US-DavisNeural",
    },
    "en-GB": {
        "male": "en-GB-RyanNeural",
        "female": "en-GB-SoniaNeural",
    },
}

VOICE_STYLES = {
    "calm": {"style": "calm", "pitch": "-2%", "rate": "-5%"},
    "energetic": {"style": "cheerful", "pitch": "+5%", "rate": "+10%"},
    "serious": {"style": "newscast-formal", "pitch": "-2%", "rate": "-5%"},
    "inspirational": {"style": "empathetic", "pitch": "+3%", "rate": "-3%"},
    "dramatic": {"style": "narration-professional", "pitch": "0%", "rate": "-5%"},
}


class TTSEngine:
    """
    Azure TTS engine with word-level timing.
    
    Features:
    - Word-level timing extraction
    - Voice style per scene
    - 5 retries with exponential backoff
    - Loudness normalization to -14 LUFS
    """
    
    MAX_RETRIES = 5
    TARGET_LUFS = -14.0
    
    def __init__(self,
                 azure_key: str = "",
                 azure_region: str = "",
                 output_dir: Optional[Path] = None,
                 voice_locale: str = "en-US",
                 voice_gender: str = "male"):
        """
        Initialize TTS engine.
        
        Args:
            azure_key: Azure Speech API key
            azure_region: Azure region
            output_dir: Directory for audio files
            voice_locale: Voice locale (e.g., en-US)
            voice_gender: male, female, or narrator
        """
        self.azure_key = azure_key or os.getenv("AZURE_TTS_KEY", "")
        self.azure_region = azure_region or os.getenv("AZURE_TTS_REGION", "")
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/audio")
        self.voice_locale = voice_locale
        self.voice_gender = voice_gender
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-loaded SDK
        self._speech_config = None
        self._sdk_available = False
        
        # Verify availability
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Azure SDK is available."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            self._sdk_available = True
            return True
        except ImportError:
            logger.warning("azure-cognitiveservices-speech not installed")
            return False
    
    def _get_speech_config(self):
        """Get or create speech config."""
        if self._speech_config is not None:
            return self._speech_config
        
        if not self._sdk_available:
            return None
        
        if not self.azure_key or not self.azure_region:
            logger.warning("Azure TTS credentials not configured")
            return None
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_key,
                region=self.azure_region
            )
            # Request word-level timing
            self._speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_RequestWordLevelTimestamps,
                "true"
            )
            
            return self._speech_config
            
        except Exception as e:
            logger.error(f"Failed to create speech config: {e}")
            return None
    
    def _get_voice_name(self) -> str:
        """Get voice name based on locale and gender."""
        voices = AZURE_VOICES.get(self.voice_locale, AZURE_VOICES["en-US"])
        return voices.get(self.voice_gender, voices["male"])
    
    def _build_ssml(self, 
                    text: str,
                    voice_style: str = "calm",
                    pitch: str = "0%",
                    rate: str = "0%") -> str:
        """Build SSML for styled speech."""
        voice_name = self._get_voice_name()
        
        # Get style config
        style_config = VOICE_STYLES.get(voice_style, VOICE_STYLES["calm"])
        actual_style = style_config.get("style", "calm")
        pitch = style_config.get("pitch", pitch)
        rate = style_config.get("rate", rate)
        
        ssml = f"""
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{self.voice_locale}">
    <voice name="{voice_name}">
        <mstts:express-as style="{actual_style}">
            <prosody pitch="{pitch}" rate="{rate}">
                {text}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>
""".strip()
        
        return ssml
    
    def _normalize_loudness(self, audio_path: str) -> bool:
        """Normalize audio to target LUFS."""
        import subprocess
        
        temp_path = audio_path + ".tmp.wav"
        
        try:
            # Measure current loudness
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-af", f"loudnorm=I={self.TARGET_LUFS}:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Apply loudnorm
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", f"loudnorm=I={self.TARGET_LUFS}:TP=-1.5:LRA=11",
                "-ar", "44100",
                "-c:a", "pcm_s16le",
                temp_path
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            # Replace original
            Path(temp_path).replace(audio_path)
            return True
            
        except Exception as e:
            logger.debug(f"Loudness normalization failed: {e}")
            # Clean up temp file if exists
            Path(temp_path).unlink(missing_ok=True)
            return False
    
    def synthesize(self,
                   text: str,
                   scene_idx: int = 0,
                   voice_style: str = "calm",
                   output_name: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech with word-level timing.
        
        Args:
            text: Text to synthesize
            scene_idx: Scene index for filename
            voice_style: Voice style preset
            output_name: Optional output filename
            
        Returns:
            TTSResult with audio path and timings
        """
        if not text.strip():
            return TTSResult(
                success=False,
                text=text,
                audio_path="",
                duration_ms=0,
                error="Empty text"
            )
        
        speech_config = self._get_speech_config()
        
        if speech_config is None:
            # Fallback to offline TTS
            return self._fallback_synthesize(text, scene_idx, voice_style)
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Build SSML
            ssml = self._build_ssml(text, voice_style)
            
            # Output path
            if output_name:
                audio_path = self.output_dir / output_name
            else:
                audio_path = self.output_dir / f"tts_scene_{scene_idx}.wav"
            
            # Audio config
            audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_path))
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Collect word timings
            word_timings = []
            
            def word_boundary_cb(evt):
                if evt.boundary_type == speechsdk.SpeechSynthesisBoundaryType.Word:
                    word_timings.append(WordTiming(
                        word=evt.text,
                        start_ms=evt.audio_offset // 10000,  # Convert 100ns to ms
                        end_ms=(evt.audio_offset + evt.duration) // 10000
                    ))
            
            synthesizer.synthesis_word_boundary.connect(word_boundary_cb)
            
            # Synthesize with retries
            result = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    result = synthesizer.speak_ssml_async(ssml).get()
                    
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        break
                    elif result.reason == speechsdk.ResultReason.Canceled:
                        cancellation = result.cancellation_details
                        logger.warning(f"TTS canceled: {cancellation.reason}")
                        if attempt < self.MAX_RETRIES - 1:
                            wait = 2 ** attempt
                            time.sleep(wait)
                        
                except Exception as e:
                    logger.warning(f"TTS attempt {attempt+1} failed: {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
            
            if result is None or result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                return TTSResult(
                    success=False,
                    text=text,
                    audio_path="",
                    duration_ms=0,
                    error="Synthesis failed after retries"
                )
            
            # Get duration
            duration_ms = len(result.audio_data) / 44100 * 1000  # Assuming 44.1kHz
            
            # Normalize loudness
            self._normalize_loudness(str(audio_path))
            
            logger.info(f"Synthesized {len(text)} chars, {len(word_timings)} words, {duration_ms:.0f}ms")
            
            return TTSResult(
                success=True,
                text=text,
                audio_path=str(audio_path),
                duration_ms=int(duration_ms),
                word_timings=word_timings,
                voice_style=voice_style
            )
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                text=text,
                audio_path="",
                duration_ms=0,
                error=str(e)
            )
    
    def _fallback_synthesize(self,
                              text: str,
                              scene_idx: int,
                              voice_style: str) -> TTSResult:
        """Fallback TTS using pyttsx3 or gTTS."""
        audio_path = self.output_dir / f"tts_scene_{scene_idx}.wav"
        
        try:
            # Try pyttsx3 first (offline)
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.save_to_file(text, str(audio_path))
            engine.runAndWait()
            
            # Estimate word timings
            words = text.split()
            duration_per_word = 300  # Approximate ms per word
            timings = []
            current_time = 0
            
            for word in words:
                timings.append(WordTiming(
                    word=word,
                    start_ms=current_time,
                    end_ms=current_time + duration_per_word
                ))
                current_time += duration_per_word
            
            return TTSResult(
                success=True,
                text=text,
                audio_path=str(audio_path),
                duration_ms=current_time,
                word_timings=timings,
                voice_style=voice_style
            )
            
        except ImportError:
            pass
        
        # If all else fails
        return TTSResult(
            success=False,
            text=text,
            audio_path="",
            duration_ms=0,
            error="No TTS engine available"
        )
    
    def batch_synthesize(self,
                         texts: List[Dict[str, Any]]) -> List[TTSResult]:
        """
        Synthesize multiple texts.
        
        Args:
            texts: List of dicts with 'text', 'scene_idx', 'voice_style'
            
        Returns:
            List of TTSResults
        """
        results = []
        
        for item in texts:
            result = self.synthesize(
                text=item.get("text", ""),
                scene_idx=item.get("scene_idx", len(results)),
                voice_style=item.get("voice_style", "calm")
            )
            results.append(result)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def synthesize_text(text: str,
                    scene_idx: int = 0,
                    voice_style: str = "calm") -> TTSResult:
    """Synthesize a single text."""
    engine = TTSEngine()
    return engine.synthesize(text, scene_idx, voice_style)


def get_word_timings(tts_result: TTSResult) -> List[Dict]:
    """Extract word timings from TTS result."""
    return [w.to_dict() for w in tts_result.word_timings]
