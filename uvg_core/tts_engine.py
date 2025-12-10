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


# =============================================================================
# TTS ADAPTER SYSTEM (Pluggable)
# =============================================================================

from abc import ABC, abstractmethod


class TTSAdapter(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    def synthesize(self, text: str, voice_style: str, output_path: str) -> TTSResult:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_style: Voice style preset
            output_path: Path to save audio file
            
        Returns:
            TTSResult with audio path and word timings
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this adapter is available for use."""
        pass


class MockTTSAdapter(TTSAdapter):
    """
    # MOCK — Generates silence WAV with estimated word timings.
    Used for testing without external API calls.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def synthesize(self, text: str, voice_style: str, output_path: str) -> TTSResult:
        """Generate mock audio with estimated word timings."""
        import struct
        import wave
        
        words = text.split()
        # Estimate 200ms per word
        total_duration_ms = len(words) * 200 + 500  # +500ms padding
        
        # Generate word timings
        word_timings = []
        current_ms = 100
        for word in words:
            duration = 150 + len(word) * 10  # Longer words take more time
            word_timings.append(WordTiming(
                word=word,
                start_ms=current_ms,
                end_ms=current_ms + duration
            ))
            current_ms += duration + 50  # 50ms gap between words
        
        total_duration_ms = current_ms + 100
        
        # Generate silence WAV
        try:
            sample_rate = 44100
            num_samples = int(sample_rate * total_duration_ms / 1000)
            
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                # Write silence
                silence = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))
                wav_file.writeframes(silence)
            
            logger.info(f"[MOCK] Generated mock audio: {output_path} ({total_duration_ms}ms)")
            
            return TTSResult(
                success=True,
                text=text,
                audio_path=output_path,
                duration_ms=total_duration_ms,
                word_timings=word_timings,
                voice_style=voice_style,
            )
        except Exception as e:
            return TTSResult(
                success=False,
                text=text,
                audio_path="",
                duration_ms=0,
                error=f"Mock TTS failed: {e}"
            )


class GeminiTTSAdapter(TTSAdapter):
    """
    # MOCK — Implement real call here when gemini-2.5-flash-tts available.
    
    Placeholder for Gemini TTS API integration.
    When the API becomes available, replace the mock implementation with real calls.
    """
    
    def __init__(self, api_key: str, model_name: str, output_dir: Path):
        self.api_key = api_key
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def is_available(self) -> bool:
        # MOCK: Would check if API key is valid and model exists
        return bool(self.api_key)
    
    def synthesize(self, text: str, voice_style: str, output_path: str) -> TTSResult:
        """
        # MOCK — Replace with real Gemini TTS API call when available.
        
        Expected future implementation:
        1. Call Gemini TTS API with text and voice_style
        2. Receive audio bytes and word timings
        3. Save audio to output_path
        4. Return TTSResult with word_timings
        """
        logger.warning(f"[MOCK] GeminiTTSAdapter.synthesize() - real API not yet available")
        logger.warning(f"[MOCK] Would use model: {self.model_name}")
        
        # Fall back to MockTTSAdapter behavior
        mock_adapter = MockTTSAdapter(self.output_dir)
        return mock_adapter.synthesize(text, voice_style, output_path)


class NativeAudioAdapter(TTSAdapter):
    """
    # MOCK — Placeholder for gemini-native-audio API if available.
    
    Future adapter for native audio generation capabilities.
    """
    
    def __init__(self, api_key: str, output_dir: Path):
        self.api_key = api_key
        self.output_dir = output_dir
    
    def is_available(self) -> bool:
        # MOCK: Not available yet
        return False
    
    def synthesize(self, text: str, voice_style: str, output_path: str) -> TTSResult:
        """# MOCK — Implement when gemini-native-audio becomes available."""
        return TTSResult(
            success=False,
            text=text,
            audio_path="",
            duration_ms=0,
            error="NativeAudioAdapter not yet implemented"
        )


class ElevenLabsAdapter(TTSAdapter):
    """
    ElevenLabs TTS adapter.
    
    HOOK ONLY - NOT IMPLEMENTED.
    
    Future features:
    - High-quality neural voices
    - Voice cloning support
    - Emotion styles
    
    To enable:
    1. pip install elevenlabs
    2. Set ELEVENLABS_API_KEY environment variable
    3. Uncomment implementation
    """
    
    def __init__(self, api_key: str = None, output_dir: Path = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ElevenLabsAdapter initialized (HOOK ONLY - not implemented)")
    
    def is_available(self) -> bool:
        """ElevenLabs is NOT available - hook only."""
        return False
    
    def synthesize(self, text: str, voice_style: str, output_path: str) -> TTSResult:
        """
        HOOK ONLY - Not implemented.
        
        Future implementation:
        1. Call ElevenLabs API with text
        2. Get audio stream
        3. Extract word timings (via Whisper post-processing)
        4. Save audio to output_path
        """
        raise NotImplementedError(
            "ElevenLabsAdapter is a hook for future implementation. "
            "Use AzureTTSAdapter or MockTTSAdapter instead. "
            "To implement: pip install elevenlabs and add API integration."
        )
    
    def list_voices(self) -> list:
        """List available voices (hook only)."""
        raise NotImplementedError("ElevenLabs voice listing not implemented")
    
    def clone_voice(self, samples: list) -> str:
        """Clone voice from samples (hook only)."""
        raise NotImplementedError("ElevenLabs voice cloning not implemented")


class FishSpeechAdapter(TTSAdapter):
    """
    Fish-Speech S1 TTS adapter.
    
    Primary TTS engine for UVG MAX. Features:
    - 50+ emotion markers
    - Natural prosody
    - Zero API cost (local model)
    - Automatic word timestamps via Whisper
    
    Uses uvg_core.fish_speech_adapter for actual synthesis.
    """
    
    def __init__(self, output_dir: Path = None, model_path: str = None):
        """
        Initialize Fish-Speech adapter.
        
        Args:
            output_dir: Output directory for audio files
            model_path: Path to Fish-Speech model (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self._adapter = None
        
        logger.info("FishSpeechAdapter initialized")
    
    def _get_adapter(self):
        """Lazy load the actual Fish-Speech adapter."""
        if self._adapter is None:
            try:
                from uvg_core.fish_speech_adapter import FishSpeechTTS
                self._adapter = FishSpeechTTS(output_dir=self.output_dir)
            except ImportError:
                logger.warning("fish_speech_adapter not available, using mock")
                return None
        return self._adapter
    
    def is_available(self) -> bool:
        """Check if Fish-Speech is available."""
        adapter = self._get_adapter()
        return adapter is not None and adapter.is_available()
    
    def synthesize(self, text: str, voice_style: str, output_path: str) -> TTSResult:
        """
        Synthesize speech with Fish-Speech S1.
        
        Args:
            text: Text to synthesize
            voice_style: Voice preset (documentary, motivational, etc.)
            output_path: Path to save audio
            
        Returns:
            TTSResult with audio path and word timings
        """
        adapter = self._get_adapter()
        
        if adapter is None:
            # Fall back to mock
            logger.warning("Fish-Speech not available, using mock synthesis")
            return TTSResult(
                success=False,
                text=text,
                audio_path="",
                duration_ms=0,
                error="Fish-Speech adapter not available"
            )
        
        try:
            # Use the actual Fish-Speech adapter
            result = adapter.synthesize(text=text, voice_style=voice_style)
            
            if result.success:
                # Convert FishSpeechResult to TTSResult
                word_timings = [
                    WordTiming(word=w["word"], start_ms=w["start_ms"], end_ms=w["end_ms"])
                    for w in result.word_timings
                ]
                
                return TTSResult(
                    success=True,
                    text=text,
                    audio_path=result.audio_path,
                    duration_ms=result.duration_ms,
                    word_timings=word_timings,
                    voice_style=voice_style
                )
            else:
                return TTSResult(
                    success=False,
                    text=text,
                    audio_path="",
                    duration_ms=0,
                    error=result.error
                )
                
        except Exception as e:
            logger.error(f"Fish-Speech synthesis failed: {e}")
            return TTSResult(
                success=False,
                text=text,
                audio_path="",
                duration_ms=0,
                error=str(e)
            )


class AzureTTSAdapter(TTSAdapter):
    """
    Azure Cognitive Services TTS adapter.
    
    Features:
    - SSML support with emotion styles
    - Word-level timestamps via Speech SDK
    - Professional neural voices
    - Rate and pitch control
    """
    
    # Voice mapping
    VOICE_MAP = {
        "en-US": "en-US-JennyNeural",

        "en-GB": "en-GB-SoniaNeural",
        "default": "en-US-JennyNeural",
    }
    
    # Style mapping
    STYLE_MAP = {
        "calm": "calm",
        "cheerful": "cheerful",
        "excited": "excited",
        "friendly": "friendly",
        "hopeful": "hopeful",
        "sad": "sad",
        "angry": "angry",
        "fearful": "fearful",
        "gentle": "gentle",
        "serious": "serious",
        "neutral": "neutral",
        "empathetic": "empathetic",
    }
    
    def __init__(
        self,
        subscription_key: str,
        region: str = "eastus",
        output_dir: Path = None,
        voice_name: str = None,
        locale: str = "en-US"
    ):
        """
        Initialize Azure TTS adapter.
        
        Args:
            subscription_key: Azure Speech subscription key
            region: Azure region (e.g., "eastus")
            output_dir: Output directory for audio files
            voice_name: Voice name override
            locale: Language locale
        """
        self.subscription_key = subscription_key
        self.region = region
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.locale = locale
        self.voice_name = voice_name or self.VOICE_MAP.get(locale, self.VOICE_MAP["default"])
        
        self._sdk_available = False
        self._speech_config = None
        self._init_sdk()
    
    def _init_sdk(self):
        """Initialize Azure Speech SDK."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            self._speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
            )
            self._sdk_available = True
            logger.info(f"Azure Speech SDK initialized (voice={self.voice_name})")
            
        except ImportError:
            logger.warning("Azure Speech SDK not installed. Install with: pip install azure-cognitiveservices-speech")
            self._sdk_available = False
        except Exception as e:
            logger.error(f"Azure SDK init failed: {e}")
            self._sdk_available = False
    
    def is_available(self) -> bool:
        """Check if Azure TTS is available."""
        return self._sdk_available and bool(self.subscription_key)
    
    def _build_ssml(
        self,
        text: str,
        voice_style: str = "neutral",
        rate: str = "+0%",
        pitch: str = "+0%"
    ) -> str:
        """
        Build SSML markup for speech synthesis.
        
        Args:
            text: Text to synthesize
            voice_style: Emotion style
            rate: Speaking rate adjustment
            pitch: Pitch adjustment
            
        Returns:
            SSML string
        """
        # Map style
        style = self.STYLE_MAP.get(voice_style, "neutral")
        
        # Build SSML
        ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="http://www.w3.org/2001/mstts"
    xml:lang="{self.locale}">
  <voice name="{self.voice_name}">
    <mstts:express-as style="{style}">
      <prosody rate="{rate}" pitch="{pitch}">
        {text}
      </prosody>
    </mstts:express-as>
  </voice>
</speak>'''
        
        return ssml
    
    def synthesize(
        self,
        text: str,
        voice_style: str = "neutral",
        output_path: str = None
    ) -> TTSResult:
        """
        Synthesize speech with Azure TTS.
        
        Args:
            text: Text to synthesize
            voice_style: Emotion style
            output_path: Output audio file path
            
        Returns:
            TTSResult with audio path and word timings
        """
        if not self.is_available():
            logger.warning("Azure TTS not available, falling back to mock")
            return MockTTSAdapter(self.output_dir).synthesize(text, voice_style, output_path)
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            if output_path is None:
                output_path = str(self.output_dir / f"azure_tts_{hash(text) % 10000}.wav")
            
            # Build SSML
            ssml = self._build_ssml(text, voice_style)
            logger.debug(f"Azure TTS SSML: {ssml[:200]}...")
            
            # Set up audio config
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config,
                audio_config=audio_config
            )
            
            # Collect word timings
            word_timings = []
            
            def word_boundary_callback(evt):
                """Callback for word boundary events."""
                word_timings.append(WordTiming(
                    word=evt.text,
                    start_ms=int(evt.audio_offset / 10000),  # 100ns to ms
                    end_ms=int((evt.audio_offset + evt.duration) / 10000)
                ))
            
            synthesizer.synthesis_word_boundary.connect(word_boundary_callback)
            
            # Synthesize with SSML
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                duration_ms = int(result.audio_duration.total_seconds() * 1000)
                
                logger.info(f"Azure TTS: synthesized {len(text)} chars, {duration_ms}ms, {len(word_timings)} words")
                
                return TTSResult(
                    success=True,
                    text=text,
                    audio_path=output_path,
                    duration_ms=duration_ms,
                    word_timings=word_timings,
                    voice_style=voice_style
                )
            else:
                error_msg = f"Azure TTS failed: {result.reason}"
                if result.cancellation_details:
                    error_msg += f" - {result.cancellation_details.reason}"
                
                logger.error(error_msg)
                return TTSResult(
                    success=False,
                    text=text,
                    audio_path="",
                    duration_ms=0,
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            return TTSResult(
                success=False,
                text=text,
                audio_path="",
                duration_ms=0,
                error=str(e)
            )


class TTSEngine:
    """
    TTS engine with pluggable adapter system.
    
    Features:
    - Pluggable adapters: mock, gemini, azure
    - Word-level timing extraction
    - Voice style per scene
    - 5 retries with exponential backoff
    - Loudness normalization to -14 LUFS
    
    Adapter selection:
    - UVG_MOCK_MODE=true → MockTTSAdapter
    - UVG_TTS_PROVIDER=gemini + key → GeminiTTSAdapter
    - UVG_TTS_PROVIDER=azure + key → AzureTTSAdapter (built-in)
    - Default → MockTTSAdapter
    """
    
    MAX_RETRIES = 5
    TARGET_LUFS = -14.0
    
    def __init__(self,
                 azure_key: str = "",
                 azure_region: str = "",
                 output_dir: Optional[Path] = None,
                 voice_locale: str = "en-US",
                 voice_gender: str = "male",
                 mock_mode: bool = False,
                 tts_provider: str = ""):
        """
        Initialize TTS engine.
        
        Args:
            azure_key: Azure Speech API key
            azure_region: Azure region
            output_dir: Directory for audio files
            voice_locale: Voice locale (e.g., en-US)
            voice_gender: male, female, or narrator
            mock_mode: Use mock TTS (generates silence with timings)
            tts_provider: mock, gemini, or azure
        """
        self.azure_key = azure_key or os.getenv("AZURE_TTS_KEY", "")
        self.azure_region = azure_region or os.getenv("AZURE_TTS_REGION", "")
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/audio")
        self.voice_locale = voice_locale
        self.voice_gender = voice_gender
        self.mock_mode = mock_mode or os.getenv("UVG_MOCK_MODE", "true").lower() == "true"
        self.tts_provider = tts_provider or os.getenv("UVG_TTS_PROVIDER", "mock")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-loaded SDK
        self._speech_config = None
        self._sdk_available = False
        
        # Select adapter
        self._adapter = self._select_adapter()
        
        # Verify Azure availability if needed
        if self.tts_provider == "azure" and not self.mock_mode:
            self._check_availability()
    
    def _select_adapter(self) -> TTSAdapter:
        """Select appropriate TTS adapter based on config."""
        # Mock mode always uses MockTTSAdapter
        if self.mock_mode:
            logger.info("Using MockTTSAdapter (UVG_MOCK_MODE=true)")
            return MockTTSAdapter(self.output_dir)
        
        # Select by provider
        if self.tts_provider == "gemini":
            gemini_tts_key = os.getenv("UVG_GEMINI_TTS_KEY", "") or os.getenv("GEMINI_API_KEY", "")
            model_name = os.getenv("UVG_GEMINI_TTS_MODEL", "gemini-2.5-flash-tts")
            if gemini_tts_key:
                logger.info(f"Using GeminiTTSAdapter with model {model_name}")
                return GeminiTTSAdapter(gemini_tts_key, model_name, self.output_dir)
            else:
                logger.warning("No Gemini TTS key found, falling back to MockTTSAdapter")
                return MockTTSAdapter(self.output_dir)
        
        elif self.tts_provider == "azure":
            if self.azure_key and self.azure_region:
                logger.info("Using Azure TTS (built-in adapter)")
                return None  # Use built-in Azure implementation
            else:
                logger.warning("No Azure TTS credentials, falling back to MockTTSAdapter")
                return MockTTSAdapter(self.output_dir)
        
        # Default: mock
        logger.info("Using MockTTSAdapter (default)")
        return MockTTSAdapter(self.output_dir)
    
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
