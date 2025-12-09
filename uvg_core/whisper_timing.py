# uvg_core/whisper_timing.py
"""
Whisper-based word timing extraction for UVG MAX.

Extracts precise word-level timestamps from audio files.
Used for caption sync and audio ducking curves.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import whisper
try:
    import whisper
    HAVE_WHISPER = True
except ImportError:
    HAVE_WHISPER = False
    logger.debug("openai-whisper not installed - using fallback timing")


@dataclass
class WhisperWord:
    """Word with timing from Whisper."""
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
        }


@dataclass
class WhisperResult:
    """Result of Whisper transcription."""
    success: bool
    words: List[WhisperWord]
    text: str
    duration_ms: int
    model_used: str
    error: str = ""
    
    def to_word_timings(self):
        """Convert to TTSEngine WordTiming format."""
        try:
            from .tts_engine import WordTiming
            return [
                WordTiming(
                    word=w.word,
                    start_ms=w.start_ms,
                    end_ms=w.end_ms
                )
                for w in self.words
            ]
        except ImportError:
            return self.words


class WhisperTimingExtractor:
    """
    Extract word-level timings from audio using Whisper.
    
    Features:
    - Multiple model sizes (tiny, base, small, medium)
    - Word-level timestamps
    - Confidence scores
    - Fallback for when Whisper unavailable
    """
    
    MODELS = ["tiny", "base", "small", "medium", "large"]
    
    def __init__(self, model_name: str = "base", device: str = None):
        """
        Initialize Whisper extractor.
        
        Args:
            model_name: Whisper model size
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self.available = HAVE_WHISPER
        
        if not HAVE_WHISPER:
            logger.warning("Whisper not available. Install: pip install openai-whisper")
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is not None:
            return self._model
        
        if not HAVE_WHISPER:
            return None
        
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper model loaded on {self._model.device}")
            return self._model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return None
    
    def extract_timings(
        self,
        audio_path: str,
        language: str = "en"
    ) -> WhisperResult:
        """
        Extract word-level timings from audio.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code
            
        Returns:
            WhisperResult with word timings
        """
        if not Path(audio_path).exists():
            return WhisperResult(
                success=False,
                words=[],
                text="",
                duration_ms=0,
                model_used="none",
                error=f"Audio file not found: {audio_path}"
            )
        
        if not HAVE_WHISPER:
            return self._fallback_timing(audio_path)
        
        model = self._load_model()
        if model is None:
            return self._fallback_timing(audio_path)
        
        try:
            # Transcribe with word timestamps
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            
            words = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    words.append(WhisperWord(
                        word=word_info["word"].strip(),
                        start_ms=int(word_info["start"] * 1000),
                        end_ms=int(word_info["end"] * 1000),
                        confidence=word_info.get("probability", 1.0)
                    ))
            
            # Calculate duration
            duration_ms = int(result.get("segments", [{}])[-1].get("end", 0) * 1000) if result.get("segments") else 0
            
            logger.info(f"Whisper extracted {len(words)} words from {audio_path}")
            
            return WhisperResult(
                success=True,
                words=words,
                text=result.get("text", "").strip(),
                duration_ms=duration_ms,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Whisper extraction failed: {e}")
            return WhisperResult(
                success=False,
                words=[],
                text="",
                duration_ms=0,
                model_used=self.model_name,
                error=str(e)
            )
    
    def _fallback_timing(self, audio_path: str) -> WhisperResult:
        """
        Fallback timing estimation when Whisper unavailable.
        Uses audio duration and assumed word rate.
        """
        logger.warning("Using fallback timing (Whisper not available)")
        
        # Try to get audio duration
        duration_ms = self._get_audio_duration(audio_path)
        
        return WhisperResult(
            success=True,
            words=[],
            text="",
            duration_ms=duration_ms,
            model_used="fallback",
            error="Whisper not available - using duration only"
        )
    
    def _get_audio_duration(self, audio_path: str) -> int:
        """Get audio duration in milliseconds."""
        try:
            import subprocess
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return int(float(result.stdout.strip()) * 1000)
        except Exception:
            return 0
    
    def release(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model released")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_extractor = None

def get_extractor(model: str = "base") -> WhisperTimingExtractor:
    """Get or create Whisper extractor singleton."""
    global _extractor
    if _extractor is None or _extractor.model_name != model:
        _extractor = WhisperTimingExtractor(model)
    return _extractor


def extract_word_timings(audio_path: str, model: str = "base") -> List[WhisperWord]:
    """
    Extract word timings from audio file.
    
    Args:
        audio_path: Path to audio
        model: Whisper model size
        
    Returns:
        List of WhisperWord objects
    """
    extractor = get_extractor(model)
    result = extractor.extract_timings(audio_path)
    return result.words


def is_whisper_available() -> bool:
    """Check if Whisper is available."""
    return HAVE_WHISPER
