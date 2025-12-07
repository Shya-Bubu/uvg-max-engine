"""
UVG MAX Subtitle Engine Module

Full subtitle intelligence with word-level timing, smart placement,
auto font sizing, and style presets.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class CaptionStyle(Enum):
    """Caption style presets."""
    TIKTOK = "tiktok"       # Bold, pop, large
    INSTAGRAM = "instagram" # Clean, minimal
    YOUTUBE = "youtube"     # Cinematic, elegant
    MINIMAL = "minimal"     # Subtle, small
    BOLD = "bold"           # Maximum impact
    ELEGANT = "elegant"     # Script-like


@dataclass
class Subtitle:
    """A single subtitle entry."""
    start_ms: int
    end_ms: int
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)  # Word-level timings
    style: str = "youtube"
    position: Tuple[int, int] = (0, 0)  # x, y in pixels
    font_size: int = 48
    highlight_word_idx: int = -1  # Current highlighted word
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "words": self.words,
            "style": self.style,
            "position": self.position,
            "font_size": self.font_size,
        }
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


# =============================================================================
# STYLE CONFIGURATIONS
# =============================================================================

STYLE_CONFIGS = {
    "tiktok": {
        "font_family": "Impact",
        "font_weight": "bold",
        "font_size_base": 72,
        "color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 4,
        "shadow": True,
        "shadow_offset": 3,
        "background": None,
        "animation": "pop_bounce",
        "max_chars_per_line": 20,
        "max_lines": 2,
        "position_y_ratio": 0.7,  # 70% from top
    },
    "instagram": {
        "font_family": "Helvetica Neue",
        "font_weight": "medium",
        "font_size_base": 54,
        "color": "#FFFFFF",
        "stroke_color": "#333333",
        "stroke_width": 2,
        "shadow": False,
        "background": None,
        "animation": "fade_slide",
        "max_chars_per_line": 30,
        "max_lines": 2,
        "position_y_ratio": 0.75,
    },
    "youtube": {
        "font_family": "Montserrat",
        "font_weight": "semibold",
        "font_size_base": 56,
        "color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 3,
        "shadow": True,
        "shadow_offset": 2,
        "background": "rgba(0,0,0,0.4)",
        "animation": "fade_slide",
        "max_chars_per_line": 35,
        "max_lines": 2,
        "position_y_ratio": 0.85,
    },
    "minimal": {
        "font_family": "Arial",
        "font_weight": "normal",
        "font_size_base": 42,
        "color": "#FFFFFF",
        "stroke_color": "#666666",
        "stroke_width": 1,
        "shadow": False,
        "background": None,
        "animation": "fade",
        "max_chars_per_line": 40,
        "max_lines": 2,
        "position_y_ratio": 0.9,
    },
    "bold": {
        "font_family": "Anton",
        "font_weight": "bold",
        "font_size_base": 80,
        "color": "#FFFF00",  # Yellow
        "stroke_color": "#000000",
        "stroke_width": 5,
        "shadow": True,
        "shadow_offset": 4,
        "background": None,
        "animation": "scale_bounce",
        "max_chars_per_line": 15,
        "max_lines": 2,
        "position_y_ratio": 0.65,
    }, 
    "elegant": {
        "font_family": "Playfair Display",
        "font_weight": "normal",
        "font_size_base": 50,
        "color": "#FFFFFF",
        "stroke_color": "#222222",
        "stroke_width": 2,
        "shadow": True,
        "shadow_offset": 1,
        "background": None,
        "animation": "fade",
        "max_chars_per_line": 35,
        "max_lines": 2,
        "position_y_ratio": 0.8,
    },
}


class SubtitleEngine:
    """
    Full subtitle intelligence engine.
    
    Features:
    - Word-level timing from TTS
    - Auto line-breaking (max 2 lines, ~12-35 words based on style)
    - Duration extension for long sentences
    - Smart face-safe placement
    - Auto font size based on text length
    - Dark-mode auto-contrast
    - Style presets
    """
    
    MIN_SUBTITLE_DURATION = 1000  # 1 second minimum
    MIN_WORD_DURATION = 200       # 200ms per word minimum
    
    def __init__(self,
                 video_width: int = 1080,
                 video_height: int = 1920,
                 default_style: str = "youtube"):
        """
        Initialize subtitle engine.
        
        Args:
            video_width: Video width in pixels
            video_height: Video height in pixels
            default_style: Default caption style
        """
        self.video_width = video_width
        self.video_height = video_height
        self.default_style = default_style
    
    def generate_subtitles(self,
                           tts_result: Any,  # TTSResult
                           style: Optional[str] = None) -> List[Subtitle]:
        """
        Generate subtitles from TTS result.
        
        Args:
            tts_result: TTS result with word timings
            style: Caption style preset
            
        Returns:
            List of Subtitle objects
        """
        style = style or self.default_style
        style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["youtube"])
        
        word_timings = tts_result.word_timings if hasattr(tts_result, 'word_timings') else []
        
        if not word_timings:
            # Fallback: single subtitle for entire text
            text = tts_result.text if hasattr(tts_result, 'text') else ""
            duration_ms = tts_result.duration_ms if hasattr(tts_result, 'duration_ms') else 3000
            
            return [Subtitle(
                start_ms=0,
                end_ms=duration_ms,
                text=text,
                style=style,
                font_size=self._auto_font_size(text, style_config),
                position=self._calculate_position(style_config, []),
            )]
        
        # Group words into subtitle chunks
        chunks = self._chunk_words(word_timings, style_config)
        
        subtitles = []
        for chunk in chunks:
            words = chunk["words"]
            text = " ".join(w.word if hasattr(w, 'word') else w.get("word", "") for w in words)
            
            # Get timing
            start_ms = words[0].start_ms if hasattr(words[0], 'start_ms') else words[0].get("start_ms", 0)
            end_ms = words[-1].end_ms if hasattr(words[-1], 'end_ms') else words[-1].get("end_ms", start_ms + 1000)
            
            # Extend short durations
            if end_ms - start_ms < self.MIN_SUBTITLE_DURATION:
                end_ms = start_ms + self.MIN_SUBTITLE_DURATION
            
            # Build word timings for this chunk
            word_list = []
            for w in words:
                word_list.append({
                    "word": w.word if hasattr(w, 'word') else w.get("word", ""),
                    "start_ms": w.start_ms if hasattr(w, 'start_ms') else w.get("start_ms", 0),
                    "end_ms": w.end_ms if hasattr(w, 'end_ms') else w.get("end_ms", 0),
                })
            
            subtitle = Subtitle(
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                words=word_list,
                style=style,
                font_size=self._auto_font_size(text, style_config),
                position=self._calculate_position(style_config, []),
            )
            
            subtitles.append(subtitle)
        
        return subtitles
    
    def _chunk_words(self,
                     word_timings: List[Any],
                     style_config: Dict) -> List[Dict]:
        """Split word timings into displayable chunks."""
        max_chars = style_config.get("max_chars_per_line", 30) * style_config.get("max_lines", 2)
        
        chunks = []
        current_chunk = {"words": [], "char_count": 0}
        
        for word in word_timings:
            word_text = word.word if hasattr(word, 'word') else word.get("word", "")
            word_len = len(word_text)
            
            # Check if adding this word exceeds limit
            if current_chunk["char_count"] + word_len + 1 > max_chars and current_chunk["words"]:
                # Save current chunk and start new
                chunks.append(current_chunk)
                current_chunk = {"words": [], "char_count": 0}
            
            current_chunk["words"].append(word)
            current_chunk["char_count"] += word_len + 1  # +1 for space
        
        # Add final chunk
        if current_chunk["words"]:
            chunks.append(current_chunk)
        
        return chunks
    
    def _auto_font_size(self, text: str, style_config: Dict) -> int:
        """Calculate optimal font size based on text length."""
        base_size = style_config.get("font_size_base", 48)
        max_chars = style_config.get("max_chars_per_line", 30)
        
        text_len = len(text)
        
        if text_len <= max_chars // 2:
            # Short text: can be larger
            return int(base_size * 1.2)
        elif text_len <= max_chars:
            # Normal text
            return base_size
        elif text_len <= max_chars * 1.5:
            # Longer text: slightly smaller
            return int(base_size * 0.85)
        else:
            # Very long text: smaller
            return int(base_size * 0.7)
    
    def _calculate_position(self,
                             style_config: Dict,
                             face_boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        """
        Calculate subtitle position, avoiding faces.
        
        Args:
            style_config: Style configuration
            face_boxes: List of face bounding boxes (x, y, w, h)
            
        Returns:
            (x, y) position
        """
        y_ratio = style_config.get("position_y_ratio", 0.8)
        
        # Default position
        x = self.video_width // 2
        y = int(self.video_height * y_ratio)
        
        # Check for face overlap
        if face_boxes:
            # Get face regions
            for (fx, fy, fw, fh) in face_boxes:
                face_bottom = fy + fh
                
                # If subtitle would overlap with face, move it
                if abs(y - face_bottom) < 100:
                    # Try above face
                    if fy > self.video_height * 0.3:
                        y = int(fy - 50)
                    # Or move further down
                    else:
                        y = min(int(self.video_height * 0.9), face_bottom + 100)
        
        return (x, y)
    
    def smart_placement(self,
                        frame: Any,
                        face_boxes: List[Tuple[int, int, int, int]],
                        style: str = "youtube") -> Tuple[int, int]:
        """
        Calculate smart subtitle placement avoiding faces.
        
        Args:
            frame: Video frame (for brightness analysis)
            face_boxes: Detected face bounding boxes
            style: Caption style
            
        Returns:
            (x, y) optimal position
        """
        style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["youtube"])
        return self._calculate_position(style_config, face_boxes)
    
    def auto_contrast(self, background_brightness: float) -> Dict[str, str]:
        """
        Determine caption colors based on background brightness.
        
        Args:
            background_brightness: 0-1, where 1 is white
            
        Returns:
            Dict with text_color and stroke_color
        """
        if background_brightness > 0.6:
            # Light background: dark text
            return {
                "text_color": "#000000",
                "stroke_color": "#FFFFFF",
                "mode": "dark_text"
            }
        elif background_brightness < 0.3:
            # Dark background: light text (default)
            return {
                "text_color": "#FFFFFF",
                "stroke_color": "#000000",
                "mode": "light_text"
            }
        else:
            # Medium: high contrast white with strong stroke
            return {
                "text_color": "#FFFFFF",
                "stroke_color": "#000000",
                "stroke_width_multiplier": 1.5,
                "mode": "high_contrast"
            }
    
    def line_break(self, text: str, style: str = "youtube") -> List[str]:
        """
        Auto line-break text for display.
        
        Args:
            text: Text to break
            style: Caption style
            
        Returns:
            List of lines
        """
        style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["youtube"])
        max_chars = style_config.get("max_chars_per_line", 30)
        max_lines = style_config.get("max_lines", 2)
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                
                if len(lines) >= max_lines:
                    # Merge remaining words into last line
                    remaining_idx = words.index(word)
                    lines[-1] = lines[-1] + " " + " ".join(words[remaining_idx:])
                    return lines
        
        if current_line:
            lines.append(current_line)
        
        return lines[:max_lines]
    
    def to_srt(self, subtitles: List[Subtitle], output_path: Path) -> None:
        """Export subtitles to SRT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subtitles, 1):
                start = self._ms_to_srt_time(sub.start_ms)
                end = self._ms_to_srt_time(sub.end_ms)
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{sub.text}\n\n")
    
    def _ms_to_srt_time(self, ms: int) -> str:
        """Convert milliseconds to SRT timestamp."""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def to_ass(self, subtitles: List[Subtitle], output_path: Path) -> None:
        """Export subtitles to ASS format with styling."""
        style_config = STYLE_CONFIGS.get(self.default_style, STYLE_CONFIGS["youtube"])
        
        header = f"""[Script Info]
Title: UVG MAX Subtitles
ScriptType: v4.00+
PlayResX: {self.video_width}
PlayResY: {self.video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style_config['font_family']},{style_config['font_size_base']},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            
            for sub in subtitles:
                start = self._ms_to_ass_time(sub.start_ms)
                end = self._ms_to_ass_time(sub.end_ms)
                text = sub.text.replace('\n', '\\N')
                
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
    
    def _ms_to_ass_time(self, ms: int) -> str:
        """Convert milliseconds to ASS timestamp."""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        centiseconds = (ms % 1000) // 10
        
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_subtitles(tts_result: Any,
                       style: str = "youtube") -> List[Subtitle]:
    """Generate subtitles from TTS result."""
    engine = SubtitleEngine()
    return engine.generate_subtitles(tts_result, style)


def export_srt(subtitles: List[Subtitle], path: Path) -> None:
    """Export subtitles to SRT file."""
    engine = SubtitleEngine()
    engine.to_srt(subtitles, path)
