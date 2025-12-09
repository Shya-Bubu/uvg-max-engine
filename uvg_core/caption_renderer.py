# uvg_core/caption_renderer.py
"""
Caption Renderer for UVG MAX.

Animated captions with multiple backends:
- ASS (primary)
- MoviePy (advanced)
- Manim (hook only)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CaptionStyle:
    """Caption styling configuration."""
    font_name: str = "Arial"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"  # ASS format (AABBGGRR)
    outline_color: str = "&H00000000"
    outline_width: int = 2
    shadow_depth: int = 1
    alignment: int = 2  # 2 = bottom-center
    margin_v: int = 50
    animation: str = "fade"  # fade, pop, slide


# Preset styles
CAPTION_STYLES = {
    "youtube": CaptionStyle(
        font_name="Arial Bold",
        font_size=54,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        outline_width=3,
        animation="pop"
    ),
    "tiktok": CaptionStyle(
        font_name="Impact",
        font_size=60,
        primary_color="&H00FFFFFF",
        outline_color="&H00FF0000",
        outline_width=4,
        animation="slide"
    ),
    "cinematic": CaptionStyle(
        font_name="Montserrat",
        font_size=42,
        primary_color="&H00E0E0E0",
        outline_color="&H00202020",
        outline_width=2,
        animation="fade"
    ),
    "minimal": CaptionStyle(
        font_name="Helvetica",
        font_size=36,
        primary_color="&H00FFFFFF",
        outline_width=0,
        animation="fade"
    ),
}


@dataclass
class CaptionResult:
    """Caption rendering result."""
    success: bool
    output_path: str
    backend: str
    caption_count: int
    error: str = ""


class CaptionRenderer:
    """
    Render animated captions.
    
    Backends:
    - ASS: FFmpeg-compatible .ass files
    - MoviePy: Python-rendered overlays
    - Manim: Vector animations (hook only)
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize caption renderer.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path("uvg_output/captions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def render(
        self,
        word_timings: List[Dict],
        style: str = "youtube",
        backend: str = "ass",
        output_path: str = None
    ) -> CaptionResult:
        """
        Render captions with specified backend.
        
        Args:
            word_timings: List of {word, start_ms, end_ms}
            style: Style preset name
            backend: "ass", "moviepy", or "manim"
            output_path: Output path
            
        Returns:
            CaptionResult
        """
        if backend == "ass":
            return self.render_ass(word_timings, style, output_path)
        elif backend == "moviepy":
            return self.render_moviepy(word_timings, style, output_path)
        elif backend == "manim":
            return self.render_manim(word_timings, style, output_path)
        else:
            return CaptionResult(
                success=False, output_path="", backend=backend,
                caption_count=0, error=f"Unknown backend: {backend}"
            )
    
    def render_ass(
        self,
        word_timings: List[Dict],
        style: str = "youtube",
        output_path: str = None
    ) -> CaptionResult:
        """
        Generate ASS subtitle file with animations.
        
        Args:
            word_timings: Word timing list
            style: Style preset
            output_path: Output .ass path
            
        Returns:
            CaptionResult
        """
        if output_path is None:
            output_path = str(self.output_dir / "captions.ass")
        
        style_config = CAPTION_STYLES.get(style, CAPTION_STYLES["youtube"])
        
        # Build ASS file
        ass_content = self._build_ass_header(style_config)
        
        # Group words into phrases (6-8 words per line)
        phrases = self._group_into_phrases(word_timings, max_words=6)
        
        for phrase in phrases:
            if not phrase:
                continue
            
            text = " ".join(w.get("word", "") for w in phrase)
            start = phrase[0].get("start_ms", 0)
            end = phrase[-1].get("end_ms", start + 2000)
            
            # Format times
            start_time = self._ms_to_ass_time(start)
            end_time = self._ms_to_ass_time(end)
            
            # Add animation effects
            effects = self._get_animation_effects(style_config.animation, end - start)
            
            # Add dialogue line
            ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{effects}{text}\n"
        
        # Write file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            return CaptionResult(
                success=True,
                output_path=output_path,
                backend="ass",
                caption_count=len(phrases)
            )
        except Exception as e:
            return CaptionResult(
                success=False, output_path="", backend="ass",
                caption_count=0, error=str(e)
            )
    
    def _build_ass_header(self, style: CaptionStyle) -> str:
        """Build ASS file header."""
        return f"""[Script Info]
Title: UVG MAX Captions
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{style.primary_color},&H000000FF,{style.outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow_depth},{style.alignment},10,10,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def _ms_to_ass_time(self, ms: int) -> str:
        """Convert milliseconds to ASS time format."""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        centiseconds = (ms % 1000) // 10
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    
    def _get_animation_effects(self, animation: str, duration_ms: int) -> str:
        """Get ASS animation tags."""
        if animation == "fade":
            return "{\\fad(200,200)}"
        elif animation == "pop":
            return "{\\fad(100,100)\\t(0,150,\\fscx110\\fscy110)\\t(150,300,\\fscx100\\fscy100)}"
        elif animation == "slide":
            return "{\\fad(150,150)\\move(960,600,960,540,0,200)}"
        else:
            return ""
    
    def _group_into_phrases(
        self,
        word_timings: List[Dict],
        max_words: int = 6
    ) -> List[List[Dict]]:
        """Group words into display phrases."""
        phrases = []
        current_phrase = []
        
        for word in word_timings:
            current_phrase.append(word)
            
            if len(current_phrase) >= max_words:
                phrases.append(current_phrase)
                current_phrase = []
        
        if current_phrase:
            phrases.append(current_phrase)
        
        return phrases
    
    def render_moviepy(
        self,
        word_timings: List[Dict],
        style: str = "youtube",
        output_path: str = None
    ) -> CaptionResult:
        """
        Render captions using MoviePy.
        
        Args:
            word_timings: Word timing list
            style: Style preset
            output_path: Output video path
            
        Returns:
            CaptionResult
        """
        try:
            from moviepy.editor import TextClip, CompositeVideoClip, ColorClip
        except ImportError:
            return CaptionResult(
                success=False, output_path="", backend="moviepy",
                caption_count=0, error="MoviePy not installed"
            )
        
        if output_path is None:
            output_path = str(self.output_dir / "captions_overlay.mp4")
        
        style_config = CAPTION_STYLES.get(style, CAPTION_STYLES["youtube"])
        phrases = self._group_into_phrases(word_timings, max_words=6)
        
        clips = []
        
        for phrase in phrases:
            if not phrase:
                continue
            
            text = " ".join(w.get("word", "") for w in phrase)
            start = phrase[0].get("start_ms", 0) / 1000
            end = phrase[-1].get("end_ms", start * 1000 + 2000) / 1000
            
            try:
                txt_clip = TextClip(
                    text,
                    fontsize=style_config.font_size,
                    color='white',
                    stroke_color='black',
                    stroke_width=style_config.outline_width,
                    method='caption',
                    size=(1600, None)
                )
                
                txt_clip = txt_clip.set_start(start).set_end(end)
                txt_clip = txt_clip.set_position(('center', 'bottom'))
                
                # Add fade
                txt_clip = txt_clip.crossfadein(0.2).crossfadeout(0.2)
                
                clips.append(txt_clip)
            except Exception:
                continue
        
        if not clips:
            return CaptionResult(
                success=False, output_path="", backend="moviepy",
                caption_count=0, error="No clips created"
            )
        
        try:
            # Create transparent background
            duration = max(c.end for c in clips)
            bg = ColorClip((1920, 1080), color=(0, 0, 0, 0), duration=duration)
            
            final = CompositeVideoClip([bg] + clips, size=(1920, 1080))
            final.write_videofile(output_path, fps=30, codec='libx264', audio=False)
            
            return CaptionResult(
                success=True,
                output_path=output_path,
                backend="moviepy",
                caption_count=len(phrases)
            )
        except Exception as e:
            return CaptionResult(
                success=False, output_path="", backend="moviepy",
                caption_count=0, error=str(e)
            )
    
    def render_manim(
        self,
        word_timings: List[Dict],
        style: str = "youtube",
        output_path: str = None
    ) -> CaptionResult:
        """
        Render captions using Manim (HOOK ONLY).
        
        Args:
            word_timings: Word timing list
            style: Style preset
            output_path: Output path
            
        Returns:
            CaptionResult
        """
        # Future implementation
        raise NotImplementedError("Manim caption backend coming in future update")
    
    def burn_into_video(
        self,
        video_path: str,
        ass_path: str,
        output_path: str = None
    ) -> str:
        """
        Burn ASS captions into video.
        
        Args:
            video_path: Video file
            ass_path: ASS subtitle file
            output_path: Output path
            
        Returns:
            Output video path
        """
        import subprocess
        
        if output_path is None:
            output_path = str(self.output_dir / f"captioned_{Path(video_path).name}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"ass={ass_path}",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return output_path
        except Exception:
            pass
        
        return video_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def render_captions(
    word_timings: List[Dict],
    style: str = "youtube"
) -> str:
    """Render captions and return ASS path."""
    renderer = CaptionRenderer()
    result = renderer.render_ass(word_timings, style)
    return result.output_path if result.success else ""


def get_caption_styles() -> List[str]:
    """Get available caption styles."""
    return list(CAPTION_STYLES.keys())
