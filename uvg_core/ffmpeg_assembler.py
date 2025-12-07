"""
UVG MAX FFmpeg Assembler Module

Final video assembly with GPU/CPU paths.
"""

import logging
import subprocess
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AssemblyScene:
    """A scene for assembly."""
    index: int
    video_path: str
    audio_path: str
    duration: float
    transition_type: str = "fade"
    transition_duration: float = 0.5


@dataclass
class AssemblyResult:
    """Result of video assembly."""
    success: bool
    output_path: str
    duration: float
    file_size_mb: float
    error: str = ""


class FFmpegAssembler:
    """
    Final video assembly using FFmpeg.
    
    Features:
    - GPU encoding (NVENC) with CPU fallback
    - xfade transitions
    - Audio/video sync
    - Subtitle burning
    """
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 target_width: int = 1080,
                 target_height: int = 1920,
                 fps: int = 30,
                 use_cuda: bool = True):
        """
        Initialize assembler.
        
        Args:
            output_dir: Output directory
            target_width: Output width
            target_height: Output height
            fps: Output framerate
            use_cuda: Use GPU encoding
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output/final")
        self.target_width = target_width
        self.target_height = target_height
        self.fps = fps
        self.use_cuda = use_cuda
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check NVENC availability
        self._nvenc_available = self._check_nvenc()
    
    def _check_nvenc(self) -> bool:
        """Check if NVENC is available."""
        if not self.use_cuda:
            return False
        
        try:
            result = subprocess.run([
                "ffmpeg", "-hide_banner", "-encoders"
            ], capture_output=True, text=True, timeout=10)
            
            return "h264_nvenc" in result.stdout
        except Exception:
            return False
    
    def _get_encoder_settings(self) -> List[str]:
        """Get encoder settings based on availability."""
        if self._nvenc_available:
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-b:v", "8M",
                "-maxrate", "10M",
                "-bufsize", "16M",
            ]
        else:
            return [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
            ]
    
    def _run_ffmpeg(self, cmd: List[str], timeout: int = 600) -> tuple:
        """Run FFmpeg command."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def create_concat_file(self, scenes: List[AssemblyScene]) -> Path:
        """Create FFmpeg concat demuxer file."""
        concat_path = self.output_dir / "concat_list.txt"
        
        with open(concat_path, 'w') as f:
            for scene in scenes:
                video_path = Path(scene.video_path).absolute()
                f.write(f"file '{video_path}'\n")
        
        return concat_path
    
    def assemble_simple(self,
                         scenes: List[AssemblyScene],
                         output_name: str = "final.mp4") -> AssemblyResult:
        """
        Simple concatenation without transitions.
        
        Args:
            scenes: List of scenes to assemble
            output_name: Output filename
            
        Returns:
            AssemblyResult
        """
        output_path = self.output_dir / output_name
        
        # Create concat file
        concat_path = self.create_concat_file(scenes)
        
        # Build command
        encoder_settings = self._get_encoder_settings()
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_path),
            *encoder_settings,
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        success, error = self._run_ffmpeg(cmd)
        
        if not success and self._nvenc_available:
            # Fallback to CPU
            logger.warning("NVENC failed, falling back to CPU encoding")
            self._nvenc_available = False
            encoder_settings = self._get_encoder_settings()
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0", 
                "-i", str(concat_path),
                *encoder_settings,
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_path)
            ]
            
            success, error = self._run_ffmpeg(cmd)
        
        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            duration = sum(s.duration for s in scenes)
            
            return AssemblyResult(
                success=True,
                output_path=str(output_path),
                duration=duration,
                file_size_mb=file_size
            )
        
        return AssemblyResult(
            success=False,
            output_path="",
            duration=0,
            file_size_mb=0,
            error=error[:500]
        )
    
    def assemble_with_transitions(self,
                                   scenes: List[AssemblyScene],
                                   output_name: str = "final.mp4") -> AssemblyResult:
        """
        Assemble with xfade transitions.
        
        Args:
            scenes: List of scenes
            output_name: Output filename
            
        Returns:
            AssemblyResult
        """
        if len(scenes) < 2:
            return self.assemble_simple(scenes, output_name)
        
        output_path = self.output_dir / output_name
        
        # Build inputs
        inputs = []
        for scene in scenes:
            inputs.extend(["-i", scene.video_path])
        
        # Build xfade filter chain
        filter_parts = []
        current_offset = 0
        
        for i in range(len(scenes) - 1):
            trans_dur = scenes[i].transition_duration
            clip_dur = scenes[i].duration
            
            if i == 0:
                input_a = "[0:v]"
            else:
                input_a = f"[v{i}]"
            
            input_b = f"[{i+1}:v]"
            offset = current_offset + clip_dur - trans_dur
            
            output_label = f"[v{i+1}]" if i < len(scenes) - 2 else "[outv]"
            
            filter_parts.append(
                f"{input_a}{input_b}xfade=transition={scenes[i].transition_type}:"
                f"duration={trans_dur}:offset={offset}{output_label}"
            )
            
            current_offset = offset
        
        # Audio concat
        audio_filter = "".join(f"[{i}:a]" for i in range(len(scenes)))
        audio_filter += f"concat=n={len(scenes)}:v=0:a=1[outa]"
        
        filter_complex = ";".join(filter_parts) + ";" + audio_filter
        
        encoder_settings = self._get_encoder_settings()
        
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            *encoder_settings,
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        success, error = self._run_ffmpeg(cmd, timeout=900)
        
        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            duration = current_offset + scenes[-1].duration
            
            return AssemblyResult(
                success=True,
                output_path=str(output_path),
                duration=duration,
                file_size_mb=file_size
            )
        
        # Fallback to simple concat
        logger.warning("Transition assembly failed, using simple concat")
        return self.assemble_simple(scenes, output_name)
    
    def add_subtitles(self,
                       video_path: str,
                       subtitle_path: str,
                       output_path: Optional[str] = None) -> str:
        """
        Burn subtitles into video.
        
        Args:
            video_path: Input video
            subtitle_path: SRT or ASS subtitle file
            output_path: Output path
            
        Returns:
            Output path
        """
        if output_path is None:
            output_path = self.output_dir / f"subtitled_{Path(video_path).stem}.mp4"
        
        # Determine subtitle filter based on extension
        sub_ext = Path(subtitle_path).suffix.lower()
        if sub_ext == ".ass":
            sub_filter = f"ass={subtitle_path}"
        else:
            sub_filter = f"subtitles={subtitle_path}"
        
        encoder_settings = self._get_encoder_settings()
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", sub_filter,
            *encoder_settings,
            "-c:a", "copy",
            str(output_path)
        ]
        
        success, _ = self._run_ffmpeg(cmd)
        
        if success:
            return str(output_path)
        return video_path
    
    def add_background_music(self,
                              video_path: str,
                              music_path: str,
                              music_volume: float = 0.2,
                              output_path: Optional[str] = None) -> str:
        """
        Add background music to video.
        
        Args:
            video_path: Input video
            music_path: Background music
            music_volume: Music volume (0-1)
            output_path: Output path
            
        Returns:
            Output path
        """
        if output_path is None:
            output_path = self.output_dir / f"music_{Path(video_path).stem}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path)
        ]
        
        success, _ = self._run_ffmpeg(cmd)
        
        if success:
            return str(output_path)
        return video_path
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,size",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return json.loads(result.stdout)
        except Exception:
            return {}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def assemble_video(scenes: List[Dict], output_name: str = "final.mp4") -> AssemblyResult:
    """Assemble video from scenes."""
    assembler = FFmpegAssembler()
    
    assembly_scenes = [
        AssemblyScene(
            index=s.get("index", i),
            video_path=s.get("video_path", ""),
            audio_path=s.get("audio_path", ""),
            duration=s.get("duration", 4.0),
            transition_type=s.get("transition", "fade"),
            transition_duration=s.get("transition_duration", 0.5),
        )
        for i, s in enumerate(scenes)
    ]
    
    return assembler.assemble_with_transitions(assembly_scenes, output_name)
