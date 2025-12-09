"""
Unit tests for ffmpeg_assembler module.
"""

import pytest


class TestFFmpegAssembler:
    """Test FFmpegAssembler class."""
    
    def test_assembler_creation(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        
        assert assembler.target_width == 1080
        assert assembler.target_height == 1920
        assert assembler.fps == 30
    
    def test_get_encoder_settings_cpu(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir, use_cuda=False)
        settings = assembler._get_encoder_settings()
        
        assert "-c:v" in settings
        assert "libx264" in settings


class TestCameraMotionFilters:
    """Test camera motion filter generation."""
    
    def test_get_camera_motion_filter_zoom_in(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_camera_motion_filter("slow-zoom-in", 4.0)
        
        assert "zoompan" in filter_str
        assert "zoom+" in filter_str
    
    def test_get_camera_motion_filter_pan(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_camera_motion_filter("pan-left", 4.0)
        
        assert "zoompan" in filter_str
    
    def test_get_camera_motion_filter_static(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_camera_motion_filter("static", 4.0)
        
        assert filter_str == ""
    
    def test_get_camera_motion_filter_unknown(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_camera_motion_filter("nonexistent", 4.0)
        
        assert filter_str == ""


class TestColorGradeFilters:
    """Test color grade filter generation."""
    
    def test_get_color_grade_warm(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_color_grade_filter("warm")
        
        assert "colorbalance" in filter_str
        assert "saturation" in filter_str
    
    def test_get_color_grade_cold(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_color_grade_filter("cold")
        
        assert "colorbalance" in filter_str
        # Cold has negative red shift
        assert "rs=-.1" in filter_str
    
    def test_get_color_grade_cinematic(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_color_grade_filter("cinematic")
        
        assert "unsharp" in filter_str  # Cinematic includes sharpening
    
    def test_get_color_grade_unknown(self, temp_output_dir):
        from uvg_core.ffmpeg_assembler import FFmpegAssembler
        
        assembler = FFmpegAssembler(output_dir=temp_output_dir)
        filter_str = assembler.get_color_grade_filter("nonexistent")
        
        assert filter_str == ""


class TestAssemblyScene:
    """Test AssemblyScene dataclass."""
    
    def test_assembly_scene_creation(self):
        from uvg_core.ffmpeg_assembler import AssemblyScene
        
        scene = AssemblyScene(
            index=0,
            video_path="/path/to/video.mp4",
            audio_path="/path/to/audio.wav",
            duration=4.0
        )
        
        assert scene.index == 0
        assert scene.video_path == "/path/to/video.mp4"
        assert scene.duration == 4.0
        assert scene.transition_type == "fade"  # default


class TestAssemblyResult:
    """Test AssemblyResult dataclass."""
    
    def test_assembly_result_success(self):
        from uvg_core.ffmpeg_assembler import AssemblyResult
        
        result = AssemblyResult(
            success=True,
            output_path="/path/to/output.mp4",
            duration=30.0,
            file_size_mb=25.5
        )
        
        assert result.success is True
        assert result.file_size_mb == 25.5
        assert result.error == ""
    
    def test_assembly_result_failure(self):
        from uvg_core.ffmpeg_assembler import AssemblyResult
        
        result = AssemblyResult(
            success=False,
            output_path="",
            duration=0,
            file_size_mb=0,
            error="FFmpeg not found"
        )
        
        assert result.success is False
        assert result.error == "FFmpeg not found"
