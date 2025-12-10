# uvg_core/uvg_pipeline.py
"""
UVG MAX Main Pipeline.

Complete video generation orchestrator integrating all modules:
- Script generation/loading
- TTS synthesis with word timing
- Media search and clip selection
- Visual processing (motion, overlays, color)
- Caption rendering
- Audio mastering
- Export optimization
"""

import logging
import os
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Directories
    output_dir: Path = field(default_factory=lambda: Path("uvg_output"))
    cache_dir: Path = field(default_factory=lambda: Path("uvg_cache"))
    
    # Quality settings
    target_platform: str = "youtube"
    style_pack: str = "cinematic"
    caption_style: str = "youtube"
    
    # Resolution
    width: int = 1080
    height: int = 1920
    fps: int = 30
    
    # Feature toggles
    enable_overlays: bool = True
    enable_color_matching: bool = True
    enable_motion_effects: bool = True
    enable_sfx: bool = True
    enable_mastering: bool = True
    enable_captions: bool = True
    
    # Performance
    use_whisper: bool = True
    mock_mode: bool = False
    max_retries: int = 3
    checkpoint_interval: int = 60
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create config from environment variables."""
        return cls(
            output_dir=Path(os.getenv("UVG_OUTPUT_DIR", "uvg_output")),
            cache_dir=Path(os.getenv("UVG_CACHE_DIR", "uvg_cache")),
            target_platform=os.getenv("UVG_PLATFORM", "youtube"),
            style_pack=os.getenv("UVG_STYLE", "cinematic"),
            mock_mode=os.getenv("UVG_MOCK_MODE", "false").lower() == "true",
        )


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    success: bool
    output_path: str
    thumbnail_path: str = ""
    duration_sec: float = 0.0
    scenes_processed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class SceneResult:
    """Result of processing a single scene."""
    index: int
    success: bool
    video_path: str = ""
    audio_path: str = ""
    caption_path: str = ""
    error: str = ""


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class UVGPipeline:
    """
    UVG MAX main video generation pipeline.
    
    Complete orchestration of all modules:
    1. Script loading/generation
    2. TTS synthesis
    3. Media search and selection
    4. Clip preparation (motion, color)
    5. Caption rendering
    6. Audio mastering
    7. Final assembly
    8. Export optimization
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize pipeline with configuration."""
        self.config = config or PipelineConfig.from_env()
        
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._modules = {}
        self._checkpoint_time = 0
        self._scene_results: List[SceneResult] = []
    
    # -------------------------------------------------------------------------
    # Module Loading
    # -------------------------------------------------------------------------
    
    def _get_module(self, name: str):
        """Lazy-load a module by name."""
        if name not in self._modules:
            self._modules[name] = self._init_module(name)
        return self._modules[name]
    
    def _init_module(self, name: str):
        """Initialize a specific module."""
        out = self.config.output_dir
        cache = self.config.cache_dir
        
        module_map = {
            "config": lambda: self._load_uvg_config(),
            "script_generator": lambda: self._load_script_generator(),
            "script_loader": lambda: self._load_script_loader(),
            "tts": lambda: self._load_tts_engine(),
            "whisper": lambda: self._load_whisper(),
            "media_search": lambda: self._load_media_search(),
            "creative_director": lambda: self._load_creative_director(),
            "clip_preparer": lambda: self._load_clip_preparer(),
            "clip_trimmer": lambda: self._load_clip_trimmer(),
            "audio_mastering": lambda: self._load_audio_mastering(),
            "audio_engine": lambda: self._load_audio_engine(),
            "visual_overlays": lambda: self._load_visual_overlays(),
            "vfx_engine": lambda: self._load_vfx_engine(),
            "color_matching": lambda: self._load_color_matching(),
            "motion_effects": lambda: self._load_motion_effects(),
            "captions": lambda: self._load_captions(),
            "subtitle_engine": lambda: self._load_subtitle_engine(),
            "kinetic_captions": lambda: self._load_kinetic_captions(),
            "sfx": lambda: self._load_sfx_engine(),
            "composer": lambda: self._load_composer(),
            "ffmpeg": lambda: self._load_ffmpeg_assembler(),
            "export": lambda: self._load_export_optimizer(),
            "thumbnail": lambda: self._load_thumbnail_generator(),
            # NEW modules for v2.1
            "deterministic": lambda: self._load_deterministic(),
            "license_tracker": lambda: self._load_license_tracker(),
            "scene_emotion": lambda: self._load_scene_emotion(),
            "speed_ramp": lambda: self._load_speed_ramp(),
            "music_sync": lambda: self._load_music_sync(),
        }
        
        if name in module_map:
            return module_map[name]()
        raise ValueError(f"Unknown module: {name}")
    
    # Module loaders
    def _load_uvg_config(self):
        from .config import UVGConfig
        return UVGConfig.from_env()
    
    def _load_script_generator(self):
        from .script_generator import ScriptGenerator
        return ScriptGenerator(mock_mode=self.config.mock_mode)
    
    def _load_script_loader(self):
        from .script_loader import ScriptLoader
        return ScriptLoader()
    
    def _load_tts_engine(self):
        from .tts_engine import TTSEngine
        return TTSEngine(
            output_dir=self.config.output_dir / "audio",
            mock_mode=self.config.mock_mode
        )
    
    def _load_whisper(self):
        from .whisper_timing import WhisperTimingExtractor
        return WhisperTimingExtractor()
    
    def _load_media_search(self):
        from .media_search import MediaSearchEngine
        return MediaSearchEngine(clips_dir=self.config.cache_dir / "media")
    
    def _load_creative_director(self):
        from .creative_director import CreativeDirector
        return CreativeDirector(mock_mode=self.config.mock_mode)
    
    def _load_clip_preparer(self):
        from .clip_preparer import ClipPreparer
        return ClipPreparer(output_dir=self.config.output_dir / "clips")
    
    def _load_clip_trimmer(self):
        from .clip_trimmer import ClipTrimmer
        return ClipTrimmer(output_dir=self.config.output_dir / "trimmed")
    
    def _load_audio_mastering(self):
        from .audio_mastering import AudioMasteringEngine
        return AudioMasteringEngine(output_dir=self.config.output_dir / "audio")
    
    def _load_audio_engine(self):
        from .audio_engine import AudioEngine
        return AudioEngine(output_dir=self.config.output_dir / "audio")
    
    def _load_visual_overlays(self):
        from .visual_overlays import VisualOverlayEngine
        return VisualOverlayEngine(output_dir=self.config.output_dir / "processed")
    
    def _load_vfx_engine(self):
        from .vfx_engine import VFXEngine
        return VFXEngine(output_dir=self.config.output_dir / "vfx")
    
    def _load_color_matching(self):
        from .color_matching import ColorMatchingEngine
        return ColorMatchingEngine(output_dir=self.config.output_dir / "color")
    
    def _load_motion_effects(self):
        from .motion_effects import MotionEffectsEngine
        return MotionEffectsEngine(output_dir=self.config.output_dir / "motion")
    
    def _load_captions(self):
        from .caption_renderer import CaptionRenderer
        return CaptionRenderer(output_dir=self.config.output_dir / "captions")
    
    def _load_subtitle_engine(self):
        from .subtitle_engine import SubtitleEngine
        return SubtitleEngine()
    
    def _load_kinetic_captions(self):
        from .kinetic_captions import KineticCaptionEngine
        return KineticCaptionEngine()
    
    def _load_sfx_engine(self):
        from .sfx_engine import SFXEngine
        return SFXEngine(cache_dir=self.config.cache_dir / "sfx")
    
    def _load_composer(self):
        from .multi_clip_composer import MultiClipComposer
        return MultiClipComposer(output_dir=self.config.output_dir / "composite")
    
    def _load_ffmpeg_assembler(self):
        from .ffmpeg_assembler import FFmpegAssembler
        return FFmpegAssembler(
            output_dir=self.config.output_dir / "assembled",
            target_width=self.config.width,
            target_height=self.config.height,
            fps=self.config.fps
        )
    
    def _load_export_optimizer(self):
        from .export_optimizer import ExportOptimizer
        return ExportOptimizer(output_dir=self.config.output_dir / "export")
    
    def _load_thumbnail_generator(self):
        from .thumbnail_generator import ThumbnailGenerator
        return ThumbnailGenerator(output_dir=self.config.output_dir / "thumbnails")
    
    # NEW module loaders for v2.1
    def _load_deterministic(self):
        from .deterministic import get_deterministic_context
        return get_deterministic_context()
    
    def _load_license_tracker(self):
        from .license_metadata import get_license_tracker
        return get_license_tracker()
    
    def _load_scene_emotion(self):
        from .scene_emotion import SceneEmotionController
        return SceneEmotionController()
    
    def _load_speed_ramp(self):
        from .speed_ramp import SpeedRampEngine
        return SpeedRampEngine(output_dir=self.config.output_dir / "speed_ramp")
    
    def _load_music_sync(self):
        from .music_engine import MusicSyncEngine
        return MusicSyncEngine(output_dir=self.config.output_dir / "music")
    
    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------
    
    def run(
        self,
        topic: str = None,
        script: Dict = None,
        script_path: str = None,
        music_path: str = None,
        output_name: str = "final",
        on_progress: Callable[[str, float], None] = None
    ) -> PipelineResult:
        """
        Run the complete video generation pipeline.
        
        Args:
            topic: Topic for script generation
            script: Pre-made script dict
            script_path: Path to script JSON
            music_path: Optional background music
            output_name: Output filename (without extension)
            on_progress: Progress callback (step_name, percent)
            
        Returns:
            PipelineResult with output video path
        """
        start_time = time.time()
        timing = {}
        errors = []
        warnings = []
        
        # Initialize deterministic mode if needed
        if script and script.get("video_meta", {}).get("deterministic_mode", False):
            try:
                from .deterministic import init_deterministic_mode
                seed = script.get("video_meta", {}).get("random_seed", 42)
                init_deterministic_mode(enabled=True, seed=seed)
                logger.info(f"Deterministic mode enabled with seed={seed}")
            except ImportError:
                pass
        
        # Initialize license tracker
        try:
            from .license_metadata import reset_license_tracker
            reset_license_tracker()
        except ImportError:
            pass
        
        def progress(step: str, pct: float):
            if on_progress:
                on_progress(step, pct)
            logger.info(f"[{pct:.0%}] {step}")
        
        try:
            # =================================================================
            # STEP 1: Load/Generate Script
            # =================================================================
            progress("Loading script...", 0.05)
            step_start = time.time()
            
            if script:
                script_loader = self._get_module("script_loader")
                loaded_script = script_loader.load_from_dict(script)
            elif script_path:
                script_loader = self._get_module("script_loader")
                loaded_script = script_loader.load_from_json(script_path)
            elif topic:
                script_gen = self._get_module("script_generator")
                loaded_script = script_gen.generate_script(prompt=topic)
            else:
                raise ValueError("Must provide topic, script, or script_path")
            
            # Validate
            script_loader = self._get_module("script_loader")
            validation = script_loader.validate(loaded_script)
            if not validation.valid:
                return PipelineResult(success=False, output_path="", errors=validation.errors)
            
            warnings.extend(validation.warnings)
            timing["script"] = time.time() - step_start
            num_scenes = len(loaded_script.scenes)
            logger.info(f"Loaded {num_scenes} scenes")
            
            # =================================================================
            # STEP 2: Creative Direction
            # =================================================================
            progress("Analyzing scenes...", 0.10)
            step_start = time.time()
            
            director = self._get_module("creative_director")
            scene_directions = []
            
            for i, scene in enumerate(loaded_script.scenes):
                direction = director.get_scene_direction(
                    scene_idx=i,
                    scene_text=scene.text,
                    emotion=scene.emotion
                )
                scene_directions.append(direction)
            
            timing["direction"] = time.time() - step_start
            
            # =================================================================
            # STEP 3: TTS Synthesis
            # =================================================================
            progress("Synthesizing speech...", 0.15)
            step_start = time.time()
            
            tts = self._get_module("tts")
            tts_results = []
            
            for i, scene in enumerate(loaded_script.scenes):
                audio_path = str(self.config.output_dir / "audio" / f"scene_{i}.wav")
                emotion = scene_directions[i].emotion_tag if scene_directions else "neutral"
                result = tts.synthesize(scene.text, emotion, audio_path)
                tts_results.append(result)
            
            timing["tts"] = time.time() - step_start
            
            # =================================================================
            # STEP 4: Whisper Timing (optional)
            # =================================================================
            if self.config.use_whisper:
                progress("Extracting word timings...", 0.20)
                step_start = time.time()
                
                try:
                    whisper = self._get_module("whisper")
                    for i, tts_result in enumerate(tts_results):
                        if tts_result.success and tts_result.audio_path:
                            whisper_result = whisper.extract_timings(tts_result.audio_path)
                            if whisper_result.success:
                                tts_results[i].word_timings = whisper_result.to_word_timings()
                except Exception as e:
                    warnings.append(f"Whisper failed: {e}")
                
                timing["whisper"] = time.time() - step_start
            
            # =================================================================
            # STEP 5: Media Search
            # =================================================================
            progress("Searching for clips...", 0.25)
            step_start = time.time()
            
            media_search = self._get_module("media_search")
            clip_paths = []
            
            for i, direction in enumerate(scene_directions):
                query = direction.search_query if hasattr(direction, 'search_query') else loaded_script.scenes[i].text[:50]
                clips = media_search.search_and_download(query, max_candidates=5, max_downloads=1)
                # Extract downloaded_path from MediaCandidate object
                if clips and len(clips) > 0:
                    clip = clips[0]
                    # Handle both MediaCandidate objects and string paths
                    if hasattr(clip, 'downloaded_path'):
                        clip_paths.append(clip.downloaded_path)
                    else:
                        clip_paths.append(str(clip))
                else:
                    clip_paths.append(None)
            
            timing["media_search"] = time.time() - step_start
            
            # =================================================================
            # STEP 6: Process Each Scene
            # =================================================================
            progress("Processing scenes...", 0.30)
            step_start = time.time()
            
            scene_clips = []
            preparer = self._get_module("clip_preparer")
            
            for i, (scene, clip_path) in enumerate(zip(loaded_script.scenes, clip_paths)):
                pct = 0.30 + (i / num_scenes) * 0.35
                progress(f"Processing scene {i+1}/{num_scenes}...", pct)
                
                if clip_path:
                    duration = tts_results[i].duration_ms / 1000 if tts_results[i].success else 5.0
                    motion = scene_directions[i].camera_motion if scene_directions else "slow-zoom-in"
                    
                    result = preparer.prepare_clip(
                        clip_path=clip_path,
                        scene_idx=i,
                        target_duration=duration,
                        motion_type=motion
                    )
                    
                    if result.success:
                        scene_clips.append(result.output_path)
                    else:
                        scene_clips.append(None)
                        warnings.append(f"Scene {i} clip preparation failed")
                else:
                    scene_clips.append(None)
            
            timing["clip_processing"] = time.time() - step_start
            
            # =================================================================
            # STEP 7: Generate Captions
            # =================================================================
            if self.config.enable_captions:
                progress("Rendering captions...", 0.70)
                step_start = time.time()
                
                caption_renderer = self._get_module("captions")
                all_timings = []
                offset = 0
                
                for tts_result in tts_results:
                    if tts_result.word_timings:
                        for wt in tts_result.word_timings:
                            all_timings.append({
                                "word": wt.word,
                                "start_ms": wt.start_ms + offset,
                                "end_ms": wt.end_ms + offset
                            })
                    offset += tts_result.duration_ms if tts_result.success else 0
                
                if all_timings:
                    caption_result = caption_renderer.render_ass(all_timings, self.config.caption_style)
                    caption_path = caption_result.output_path if caption_result.success else None
                else:
                    caption_path = None
                
                timing["captions"] = time.time() - step_start
            else:
                caption_path = None
            
            # =================================================================
            # STEP 8: Audio Mastering
            # =================================================================
            if self.config.enable_mastering:
                progress("Mastering audio...", 0.75)
                step_start = time.time()
                
                audio_engine = self._get_module("audio_engine")
                voice_paths = [r.audio_path for r in tts_results if r.success and r.audio_path]
                
                if voice_paths:
                    # Normalize all audio
                    normalized = audio_engine.smooth_loudness_across_scenes(voice_paths)
                    
                    # Mix with music if provided
                    if music_path and os.path.exists(music_path):
                        mastered_path = str(self.config.output_dir / "audio" / "mastered.wav")
                        audio_engine.mix_layers(
                            voice_path=normalized[0] if normalized else voice_paths[0],
                            music_path=music_path,
                            output_path=mastered_path
                        )
                    else:
                        mastered_path = normalized[0] if normalized else voice_paths[0]
                else:
                    mastered_path = None
                
                timing["mastering"] = time.time() - step_start
            else:
                mastered_path = None
            
            # =================================================================
            # STEP 9: Assemble Final Video
            # =================================================================
            progress("Assembling video...", 0.85)
            step_start = time.time()
            
            ffmpeg = self._get_module("ffmpeg")
            
            # Build scene list for assembly
            from .ffmpeg_assembler import AssemblyScene
            scenes_for_assembly = []
            for i, clip_path in enumerate(scene_clips):
                if clip_path:
                    scenes_for_assembly.append(AssemblyScene(
                        index=i,
                        video_path=clip_path,
                        audio_path=tts_results[i].audio_path if tts_results[i].success else "",
                        duration=tts_results[i].duration_ms / 1000 if tts_results[i].success else 5.0,
                        transition_type=scene_directions[i].transition_type if scene_directions else "fade",
                        transition_duration=scene_directions[i].transition_duration if scene_directions else 0.5
                    ))
            
            output_path = str(self.config.output_dir / "final" / f"{output_name}.mp4")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if scenes_for_assembly:
                # Use assemble_with_transitions (or assemble_simple for speed)
                assemble_result = ffmpeg.assemble_with_transitions(
                    scenes_for_assembly,
                    output_name=f"{output_name}.mp4"
                )
                
                if not assemble_result.success:
                    warnings.append(f"Assembly warning: {assemble_result.error}")
                else:
                    output_path = assemble_result.output_path
            else:
                # Create placeholder if no clips
                self._create_placeholder_video(output_path, loaded_script.total_duration)
            
            timing["assembly"] = time.time() - step_start
            
            # =================================================================
            # STEP 10: Export for Platform
            # =================================================================
            progress("Optimizing export...", 0.95)
            step_start = time.time()
            
            export = self._get_module("export")
            export_result = export.export(output_path, self.config.target_platform)
            
            if export_result.success:
                final_path = export_result.output_path
            else:
                final_path = output_path
            
            timing["export"] = time.time() - step_start
            
            # =================================================================
            # STEP 11: Generate Thumbnail
            # =================================================================
            progress("Generating thumbnail...", 0.98)
            
            try:
                thumbnail = self._get_module("thumbnail")
                thumb_result = thumbnail.generate(final_path, loaded_script.title)
                thumbnail_path = thumb_result.output_path if thumb_result.success else ""
            except Exception as e:
                thumbnail_path = ""
                warnings.append(f"Thumbnail generation failed: {e}")
            
            # =================================================================
            # DONE
            # =================================================================
            total_time = time.time() - start_time
            timing["total"] = total_time
            
            progress("Complete!", 1.0)
            logger.info(f"✅ Pipeline complete in {total_time:.1f}s")
            
            return PipelineResult(
                success=True,
                output_path=final_path,
                thumbnail_path=thumbnail_path,
                duration_sec=loaded_script.total_duration,
                scenes_processed=num_scenes,
                warnings=warnings,
                timing=timing
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return PipelineResult(
                success=False,
                output_path="",
                errors=[str(e)]
            )
    
    def _create_placeholder_video(self, output_path: str, duration: float):
        """Create placeholder black video."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={self.config.width}x{self.config.height}:d={duration}",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_pipeline(
    topic: str = None,
    script: Dict = None,
    platform: str = "youtube",
    mock_mode: bool = False
) -> PipelineResult:
    """
    Quick pipeline execution.
    
    Args:
        topic: Video topic for generation
        script: Pre-made script dict
        platform: Target platform
        mock_mode: Use mock APIs
        
    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        target_platform=platform,
        mock_mode=mock_mode
    )
    pipeline = UVGPipeline(config)
    return pipeline.run(topic=topic, script=script)


def generate_video(topic: str, style: str = "cinematic") -> str:
    """
    Simple one-liner video generation.
    
    Args:
        topic: Video topic
        style: Style preset
        
    Returns:
        Path to generated video
    """
    config = PipelineConfig(style_pack=style)
    pipeline = UVGPipeline(config)
    result = pipeline.run(topic=topic)
    return result.output_path if result.success else ""


def run_from_json(script_dict: Dict[str, Any]) -> PipelineResult:
    """
    Run pipeline from JSON schema v2.1 script.
    
    Args:
        script_dict: Complete JSON script following schema_v2.py format
        
    Returns:
        PipelineResult
    """
    video_meta = script_dict.get("video_meta", {})
    
    # Extract config from script
    config = PipelineConfig(
        target_platform=video_meta.get("narrative_style", "youtube"),
        style_pack=video_meta.get("narrative_style", "cinematic"),
        width=video_meta.get("resolution", {}).get("width", 1080),
        height=video_meta.get("resolution", {}).get("height", 1920),
        enable_captions=video_meta.get("include_captions", True),
        mock_mode=False
    )
    
    pipeline = UVGPipeline(config)
    return pipeline.run(script=script_dict)

