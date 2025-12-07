"""
UVG MAX Core Engine

Professional-grade AI video generation engine.
"""

__version__ = "1.0.0"
__author__ = "UVG MAX Team"

# Configuration
from .config import (
    UVGConfig,
    STYLE_PRESETS,
    AZURE_VOICE_STYLES,
)

# Resource Management
from .hardware_detector import (
    HardwareDetector,
    HardwareProfile,
    detect_hardware,
    get_optimal_settings,
)

from .gpu_memory_manager import (
    GPUMemoryManager,
    MemoryStatus,
    get_memory_status,
    run_with_memory_management,
)

from .disk_watchdog import (
    DiskWatchdog,
    DiskStatus,
    CleanupResult,
)

# Script Generation
from .script_generator import (
    ScriptGenerator,
    Script,
    Scene,
)

from .script_structure import (
    ScriptStructure,
    StructuredScript,
    StructuredScene,
    StoryBeat,
)

# Creative Direction
from .creative_director import (
    CreativeDirector,
    CreativeBrief,
    SceneDirection,
)

from .scene_relevance import (
    SceneRelevanceValidator,
    ValidationResult,
)

# Media Search and Scoring
from .media_search import (
    MediaSearchEngine,
    MediaCandidate,
    SearchResult,
)

from .vision_scorer import (
    VisionScorer,
    ClipMetrics,
)

from .visual_density_score import (
    VisualDensityScorer,
    DensityMetrics,
)

# Clip Processing
from .clip_trimmer import (
    ClipTrimmer,
    TrimResult,
)

from .clip_preparer import (
    ClipPreparer,
    PrepareResult,
    CameraPath,
)

# Audio
from .tts_engine import (
    TTSEngine,
    TTSResult,
    WordTiming,
)

from .audio_engine import (
    AudioEngine,
    AudioSegment,
    MasteringResult,
)

from .music_engine import (
    MusicEngine,
    MusicTrack,
    BeatInfo,
)

# Subtitles and Captions
from .subtitle_engine import (
    SubtitleEngine,
    Subtitle,
    CaptionStyle,
)

from .caption_animation import (
    CaptionAnimation,
    WordAnimation,
    AnimationType,
)

# VFX and Transitions
from .vfx_engine import (
    VFXEngine,
    VFXPreset,
    VFX_PRESETS,
)

from .transition_engine import (
    TransitionEngine,
    TransitionSpec,
    TRANSITIONS,
)

from .pacing_engine import (
    PacingEngine,
    PacingStyle,
    PacingPoint,
)

# Assembly
from .ffmpeg_assembler import (
    FFmpegAssembler,
    AssemblyScene,
    AssemblyResult,
)

# Output
from .thumbnail_generator import (
    ThumbnailGenerator,
    ThumbnailResult,
)

# Orchestration
from .orchestrator import (
    Orchestrator,
    ProjectState,
    SceneState,
    SceneStatus,
)


__all__ = [
    # Version
    "__version__",
    
    # Config
    "UVGConfig",
    "STYLE_PRESETS",
    "AZURE_VOICE_STYLES",
    
    # Hardware
    "HardwareDetector",
    "HardwareProfile",
    "GPUMemoryManager",
    "MemoryStatus",
    "DiskWatchdog",
    "DiskStatus",
    
    # Script
    "ScriptGenerator",
    "Script",
    "Scene",
    "ScriptStructure",
    "StructuredScript",
    "StoryBeat",
    
    # Creative
    "CreativeDirector",
    "CreativeBrief",
    "SceneRelevanceValidator",
    
    # Media
    "MediaSearchEngine",
    "MediaCandidate",
    "VisionScorer",
    "ClipMetrics",
    "VisualDensityScorer",
    
    # Clips
    "ClipTrimmer",
    "ClipPreparer",
    
    # Audio
    "TTSEngine",
    "TTSResult",
    "AudioEngine",
    "MusicEngine",
    
    # Captions
    "SubtitleEngine",
    "CaptionAnimation",
    
    # VFX
    "VFXEngine",
    "TransitionEngine",
    "PacingEngine",
    
    # Assembly
    "FFmpegAssembler",
    
    # Output
    "ThumbnailGenerator",
    
    # Orchestration
    "Orchestrator",
    "ProjectState",
]
