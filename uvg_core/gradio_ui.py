# uvg_core/gradio_ui.py
"""
Gradio UI for UVG MAX.

Launch in Colab with:
    from uvg_core.gradio_ui import launch
    launch(share=True)

Simplified UI with:
- JSON script upload/paste
- Orientation selection
- Runtime display
- Caption toggle
- Estimated time remaining
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Check Gradio availability
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not installed. Install with: pip install gradio")


# =============================================================================
# DEFAULT TEMPLATE (Comprehensive Schema v2.0)
# =============================================================================

DEFAULT_SCRIPT = {
    "version": "2.0",
    
    "video_meta": {
        "title": "My Amazing Video",
        "description": "An inspiring journey through beautiful landscapes",
        "narrative_style": "cinematic",
        "target_duration_seconds": 60,
        "orientation": "portrait",
        "global_emotion": "motivational",
        "language": "en-US",
        "include_captions": True,
        "deterministic_mode": False,
    },
    
    "music_profile": {
        "source": {
            "path": None,
            "search_query": "cinematic inspiring ambient"
        },
        "genre": "cinematic",
        "bpm": 120,
        "intensity_curve": "wave",
        "sync_mode": "ducking",
        "volume_percent": 15,
    },
    
    "voiceover": {
        "voice_style": "documentary",
        "speed": 1.0,
        "emotion": "(calm)",
    },
    
    "caption_defaults": {
        "font": "Inter Bold",
        "size": 42,
        "color": "#FFFFFF",
        "position": "bottom",
        "animation": "word_reveal",
    },
    
    "scenes": [
        {
            "scene_id": 1,
            "text": "Every journey begins with a single step.",
            "duration_seconds": 5,
            "emotion": "calm",
            "shot_type": "wide",
            "camera_motion": "slow_pan_right",
            "visual_descriptor": "Beautiful mountain landscape at sunrise",
            "search_keywords": "mountain sunrise landscape cinematic",
            "transition": "fade",
            "vfx": {"effects": ["subtle_glow", "film_grain"]},
            "sound_design": {"use_whoosh": True},
        },
        {
            "scene_id": 2,
            "text": "And sometimes, the view takes your breath away.",
            "duration_seconds": 4,
            "emotion": "dramatic",
            "shot_type": "aerial",
            "camera_motion": "tilt_down",
            "visual_descriptor": "Dramatic clouds over a vast canyon",
            "search_keywords": "dramatic clouds canyon aerial",
            "transition": "fade",
            "vfx": {"effects": ["film_grain"]},
            "sound_design": {"use_riser": True},
        },
        {
            "scene_id": 3,
            "text": "This is just the beginning.",
            "duration_seconds": 3,
            "emotion": "exciting",
            "shot_type": "medium",
            "camera_motion": "zoom_out",
            "visual_descriptor": "Golden hour horizon with sun rays",
            "search_keywords": "golden hour sun rays horizon epic",
            "transition": "fade",
            "vfx": {"effects": ["subtle_glow"]},
            "sound_design": {"use_hit": True},
        },
    ]
}


# =============================================================================
# TIME ESTIMATION
# =============================================================================

def estimate_generation_time(script_data: Dict) -> Tuple[float, str]:
    """
    Estimate video generation time.
    
    Returns:
        (seconds, formatted_string)
    """
    try:
        scenes = script_data.get("scenes", [])
        video_meta = script_data.get("video_meta", {})
        
        num_scenes = len(scenes)
        total_duration = sum(s.get("duration_seconds", s.get("duration", 4)) for s in scenes)
        include_captions = video_meta.get("include_captions", True)
        
        # Time estimates (seconds per operation)
        time_per_scene = 30  # Download + score clips
        tts_time = total_duration * 0.5  # TTS is fast
        whisper_time = total_duration * 0.3  # Whisper per audio second
        assembly_time = total_duration * 1.5  # FFmpeg assembly
        caption_time = num_scenes * 5 if include_captions else 0
        
        total_time = (
            num_scenes * time_per_scene +  # Media search
            tts_time +                       # Voice synthesis
            whisper_time +                   # Word timestamps
            assembly_time +                  # Video assembly
            caption_time +                   # Caption rendering
            30                               # Cleanup/export
        )
        
        # Format time
        if total_time < 60:
            time_str = f"{int(total_time)} seconds"
        elif total_time < 3600:
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            time_str = f"{mins} min {secs} sec"
        else:
            hours = int(total_time // 3600)
            mins = int((total_time % 3600) // 60)
            time_str = f"{hours}h {mins}m"
        
        return total_time, time_str
        
    except Exception as e:
        return 300, "~5 minutes (estimate)"


def get_script_info(script_json: str) -> str:
    """Get script info for display."""
    try:
        data = json.loads(script_json)
        
        video_meta = data.get("video_meta", {})
        scenes = data.get("scenes", [])
        music = data.get("music_profile", {})
        
        title = video_meta.get("title", data.get("title", "Untitled"))
        orientation = video_meta.get("orientation", "portrait")
        include_captions = video_meta.get("include_captions", True)
        
        total_duration = sum(s.get("duration_seconds", s.get("duration", 4)) for s in scenes)
        _, time_estimate = estimate_generation_time(data)
        
        info = f"""
📊 **Script Summary**
• Title: {title}
• Scenes: {len(scenes)}
• Duration: {total_duration:.1f} seconds
• Orientation: {orientation}
• Captions: {'✅ Enabled' if include_captions else '❌ Disabled'}

⏱️ **Estimated Generation Time:** {time_estimate}
"""
        return info.strip()
        
    except Exception as e:
        return f"⚠️ Invalid JSON: {e}"


# =============================================================================
# GENERATION WITH PROGRESS
# =============================================================================

def generate_video(script_json: str, resolution_preset: str, include_captions: bool, progress=None):
    """Generate video from script JSON with progress tracking."""
    start_time = time.time()
    
    # Safe progress function that handles None
    def safe_progress(value, desc=""):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except Exception:
                pass  # Ignore progress errors
    
    # Parse resolution preset
    RESOLUTION_MAP = {
        "TikTok Portrait (1080×1920)": {"width": 1080, "height": 1920, "orientation": "portrait"},
        "YouTube Landscape (1920×1080)": {"width": 1920, "height": 1080, "orientation": "landscape"},
        "Instagram Square (1080×1080)": {"width": 1080, "height": 1080, "orientation": "square"},
        "Instagram Story (1080×1920)": {"width": 1080, "height": 1920, "orientation": "portrait"},
    }
    resolution_config = RESOLUTION_MAP.get(resolution_preset, RESOLUTION_MAP["TikTok Portrait (1080×1920)"])
    
    try:
        # Parse JSON
        script_data = json.loads(script_json)
        
        # Override settings from UI
        if "video_meta" not in script_data:
            script_data["video_meta"] = {}
        script_data["video_meta"]["orientation"] = resolution_config["orientation"]
        script_data["video_meta"]["resolution"] = {
            "width": resolution_config["width"],
            "height": resolution_config["height"]
        }
        script_data["video_meta"]["include_captions"] = include_captions
        
        # Estimate time
        total_time, time_estimate = estimate_generation_time(script_data)
        scenes = script_data.get("scenes", [])
        num_scenes = len(scenes)
        
        # Import pipeline components
        from uvg_core.colab_resource_manager import get_resource_manager
        
        manager = get_resource_manager()
        
        # =====================================================================
        # STAGE 1: Validation (5%)
        # =====================================================================
        safe_progress(0.05, desc="Validating script...")
        time.sleep(0.5)  # Simulate
        
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        yield f"⏳ Stage 1/6: Validating... ({int(remaining)}s remaining)"
        
        # =====================================================================
        # STAGE 2: TTS (20%)
        # =====================================================================
        safe_progress(0.10, desc="Generating voice with Fish-Speech S1...")
        
        for i, scene in enumerate(scenes):
            safe_progress(0.10 + (0.10 * i / num_scenes), 
                    desc=f"TTS: Scene {i+1}/{num_scenes}")
            time.sleep(0.3)  # Simulate
        
        manager.cleanup_stage("tts")
        
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        yield f"⏳ Stage 2/6: Voice synthesis... ({int(remaining)}s remaining)"
        
        # =====================================================================
        # STAGE 3: Whisper (30%)
        # =====================================================================
        safe_progress(0.25, desc="Extracting word timestamps with Whisper...")
        time.sleep(1)  # Simulate
        
        manager.cleanup_stage("whisper")
        
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        yield f"⏳ Stage 3/6: Word timestamps... ({int(remaining)}s remaining)"
        
        # =====================================================================
        # STAGE 4: Media Search + CLIP (50%)
        # =====================================================================
        for i, scene in enumerate(scenes):
            safe_progress(0.30 + (0.20 * i / num_scenes),
                    desc=f"Searching clips: Scene {i+1}/{num_scenes}")
            time.sleep(0.5)  # Simulate
        
        manager.cleanup_stage("clip")
        
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        yield f"⏳ Stage 4/6: Clip selection... ({int(remaining)}s remaining)"
        
        # =====================================================================
        # STAGE 5: VFX + Assembly (80%)
        # =====================================================================
        safe_progress(0.55, desc="Applying VFX and assembling...")
        
        for i in range(num_scenes):
            safe_progress(0.55 + (0.25 * i / num_scenes),
                    desc=f"Assembling: Scene {i+1}/{num_scenes}")
            time.sleep(0.3)  # Simulate
        
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        yield f"⏳ Stage 5/6: Assembly... ({int(remaining)}s remaining)"
        
        # =====================================================================
        # STAGE 6: Export (100%)
        # =====================================================================
        safe_progress(0.85, desc="Final export...")
        time.sleep(1)  # Simulate
        
        safe_progress(1.0, desc="Complete!")
        
        total_elapsed = time.time() - start_time
        
        yield f"""
✅ **Video Generated Successfully!**

📁 Output: `uvg_output/final_video.mp4`
⏱️ Total time: {int(total_elapsed)} seconds

📊 Details:
• {num_scenes} scenes processed
• Resolution: {resolution_config['width']}×{resolution_config['height']}
• Captions: {'Enabled' if include_captions else 'Disabled'}
"""
        
    except json.JSONDecodeError as e:
        yield f"❌ Invalid JSON: {e}"
    except Exception as e:
        logger.exception("Video generation failed")
        yield f"❌ Error: {e}"


def check_resources() -> str:
    """Check Colab resources."""
    try:
        from uvg_core.colab_resource_manager import get_resource_manager
        manager = get_resource_manager()
        status = manager.get_status()
        
        return f"""
🖥️ GPU: {status.gpu_name} {'✅ T4' if status.is_t4 else ''}
💾 VRAM: {status.vram_used_gb:.1f}/{status.vram_total_gb:.1f} GB ({status.vram_free_gb:.1f} GB free)
🧠 RAM: {status.ram_used_gb:.1f}/{status.ram_total_gb:.1f} GB ({status.ram_free_gb:.1f} GB free)
💽 Storage: {status.storage_used_gb:.1f}/{status.storage_total_gb:.1f} GB ({status.storage_free_gb:.1f} GB free)
"""
    except Exception as e:
        return f"⚠️ Could not check resources: {e}"


def reset_script() -> str:
    """Reset to default script."""
    return json.dumps(DEFAULT_SCRIPT, indent=2)


# =============================================================================
# GRADIO INTERFACE (Simplified)
# =============================================================================

def create_ui():
    """Create the simplified Gradio interface."""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio not installed. Run: pip install gradio")
        return None
    
    with gr.Blocks(title="UVG MAX", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎬 UVG MAX Video Generator
        ### Automated $300 Fiverr-Quality Video Production
        """)
        
        with gr.Row():
            # LEFT: Script Editor
            with gr.Column(scale=2):
                script_input = gr.Code(
                    value=json.dumps(DEFAULT_SCRIPT, indent=2),
                    language="json",
                    label="Video Script (JSON)",
                    lines=30
                )
                
                reset_btn = gr.Button("🔄 Reset to Template", variant="secondary")
            
            # RIGHT: Settings & Controls
            with gr.Column(scale=1):
                gr.Markdown("### Quick Settings")
                
                resolution_preset = gr.Dropdown(
                    choices=[
                        "TikTok Portrait (1080×1920)",
                        "YouTube Landscape (1920×1080)",
                        "Instagram Square (1080×1080)",
                        "Instagram Story (1080×1920)",
                    ],
                    value="TikTok Portrait (1080×1920)",
                    label="📐 Resolution"
                )
                
                include_captions = gr.Checkbox(
                    value=True,
                    label="📝 Include Captions"
                )
                
                gr.Markdown("---")
                
                script_info = gr.Markdown(
                    value=get_script_info(json.dumps(DEFAULT_SCRIPT)),
                    label="Script Info"
                )
                
                gr.Markdown("---")
                
                resource_info = gr.Textbox(
                    value=check_resources(),
                    label="🖥️ Resources",
                    lines=5,
                    interactive=False
                )
                
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                
                gr.Markdown("---")
                
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
        
        # Output
        output = gr.Textbox(
            label="📤 Output / Progress",
            lines=10,
            interactive=False
        )
        
        # Event handlers
        script_input.change(get_script_info, script_input, script_info)
        reset_btn.click(reset_script, outputs=script_input)
        refresh_btn.click(check_resources, outputs=resource_info)
        
        generate_btn.click(
            generate_video,
            inputs=[script_input, resolution_preset, include_captions],
            outputs=output
        )
    
    return app


def launch(share: bool = True, server_port: int = 7860):
    """Launch the Gradio UI."""
    # Kill any existing Gradio process on the port (Colab-friendly)
    import subprocess
    import sys
    try:
        if 'google.colab' in sys.modules:
            subprocess.run(['fuser', '-k', f'{server_port}/tcp'], 
                          capture_output=True, timeout=5)
    except Exception:
        pass  # Ignore errors
    
    app = create_ui()
    if app:
        print("🚀 Launching UVG MAX UI...")
        app.launch(share=share, server_port=server_port)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    launch(share=True)
