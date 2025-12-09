#!/usr/bin/env python
"""
UVG MAX Demo Script

Demonstrates the full video generation pipeline.
Default mode: mock (no API keys required)
Real mode: requires API keys in .env

Usage:
    python scripts/run_demo.py --mock    # Mock mode (default)
    python scripts/run_demo.py --real    # Real API mode
    python scripts/run_demo.py --help    # Show help
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("UVG_DEMO")


def setup_environment(mock_mode: bool = True):
    """Set up environment variables for demo."""
    if mock_mode:
        os.environ["UVG_MOCK_MODE"] = "true"
        os.environ["UVG_TTS_PROVIDER"] = "mock"
        os.environ["UVG_DEBUG_SEED"] = "42"
        logger.info("🔧 Running in MOCK mode (no API keys required)")
    else:
        os.environ["UVG_MOCK_MODE"] = "false"
        logger.info("🔧 Running in REAL mode (API keys required)")
        
        # Load .env if exists
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
            logger.info(f"📄 Loaded environment from {env_path}")


def verify_output_dir():
    """Create output directory."""
    output_dir = Path("./uvg_output")
    output_dir.mkdir(exist_ok=True)
    
    for subdir in ["clips", "audio", "frames", "final", "thumbnails"]:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    return output_dir


def run_demo_pipeline(output_dir: Path, parallel_scenes: bool = False):
    """Run the full demo pipeline."""
    logger.info("=" * 60)
    logger.info("🎬 UVG MAX Demo Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Script Generation
    logger.info("\n📝 Step 1: Generating Script...")
    from uvg_core.script_generator import ScriptGenerator
    
    script_gen = ScriptGenerator()
    script = script_gen.generate_script(
        prompt="Create an inspiring video about achieving your dreams and never giving up.",
        target_duration=30,
        style="motivational",
        num_scenes=5
    )
    
    logger.info(f"   Generated: {len(script.scenes)} scenes, {script.total_duration}s total")
    for i, scene in enumerate(script.scenes):
        logger.info(f"   Scene {i}: [{scene.emotion}] {scene.text[:50]}...")
    
    # Step 2: Creative Direction
    logger.info("\n🎨 Step 2: Creative Direction...")
    from uvg_core.creative_director import CreativeDirector
    
    director = CreativeDirector()
    for scene in script.scenes:
        emotion = director.classify_emotion(scene.text)
        settings = director.get_mood_shot_settings(emotion)
        logger.info(f"   Scene {scene.index}: {emotion} → {settings['camera_motion']}, {settings['color_grade']}")
    
    # Step 3: TTS Synthesis
    logger.info("\n🎙️ Step 3: TTS Synthesis...")
    from uvg_core.tts_engine import TTSEngine
    
    tts_engine = TTSEngine(output_dir=output_dir / "audio")
    tts_results = []
    
    for scene in script.scenes:
        # Use adapter directly in mock mode
        if tts_engine._adapter:
            audio_path = str(output_dir / "audio" / f"scene_{scene.index}.wav")
            result = tts_engine._adapter.synthesize(scene.text, "calm", audio_path)
            tts_results.append(result)
            logger.info(f"   Scene {scene.index}: {result.duration_ms}ms, {len(result.word_timings)} words")
    
    # Step 4: Kinetic Captions
    logger.info("\n✨ Step 4: Kinetic Captions...")
    from uvg_core.kinetic_captions import KineticCaptions
    
    captions_engine = KineticCaptions()
    all_caption_layers = []
    
    for tts_result in tts_results:
        if tts_result.word_timings:
            layers = captions_engine.generate_layers(tts_result.word_timings, style="youtube")
            all_caption_layers.extend(layers)
            logger.info(f"   Generated {len(layers)} caption layers")
    
    # Step 5: Generate placeholder output
    logger.info("\n🎬 Step 5: Assembling Final Video...")
    
    output_path = output_dir / "final" / "sample.mp4"
    
    # Try to create video with FFmpeg, fall back to placeholder
    try:
        import subprocess
        
        # Check if FFmpeg is available
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Create a simple color video as placeholder
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s=1080x1920:d={script.total_duration}",
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=mono:d={script.total_duration}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-shortest",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
            logger.info(f"   Created video: {output_path}")
        else:
            raise Exception("FFmpeg not available")
            
    except Exception as e:
        # Create placeholder file
        output_path.touch()
        logger.warning(f"   FFmpeg not available, created placeholder: {output_path}")
    
    # Step 6: Generate Thumbnail
    logger.info("\n🖼️ Step 6: Generating Thumbnail...")
    thumb_path = output_dir / "thumbnails" / "sample_thumb.png"
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create simple thumbnail
        img = Image.new("RGB", (1280, 720), color=(40, 40, 60))
        draw = ImageDraw.Draw(img)
        
        # Add title text
        title = script.title[:30] if script.title else "UVG MAX Demo"
        draw.text((640, 360), title, fill="white", anchor="mm")
        
        img.save(thumb_path)
        logger.info(f"   Created thumbnail: {thumb_path}")
        
    except ImportError:
        thumb_path.touch()
        logger.warning(f"   PIL not available, created placeholder: {thumb_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Demo Complete!")
    logger.info("=" * 60)
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Thumbnail: {thumb_path}")
    logger.info(f"   Scenes: {len(script.scenes)}")
    logger.info(f"   Duration: {script.total_duration}s")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="UVG MAX Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_demo.py              # Mock mode (default)
    python scripts/run_demo.py --mock       # Mock mode explicitly
    python scripts/run_demo.py --real       # Real API mode
    python scripts/run_demo.py --parallel   # Parallel scene processing
        """
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Run in mock mode (default, no API keys required)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run in real mode (requires API keys in .env)"
    )
    parser.add_argument(
        "--parallel-scenes",
        action="store_true",
        help="Process scenes in parallel (faster but more memory)"
    )
    
    args = parser.parse_args()
    
    # Real mode takes precedence
    mock_mode = not args.real
    
    print("\n" + "=" * 60)
    print("🚀 UVG MAX - Automated AI Video Generator")
    print("=" * 60 + "\n")
    
    # Setup
    setup_environment(mock_mode)
    output_dir = verify_output_dir()
    
    # Run pipeline
    try:
        output_path = run_demo_pipeline(output_dir, args.parallel_scenes)
        
        print(f"\n✅ Success! Output: {output_path}")
        print("   Copy uvg_output/final/sample.mp4 to check the result.\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
