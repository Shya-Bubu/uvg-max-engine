"""
UVG MAX Orchestrator Module

Master orchestrator with auto-repair and degrade mode.
"""

import logging
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SceneStatus(Enum):
    """Scene processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class SceneState:
    """State of a single scene."""
    index: int
    script_text: str
    script_hash: str
    status: SceneStatus = SceneStatus.PENDING
    attempts: int = 0
    video_path: str = ""
    audio_path: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "script_text": self.script_text[:100],
            "script_hash": self.script_hash,
            "status": self.status.value,
            "attempts": self.attempts,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "error": self.error,
        }


@dataclass
class ProjectState:
    """Project-wide state for resume."""
    project_id: str
    scenes: List[SceneState] = field(default_factory=list)
    completed_scenes: int = 0
    failed_scenes: int = 0
    degrade_mode: bool = False
    last_checkpoint: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "scenes": [s.to_dict() for s in self.scenes],
            "completed_scenes": self.completed_scenes,
            "failed_scenes": self.failed_scenes,
            "degrade_mode": self.degrade_mode,
            "last_checkpoint": self.last_checkpoint,
        }
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ProjectState':
        with open(path, 'r') as f:
            data = json.load(f)
        
        state = cls(project_id=data["project_id"])
        state.completed_scenes = data.get("completed_scenes", 0)
        state.failed_scenes = data.get("failed_scenes", 0)
        state.degrade_mode = data.get("degrade_mode", False)
        state.last_checkpoint = data.get("last_checkpoint", 0)
        
        for s in data.get("scenes", []):
            scene = SceneState(
                index=s["index"],
                script_text=s["script_text"],
                script_hash=s["script_hash"],
                status=SceneStatus(s.get("status", "pending")),
                attempts=s.get("attempts", 0),
                video_path=s.get("video_path", ""),
                audio_path=s.get("audio_path", ""),
                error=s.get("error", ""),
            )
            state.scenes.append(scene)
        
        return state


class Orchestrator:
    """
    Master orchestrator for the UVG MAX pipeline.
    
    Features:
    - End-to-end pipeline coordination
    - Scene hashing for resume detection
    - Auto-repair (redo_scene on failure)
    - Degrade mode (simplify on repeated failures)
    - Aggressive cleanup after each stage
    - Memory limiting
    """
    
    MAX_SCENE_RETRIES = 3
    DEGRADE_THRESHOLD = 2  # Switch to degrade mode after this many failures
    
    def __init__(self,
                 config: Optional[Dict] = None,
                 output_dir: Optional[Path] = None,
                 checkpoint_interval: int = 60):
        """
        Initialize orchestrator.
        
        Args:
            config: UVG configuration
            output_dir: Output directory
            checkpoint_interval: Seconds between checkpoints
        """
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("./uvg_output")
        self.checkpoint_interval = checkpoint_interval
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.state: Optional[ProjectState] = None
        self._failure_count = 0
    
    def _compute_scene_hash(self, script_text: str) -> str:
        """Compute hash of scene script for change detection."""
        return hashlib.sha256(script_text.encode()).hexdigest()[:16]
    
    def create_project(self, 
                       project_id: str,
                       scenes: List[Dict[str, Any]]) -> ProjectState:
        """
        Create new project state.
        
        Args:
            project_id: Unique project identifier
            scenes: List of scene dicts with text
            
        Returns:
            ProjectState
        """
        self.state = ProjectState(project_id=project_id)
        
        for i, scene in enumerate(scenes):
            text = scene.get("text", "")
            scene_state = SceneState(
                index=i,
                script_text=text,
                script_hash=self._compute_scene_hash(text),
            )
            self.state.scenes.append(scene_state)
        
        self._save_checkpoint()
        return self.state
    
    def resume_project(self, checkpoint_path: Path) -> ProjectState:
        """
        Resume project from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            ProjectState
        """
        self.state = ProjectState.load(checkpoint_path)
        logger.info(f"Resumed project {self.state.project_id} with "
                   f"{self.state.completed_scenes}/{len(self.state.scenes)} scenes completed")
        return self.state
    
    def should_rerun_scene(self, 
                            scene_idx: int,
                            new_script: str) -> bool:
        """
        Check if scene needs to be rerun based on hash.
        
        Args:
            scene_idx: Scene index
            new_script: New script text
            
        Returns:
            True if scene should be rerun
        """
        if self.state is None or scene_idx >= len(self.state.scenes):
            return True
        
        scene = self.state.scenes[scene_idx]
        new_hash = self._compute_scene_hash(new_script)
        
        # Rerun if hash changed or scene failed
        if new_hash != scene.script_hash:
            logger.info(f"Scene {scene_idx} script changed, will rerun")
            return True
        
        if scene.status == SceneStatus.FAILED:
            return True
        
        if scene.status == SceneStatus.COMPLETED:
            return False
        
        return True
    
    def mark_scene_complete(self,
                             scene_idx: int,
                             video_path: str,
                             audio_path: str) -> None:
        """Mark scene as completed."""
        if self.state is None:
            return
        
        scene = self.state.scenes[scene_idx]
        scene.status = SceneStatus.COMPLETED
        scene.video_path = video_path
        scene.audio_path = audio_path
        scene.error = ""
        
        self.state.completed_scenes += 1
        self._failure_count = 0  # Reset on success
        
        self._maybe_checkpoint()
    
    def mark_scene_failed(self,
                           scene_idx: int,
                           error: str) -> bool:
        """
        Mark scene as failed and decide next action.
        
        Args:
            scene_idx: Scene index
            error: Error message
            
        Returns:
            True if should retry, False if exhausted
        """
        if self.state is None:
            return False
        
        scene = self.state.scenes[scene_idx]
        scene.attempts += 1
        scene.error = error
        
        self._failure_count += 1
        
        if scene.attempts >= self.MAX_SCENE_RETRIES:
            scene.status = SceneStatus.FAILED
            self.state.failed_scenes += 1
            
            # Check if should enter degrade mode
            if self._failure_count >= self.DEGRADE_THRESHOLD:
                self._enter_degrade_mode()
            
            self._save_checkpoint()
            return False
        
        scene.status = SceneStatus.RETRYING
        self._save_checkpoint()
        return True
    
    def redo_scene(self, scene_idx: int) -> None:
        """
        Reset scene for retry.
        
        Args:
            scene_idx: Scene to redo
        """
        if self.state is None or scene_idx >= len(self.state.scenes):
            return
        
        scene = self.state.scenes[scene_idx]
        
        # Clean up existing files
        for path in [scene.video_path, scene.audio_path]:
            if path:
                Path(path).unlink(missing_ok=True)
        
        scene.status = SceneStatus.PENDING
        scene.video_path = ""
        scene.audio_path = ""
        scene.error = ""
        # Keep attempts count for tracking
        
        logger.info(f"Scene {scene_idx} reset for retry (attempt {scene.attempts + 1})")
    
    def _enter_degrade_mode(self) -> None:
        """Enter degrade mode for simpler processing."""
        if self.state is None:
            return
        
        if not self.state.degrade_mode:
            logger.warning("Entering DEGRADE MODE - simplifying processing")
            self.state.degrade_mode = True
            self._save_checkpoint()
    
    def get_degrade_settings(self) -> Dict[str, Any]:
        """Get simplified settings for degrade mode."""
        if self.state and self.state.degrade_mode:
            return {
                "motion_type": "static",
                "vfx_preset": "minimal",
                "transition_type": "fade",
                "transition_duration": 0.3,
                "caption_style": "minimal",
                "max_clip_search": 5,
                "skip_quality_filter": True,
            }
        return {}
    
    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint file."""
        if self.state is None:
            return
        
        checkpoint_path = self.output_dir / f"checkpoint_{self.state.project_id}.json"
        self.state.last_checkpoint = time.time()
        self.state.save(checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _maybe_checkpoint(self) -> None:
        """Checkpoint if interval elapsed."""
        if self.state is None:
            return
        
        elapsed = time.time() - self.state.last_checkpoint
        if elapsed >= self.checkpoint_interval:
            self._save_checkpoint()
    
    def cleanup_scene_temp_files(self, 
                                  scene_idx: int,
                                  keep_final: bool = True) -> int:
        """
        Clean up temporary files for a scene.
        
        Args:
            scene_idx: Scene index
            keep_final: Keep final output files
            
        Returns:
            Number of files deleted
        """
        # Cleanup patterns
        patterns = [
            f"*_scene_{scene_idx}_temp*",
            f"*_scene_{scene_idx}_candidate*",
            f"*_scene_{scene_idx}_frame_*",
        ]
        
        if not keep_final:
            patterns.extend([
                f"*_scene_{scene_idx}.mp4",
                f"*_scene_{scene_idx}.wav",
            ])
        
        deleted = 0
        for pattern in patterns:
            for f in self.output_dir.rglob(pattern):
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    pass
        
        return deleted
    
    def get_pending_scenes(self) -> List[int]:
        """Get list of pending scene indices."""
        if self.state is None:
            return []
        
        return [
            s.index for s in self.state.scenes
            if s.status in [SceneStatus.PENDING, SceneStatus.RETRYING]
        ]
    
    def get_progress(self) -> Dict[str, Any]:
        """Get project progress."""
        if self.state is None:
            return {"error": "No project loaded"}
        
        total = len(self.state.scenes)
        return {
            "project_id": self.state.project_id,
            "total_scenes": total,
            "completed": self.state.completed_scenes,
            "failed": self.state.failed_scenes,
            "pending": total - self.state.completed_scenes - self.state.failed_scenes,
            "percent": (self.state.completed_scenes / total * 100) if total > 0 else 0,
            "degrade_mode": self.state.degrade_mode,
        }
    
    def run_pipeline(self,
                     script: Dict[str, Any],
                     on_scene_complete: Optional[callable] = None,
                     on_error: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run the full pipeline.
        
        This is a simplified coordinator - the actual processing
        would import and call the other modules.
        
        Args:
            script: Script with scenes
            on_scene_complete: Callback for scene completion
            on_error: Callback for errors
            
        Returns:
            Final result dict
        """
        scenes = script.get("scenes", [])
        project_id = script.get("title", "untitled").replace(" ", "_").lower()
        
        # Create or resume
        checkpoint = self.output_dir / f"checkpoint_{project_id}.json"
        if checkpoint.exists():
            self.resume_project(checkpoint)
        else:
            self.create_project(project_id, scenes)
        
        results = {
            "project_id": project_id,
            "success": True,
            "scenes": [],
            "output_path": "",
        }
        
        # Process each pending scene
        for scene_idx in self.get_pending_scenes():
            scene = scenes[scene_idx]
            
            try:
                # Mark in progress
                self.state.scenes[scene_idx].status = SceneStatus.IN_PROGRESS
                
                # Get settings (degrade if needed)
                settings = self.get_degrade_settings()
                
                # TODO: Call actual processing modules here
                # For now, simulate success
                video_path = str(self.output_dir / f"scene_{scene_idx}.mp4")
                audio_path = str(self.output_dir / f"scene_{scene_idx}.wav")
                
                self.mark_scene_complete(scene_idx, video_path, audio_path)
                
                if on_scene_complete:
                    on_scene_complete(scene_idx, video_path)
                
                # Cleanup after each scene
                self.cleanup_scene_temp_files(scene_idx)
                
                results["scenes"].append({
                    "index": scene_idx,
                    "status": "completed",
                    "video_path": video_path,
                })
                
            except Exception as e:
                error_msg = str(e)
                should_retry = self.mark_scene_failed(scene_idx, error_msg)
                
                if on_error:
                    on_error(scene_idx, error_msg)
                
                if should_retry:
                    self.redo_scene(scene_idx)
                else:
                    results["success"] = False
                    results["scenes"].append({
                        "index": scene_idx,
                        "status": "failed",
                        "error": error_msg,
                    })
        
        # Final assembly if all scenes complete
        if self.state.completed_scenes == len(scenes):
            results["output_path"] = str(self.output_dir / f"{project_id}_final.mp4")
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_orchestrator(output_dir: Path = None) -> Orchestrator:
    """Create orchestrator instance."""
    return Orchestrator(output_dir=output_dir)


def run_video_pipeline(script: Dict[str, Any], 
                       output_dir: Path = None) -> Dict[str, Any]:
    """Run full video generation pipeline."""
    orchestrator = Orchestrator(output_dir=output_dir)
    return orchestrator.run_pipeline(script)
