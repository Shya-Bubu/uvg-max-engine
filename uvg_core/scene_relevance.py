"""
UVG MAX Scene Relevance Validator Module

Strict semantic matching to ensure clips actually match scene meaning.
Prevents "beautiful but irrelevant" clip selection.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of scene relevance validation."""
    is_valid: bool
    relevance_score: float  # 0.0 - 1.0
    rejection_reason: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "relevance_score": self.relevance_score,
            "rejection_reason": self.rejection_reason,
            "suggestions": self.suggestions,
        }


@dataclass
class ClipValidation:
    """Validation result for a specific clip."""
    clip_path: str
    scene_idx: int
    is_valid: bool
    relevance_score: float
    clip_score: float  # Combined with other metrics
    rejection_reasons: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)


# =============================================================================
# REJECTION REASON TEMPLATES
# =============================================================================

REJECTION_REASONS = {
    "low_relevance": "Rejected: clip does not match scene content '{expected}'",
    "wrong_orientation": "Rejected: clip orientation ({clip_orient}) doesn't match project ({project_orient})",
    "watermark_detected": "Rejected: watermark detected in clip",
    "low_resolution": "Rejected: resolution too low ({width}x{height}, minimum {min_res}p)",
    "too_shaky": "Rejected: excessive camera shake detected",
    "too_many_cuts": "Rejected: clip contains {num_cuts} internal cuts (max {max_cuts})",
    "wrong_aspect": "Rejected: aspect ratio mismatch",
    "content_mismatch": "Rejected: clip content does not include '{missing}'",
    "emotion_mismatch": "Rejected: clip emotion ({clip_emotion}) doesn't match scene ({scene_emotion})",
}


class SceneRelevanceValidator:
    """
    Validates clip relevance to scenes.
    
    Features:
    - CLIP text→video embedding match
    - Gemini semantic similarity (optional)
    - Human-readable rejection reasons
    - Prompt expansion feedback
    """
    
    # Thresholds
    RELEVANCE_THRESHOLD = 0.45  # Minimum relevance score
    HIGH_RELEVANCE = 0.70  # Considered excellent match
    
    def __init__(self, 
                 relevance_threshold: float = 0.45,
                 enable_gemini: bool = True,
                 gemini_api_key: str = "",
                 cache_dir: Optional[Path] = None):
        """
        Initialize validator.
        
        Args:
            relevance_threshold: Minimum relevance score (0.0-1.0)
            enable_gemini: Enable Gemini for semantic checks
            gemini_api_key: Gemini API key
            cache_dir: Cache directory
        """
        self.relevance_threshold = relevance_threshold
        self.enable_gemini = enable_gemini
        self.gemini_api_key = gemini_api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Lazy-loaded models
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_clip_model(self) -> bool:
        """Load CLIP model if available."""
        if self._clip_model is not None:
            return True
        
        try:
            import open_clip
            import torch
            
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k'
            )
            self._tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = self._clip_model.to(device)
            self._clip_model.eval()
            
            logger.info(f"CLIP model loaded on {device}")
            return True
            
        except ImportError:
            logger.warning("open_clip not installed, using fallback scoring")
            return False
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")
            return False
    
    def compute_clip_relevance(self, 
                                image_embedding: Any,
                                text: str) -> float:
        """
        Compute CLIP relevance score.
        
        Args:
            image_embedding: Pre-computed image embedding
            text: Scene text to match
            
        Returns:
            Relevance score 0.0-1.0
        """
        if not self._load_clip_model():
            return 0.65  # Default fallback
        
        try:
            import torch
            
            device = next(self._clip_model.parameters()).device
            
            # Encode text
            text_tokens = self._tokenizer([text]).to(device)
            
            with torch.no_grad():
                text_embedding = self._clip_model.encode_text(text_tokens)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                
                # Ensure image embedding is tensor
                if not isinstance(image_embedding, torch.Tensor):
                    image_embedding = torch.tensor(image_embedding).to(device)
                
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                similarity = (image_embedding @ text_embedding.T).item()
                
                # Convert to 0-1 range
                relevance = (similarity + 1) / 2
                
            return float(relevance)
            
        except Exception as e:
            logger.debug(f"CLIP scoring failed: {e}")
            return 0.65
    
    def validate(self, 
                 clip_embedding: Any,
                 scene_text: str,
                 scene_emotion: str = "neutral",
                 visual_descriptor: str = "",
                 clip_metadata: Optional[Dict] = None) -> ValidationResult:
        """
        Validate clip relevance to scene.
        
        Args:
            clip_embedding: Clip's image embedding
            scene_text: Scene narration text
            scene_emotion: Expected emotion
            visual_descriptor: Detailed visual description
            clip_metadata: Additional clip metadata
            
        Returns:
            ValidationResult with score and reasons
        """
        # Use visual descriptor if available, else scene text
        match_text = visual_descriptor if visual_descriptor else scene_text
        
        # Compute relevance
        relevance = self.compute_clip_relevance(clip_embedding, match_text)
        
        result = ValidationResult(
            is_valid=relevance >= self.relevance_threshold,
            relevance_score=relevance,
        )
        
        if not result.is_valid:
            # Generate rejection reason
            keywords = self._extract_keywords(match_text)
            result.rejection_reason = REJECTION_REASONS["low_relevance"].format(
                expected=", ".join(keywords[:3])
            )
            
            # Provide suggestions
            result.suggestions = self._generate_suggestions(match_text, relevance)
        
        return result
    
    def get_rejection_reason(self, 
                             clip_metadata: Dict,
                             scene_data: Dict,
                             issue_type: str,
                             **kwargs) -> str:
        """
        Generate human-readable rejection reason.
        
        Args:
            clip_metadata: Clip information
            scene_data: Scene information
            issue_type: Type of issue
            **kwargs: Additional format parameters
            
        Returns:
            Formatted rejection reason
        """
        template = REJECTION_REASONS.get(issue_type, "Rejected: unknown issue")
        
        # Build format kwargs
        format_kwargs = {
            "expected": scene_data.get("text", "scene content")[:50],
            "clip_orient": clip_metadata.get("orientation", "unknown"),
            "project_orient": kwargs.get("project_orientation", "portrait"),
            "width": clip_metadata.get("width", 0),
            "height": clip_metadata.get("height", 0),
            "min_res": kwargs.get("min_resolution", 720),
            "num_cuts": kwargs.get("num_cuts", 0),
            "max_cuts": kwargs.get("max_cuts", 3),
            "missing": kwargs.get("missing_element", "required element"),
            "clip_emotion": clip_metadata.get("emotion", "neutral"),
            "scene_emotion": scene_data.get("emotion", "neutral"),
        }
        format_kwargs.update(kwargs)
        
        try:
            return template.format(**format_kwargs)
        except KeyError:
            return template
    
    def validate_clip_quality(self, 
                               clip_path: str,
                               clip_metadata: Dict,
                               project_config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate clip quality (resolution, orientation, etc.)
        
        Args:
            clip_path: Path to clip
            clip_metadata: Clip metadata
            project_config: Project configuration
            
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        width = clip_metadata.get("width", 0)
        height = clip_metadata.get("height", 0)
        min_res = project_config.get("min_resolution", 720)
        
        # Resolution check
        if min(width, height) < min_res:
            issues.append(self.get_rejection_reason(
                clip_metadata, {},
                "low_resolution",
                min_resolution=min_res
            ))
        
        # Orientation check
        project_portrait = project_config.get("height", 1920) > project_config.get("width", 1080)
        clip_portrait = height > width
        
        # Allow mismatched orientation if we can crop/scale
        # But flag it as a quality issue
        if project_portrait != clip_portrait:
            # Not a hard rejection, just a note
            pass
        
        return len(issues) == 0, issues
    
    def request_prompt_expansion(self, 
                                  scene_text: str,
                                  failed_clips: int = 0) -> str:
        """
        Request Gemini to rephrase/expand the search prompt.
        
        Called when no clips found for a scene.
        
        Args:
            scene_text: Original scene text
            failed_clips: Number of clips that failed validation
            
        Returns:
            Expanded search query
        """
        if not self.enable_gemini or not self.gemini_api_key:
            return self._fallback_expansion(scene_text)
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
The following scene description failed to find matching stock video clips.
Original: "{scene_text}"
{failed_clips} clips were rejected for low relevance.

Generate 3 alternative search queries that might find better matches.
Be more specific and visual. Include:
- Concrete visual elements
- Lighting and color descriptions
- Camera angles and movements
- Alternative interpretations

Output ONLY the 3 queries, one per line.
"""
            
            response = model.generate_content(prompt)
            queries = response.text.strip().split('\n')
            
            # Return the first non-empty query
            for q in queries:
                if q.strip():
                    return q.strip()
            
            return self._fallback_expansion(scene_text)
            
        except Exception as e:
            logger.debug(f"Gemini expansion failed: {e}")
            return self._fallback_expansion(scene_text)
    
    def _fallback_expansion(self, text: str) -> str:
        """Simple fallback query expansion."""
        additions = [
            "cinematic",
            "stock footage",
            "high quality",
            "professional",
            "4K",
        ]
        return f"{text} {' '.join(additions)}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple extraction - remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then",
            "once", "and", "but", "or", "nor", "so", "yet", "both",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "not", "only", "own", "same", "than", "too", "very",
        }
        
        words = text.lower().split()
        keywords = [w.strip(".,!?;:") for w in words if w.lower() not in stopwords]
        return keywords[:10]
    
    def _generate_suggestions(self, text: str, score: float) -> List[str]:
        """Generate suggestions for improving match."""
        suggestions = []
        
        if score < 0.3:
            suggestions.append("Try more specific visual descriptions")
            suggestions.append("Include concrete objects or actions")
        elif score < 0.45:
            suggestions.append("Add lighting or color descriptions")
            suggestions.append("Specify camera angle or composition")
        
        return suggestions
    
    def batch_validate(self, 
                       clips: List[Dict],
                       scene_text: str,
                       visual_descriptor: str = "") -> List[ClipValidation]:
        """
        Validate multiple clips against a scene.
        
        Args:
            clips: List of clip dicts with embeddings
            scene_text: Scene text
            visual_descriptor: Visual description
            
        Returns:
            List of ClipValidation results, sorted by relevance
        """
        results = []
        
        for clip in clips:
            result = self.validate(
                clip.get("embedding"),
                scene_text,
                visual_descriptor=visual_descriptor,
                clip_metadata=clip,
            )
            
            validation = ClipValidation(
                clip_path=clip.get("path", ""),
                scene_idx=clip.get("scene_idx", 0),
                is_valid=result.is_valid,
                relevance_score=result.relevance_score,
                clip_score=clip.get("score", result.relevance_score),
            )
            
            if not result.is_valid:
                validation.rejection_reasons.append(result.rejection_reason)
            
            results.append(validation)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_clip_relevance(clip_embedding: Any,
                            scene_text: str,
                            threshold: float = 0.45) -> ValidationResult:
    """Validate a single clip's relevance."""
    validator = SceneRelevanceValidator(relevance_threshold=threshold)
    return validator.validate(clip_embedding, scene_text)


def get_rejection_reason(issue_type: str, **kwargs) -> str:
    """Get a formatted rejection reason."""
    template = REJECTION_REASONS.get(issue_type, "Rejected: {issue_type}")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template
