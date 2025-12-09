# uvg_selector/object_detector.py
"""
Object Detection for UVG MAX.

YOLOv8n wrapper for detecting objects and people in video frames.
Used for clip relevance scoring.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import ultralytics
try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False
    logger.debug("ultralytics not installed - object detection disabled")


# COCO class names (YOLOv8 default)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# High-value objects for video content
HIGH_VALUE_OBJECTS = {
    'person': 1.0,      # People are always valuable
    'face': 1.0,        # Faces (if detected
    'car': 0.6,
    'dog': 0.7,
    'cat': 0.7,
    'laptop': 0.5,
    'cell phone': 0.5,
    'book': 0.4,
    'sports ball': 0.6,
}


class ObjectDetector:
    """
    YOLOv8n object detector wrapper.
    
    Features:
    - Person detection
    - Object presence scoring
    - GPU acceleration when available
    """
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initialize object detector.
        
        Args:
            model_name: YOLO model name or path
        """
        self.model = None
        self.available = False
        
        if not HAVE_YOLO:
            logger.debug("YOLO not available")
            return
        
        try:
            self.model = YOLO(model_name)
            self.available = True
            logger.info(f"Loaded YOLO model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}")
    
    def detect_objects(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: BGR image
            conf_threshold: Minimum confidence
            
        Returns:
            List of detection dicts with class, confidence, bbox
        """
        if not self.available or self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf >= conf_threshold:
                    detections.append({
                        'class_id': cls_id,
                        'class_name': COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else 'unknown',
                        'confidence': conf,
                        'bbox': box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                    })
            
            return detections
            
        except Exception as e:
            logger.debug(f"Detection failed: {e}")
            return []
    
    def detect_persons(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect only persons in frame.
        
        Args:
            frame: BGR image
            conf_threshold: Minimum confidence
            
        Returns:
            List of person detections
        """
        detections = self.detect_objects(frame, conf_threshold)
        return [d for d in detections if d['class_name'] == 'person']
    
    def has_person(self, frame: np.ndarray, conf_threshold: float = 0.3) -> bool:
        """Check if frame contains a person."""
        persons = self.detect_persons(frame, conf_threshold)
        return len(persons) > 0
    
    def get_object_score(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25
    ) -> float:
        """
        Get object presence score for a frame.
        
        Higher score = more valuable objects detected.
        
        Args:
            frame: BGR image
            conf_threshold: Minimum confidence
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not self.available:
            return 0.5  # Neutral fallback
        
        detections = self.detect_objects(frame, conf_threshold)
        
        if not detections:
            return 0.3  # Low score for empty frames
        
        # Calculate weighted score
        total_value = 0.0
        
        for det in detections:
            class_name = det['class_name']
            conf = det['confidence']
            
            # Get object value weight
            value = HIGH_VALUE_OBJECTS.get(class_name, 0.3)
            total_value += value * conf
        
        # Normalize to 0-1 range
        score = min(1.0, total_value / 2.0)  # 2 high-value objects = max
        
        return score
    
    def get_frame_objects_summary(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get summary of objects in frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Dict with object counts and score
        """
        detections = self.detect_objects(frame)
        
        # Count by class
        counts = {}
        for det in detections:
            cls = det['class_name']
            counts[cls] = counts.get(cls, 0) + 1
        
        return {
            'total_objects': len(detections),
            'has_person': 'person' in counts,
            'person_count': counts.get('person', 0),
            'object_counts': counts,
            'score': self.get_object_score(frame)
        }


# Convenience functions
def get_object_detector() -> ObjectDetector:
    """Get object detector instance."""
    return ObjectDetector()


def detect_objects_in_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect objects in a single frame."""
    detector = ObjectDetector()
    return detector.detect_objects(frame)


def has_person_in_frame(frame: np.ndarray) -> bool:
    """Check if frame contains a person."""
    detector = ObjectDetector()
    return detector.has_person(frame)
