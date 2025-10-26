"""
Modern Eye Tracking Implementation

This module provides a comprehensive eye tracking system using computer vision
techniques with dlib for facial landmark detection and OpenCV for image processing.
"""

import cv2
import dlib
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
from loguru import logger
import time


@dataclass
class EyeTrackingConfig:
    """Configuration class for eye tracking parameters."""
    predictor_path: str
    confidence_threshold: float
    device_id: int
    width: int
    height: int
    fps: int
    left_eye_points: List[int]
    right_eye_points: List[int]
    pupil_threshold: int
    min_contour_area: int
    pupil_radius: int
    text_scale: float
    text_thickness: int
    colors: Dict[str, List[int]]


@dataclass
class EyePosition:
    """Data class to store eye position information."""
    center: Tuple[int, int]
    confidence: float
    timestamp: float


class EyeTracker:
    """
    Modern eye tracking implementation with enhanced accuracy and features.
    
    This class provides real-time eye tracking capabilities using dlib's
    facial landmark detection and OpenCV for image processing.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the eye tracker with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self._load_predictor()
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Eye tracker initialized successfully")
    
    def _load_config(self, config_path: str) -> EyeTrackingConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            return EyeTrackingConfig(
                predictor_path=config_data['model']['predictor_path'],
                confidence_threshold=config_data['model']['confidence_threshold'],
                device_id=config_data['camera']['device_id'],
                width=config_data['camera']['width'],
                height=config_data['camera']['height'],
                fps=config_data['camera']['fps'],
                left_eye_points=config_data['eye_tracking']['left_eye_points'],
                right_eye_points=config_data['eye_tracking']['right_eye_points'],
                pupil_threshold=config_data['eye_tracking']['pupil_threshold'],
                min_contour_area=config_data['eye_tracking']['min_contour_area'],
                pupil_radius=config_data['visualization']['pupil_radius'],
                text_scale=config_data['visualization']['text_scale'],
                text_thickness=config_data['visualization']['text_thickness'],
                colors=config_data['visualization']['colors']
            )
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> EyeTrackingConfig:
        """Get default configuration."""
        return EyeTrackingConfig(
            predictor_path="data/models/shape_predictor_68_face_landmarks.dat",
            confidence_threshold=0.5,
            device_id=0,
            width=640,
            height=480,
            fps=30,
            left_eye_points=list(range(36, 42)),
            right_eye_points=list(range(42, 48)),
            pupil_threshold=70,
            min_contour_area=50,
            pupil_radius=5,
            text_scale=0.5,
            text_thickness=1,
            colors={
                'left_eye': [0, 255, 0],
                'right_eye': [255, 0, 0],
                'face_box': [255, 255, 0]
            }
        )
    
    def _load_predictor(self) -> dlib.shape_predictor:
        """Load the facial landmark predictor."""
        predictor_path = Path(self.config.predictor_path)
        if not predictor_path.exists():
            logger.error(f"Predictor file not found: {predictor_path}")
            logger.info("Please download shape_predictor_68_face_landmarks.dat from dlib's model zoo")
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
        
        return dlib.shape_predictor(str(predictor_path))
    
    def _get_eye_center(self, landmarks: List[Tuple[int, int]], 
                       eye_points: List[int], 
                       gray_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Calculate the center of an eye (pupil) using contour detection.
        
        Args:
            landmarks: List of facial landmark coordinates
            eye_points: Indices of eye landmark points
            gray_frame: Grayscale frame for processing
            
        Returns:
            Tuple of (x, y) coordinates of eye center, or None if not found
        """
        try:
            # Extract eye region coordinates
            eye_region = np.array([landmarks[i] for i in eye_points], np.int32)
            x, y, w, h = cv2.boundingRect(eye_region)
            
            # Extract eye ROI
            roi = gray_frame[y:y+h, x:x+w]
            if roi.size == 0:
                return None
            
            # Enhance contrast
            roi = cv2.equalizeHist(roi)
            
            # Threshold to find dark regions (pupil)
            _, threshold = cv2.threshold(roi, self.config.pupil_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area
                valid_contours = [c for c in contours if cv2.contourArea(c) > self.config.min_contour_area]
                
                if valid_contours:
                    # Use largest valid contour as pupil
                    contour = max(valid_contours, key=cv2.contourArea)
                    (cx, cy), _ = cv2.minEnclosingCircle(contour)
                    return int(x + cx), int(y + cy)
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating eye center: {e}")
            return None
    
    def _calculate_gaze_direction(self, left_center: Tuple[int, int], 
                                right_center: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate gaze direction based on eye positions.
        
        Args:
            left_center: Left eye center coordinates
            right_center: Right eye center coordinates
            
        Returns:
            Dictionary containing gaze direction information
        """
        if not left_center or not right_center:
            return {"direction": "unknown", "confidence": 0.0}
        
        # Calculate midpoint between eyes
        eye_midpoint = ((left_center[0] + right_center[0]) // 2,
                       (left_center[1] + right_center[1]) // 2)
        
        # Simple gaze direction estimation based on eye position relative to face center
        # This is a basic implementation - more sophisticated methods would use
        # additional facial landmarks and calibration
        
        gaze_info = {
            "eye_midpoint": eye_midpoint,
            "left_eye": left_center,
            "right_eye": right_center,
            "direction": "center",  # Placeholder for more sophisticated gaze estimation
            "confidence": 0.8  # Placeholder confidence
        }
        
        return gaze_info
    
    def start_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.config.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            logger.info(f"Camera initialized: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for eye tracking.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing tracking results
        """
        results = {
            "left_eye": None,
            "right_eye": None,
            "gaze_info": None,
            "faces_detected": 0,
            "fps": 0,
            "timestamp": time.time()
        }
        
        try:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            results["faces_detected"] = len(faces)
            
            if faces:
                # Process first detected face
                face = faces[0]
                
                # Get facial landmarks
                shape = self.predictor(gray, face)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                
                # Track both eyes
                left_center = self._get_eye_center(landmarks, self.config.left_eye_points, gray)
                right_center = self._get_eye_center(landmarks, self.config.right_eye_points, gray)
                
                if left_center:
                    results["left_eye"] = EyePosition(
                        center=left_center,
                        confidence=0.8,  # Placeholder confidence
                        timestamp=time.time()
                    )
                
                if right_center:
                    results["right_eye"] = EyePosition(
                        center=right_center,
                        confidence=0.8,  # Placeholder confidence
                        timestamp=time.time()
                    )
                
                # Calculate gaze direction
                if left_center and right_center:
                    results["gaze_info"] = self._calculate_gaze_direction(left_center, right_center)
            
            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                results["fps"] = self.frame_count / elapsed_time
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw tracking results on the frame.
        
        Args:
            frame: Input frame
            results: Tracking results from process_frame
            
        Returns:
            Frame with drawn annotations
        """
        try:
            # Draw left eye
            if results["left_eye"]:
                center = results["left_eye"].center
                cv2.circle(frame, center, self.config.pupil_radius, 
                          self.config.colors['left_eye'], -1)
                cv2.putText(frame, f"L: {center}", 
                           (center[0] + 10, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, self.config.text_scale,
                           self.config.colors['left_eye'], self.config.text_thickness)
            
            # Draw right eye
            if results["right_eye"]:
                center = results["right_eye"].center
                cv2.circle(frame, center, self.config.pupil_radius,
                          self.config.colors['right_eye'], -1)
                cv2.putText(frame, f"R: {center}",
                           (center[0] + 10, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, self.config.text_scale,
                           self.config.colors['right_eye'], self.config.text_thickness)
            
            # Draw FPS
            fps_text = f"FPS: {results['fps']:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw face count
            faces_text = f"Faces: {results['faces_detected']}"
            cv2.putText(frame, faces_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error drawing results: {e}")
        
        return frame
    
    def run_realtime(self) -> None:
        """Run real-time eye tracking."""
        if not self.start_camera():
            return
        
        logger.info("Starting real-time eye tracking. Press ESC to exit.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw results
                frame = self.draw_results(frame, results)
                
                # Display frame
                cv2.imshow("Eye Tracking", frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            logger.info("Eye tracking stopped by user")
        except Exception as e:
            logger.error(f"Error in real-time tracking: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources cleaned up")


def main():
    """Main function to run the eye tracker."""
    try:
        tracker = EyeTracker()
        tracker.run_realtime()
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
