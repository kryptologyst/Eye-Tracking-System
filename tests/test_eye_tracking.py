"""
Unit Tests for Eye Tracking System

This module contains comprehensive unit tests for the eye tracking functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from eye_tracking import EyeTracker, EyeTrackingConfig, EyePosition


class TestEyeTrackingConfig:
    """Test cases for EyeTrackingConfig."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = EyeTrackingConfig(
            predictor_path="test.dat",
            confidence_threshold=0.5,
            device_id=0,
            width=640,
            height=480,
            fps=30,
            left_eye_points=[36, 37, 38, 39, 40, 41],
            right_eye_points=[42, 43, 44, 45, 46, 47],
            pupil_threshold=70,
            min_contour_area=50,
            pupil_radius=5,
            text_scale=0.5,
            text_thickness=1,
            colors={"left_eye": [0, 255, 0], "right_eye": [255, 0, 0]}
        )
        
        assert config.predictor_path == "test.dat"
        assert config.confidence_threshold == 0.5
        assert config.device_id == 0
        assert config.width == 640
        assert config.height == 480
        assert config.fps == 30


class TestEyePosition:
    """Test cases for EyePosition."""
    
    def test_eye_position_creation(self):
        """Test EyePosition data class."""
        position = EyePosition(
            center=(100, 200),
            confidence=0.8,
            timestamp=1234567890.0
        )
        
        assert position.center == (100, 200)
        assert position.confidence == 0.8
        assert position.timestamp == 1234567890.0


class TestEyeTracker:
    """Test cases for EyeTracker."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return EyeTrackingConfig(
            predictor_path="test.dat",
            confidence_threshold=0.5,
            device_id=0,
            width=640,
            height=480,
            fps=30,
            left_eye_points=[36, 37, 38, 39, 40, 41],
            right_eye_points=[42, 43, 44, 45, 46, 47],
            pupil_threshold=70,
            min_contour_area=50,
            pupil_radius=5,
            text_scale=0.5,
            text_thickness=1,
            colors={"left_eye": [0, 255, 0], "right_eye": [255, 0, 0]}
        )
    
    @pytest.fixture
    def mock_tracker(self, mock_config):
        """Create a mock tracker with mocked dependencies."""
        with patch('eye_tracking.tracker.dlib.get_frontal_face_detector'), \
             patch('eye_tracking.tracker.dlib.shape_predictor'), \
             patch('eye_tracking.tracker.Path.exists', return_value=True):
            
            tracker = EyeTracker.__new__(EyeTracker)
            tracker.config = mock_config
            tracker.detector = Mock()
            tracker.predictor = Mock()
            tracker.cap = None
            tracker.frame_count = 0
            tracker.start_time = 0
            
            return tracker
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            tracker = EyeTracker.__new__(EyeTracker)
            config = tracker._load_config("nonexistent.yaml")
            
            assert config.predictor_path == "data/models/shape_predictor_68_face_landmarks.dat"
            assert config.confidence_threshold == 0.5
    
    def test_load_config_success(self):
        """Test successful config loading."""
        config_data = {
            'model': {'predictor_path': 'test.dat', 'confidence_threshold': 0.7},
            'camera': {'device_id': 1, 'width': 800, 'height': 600, 'fps': 25},
            'eye_tracking': {
                'left_eye_points': [36, 37, 38, 39, 40, 41],
                'right_eye_points': [42, 43, 44, 45, 46, 47],
                'pupil_threshold': 80,
                'min_contour_area': 60
            },
            'visualization': {
                'pupil_radius': 6,
                'text_scale': 0.6,
                'text_thickness': 2,
                'colors': {'left_eye': [0, 255, 0], 'right_eye': [255, 0, 0]}
            }
        }
        
        with patch('builtins.open', mock_open_config(config_data)):
            tracker = EyeTracker.__new__(EyeTracker)
            config = tracker._load_config("test.yaml")
            
            assert config.predictor_path == 'test.dat'
            assert config.confidence_threshold == 0.7
            assert config.device_id == 1
            assert config.width == 800
            assert config.height == 600
            assert config.fps == 25
    
    def test_get_eye_center_success(self, mock_tracker):
        """Test successful eye center calculation."""
        # Create mock landmarks
        landmarks = [(100, 100), (110, 100), (120, 100), (130, 100), (140, 100), (150, 100)]
        eye_points = [0, 1, 2, 3, 4, 5]
        
        # Create mock gray frame
        gray_frame = np.zeros((200, 200), dtype=np.uint8)
        
        # Mock cv2 functions
        with patch('cv2.boundingRect', return_value=(100, 100, 50, 20)), \
             patch('cv2.equalizeHist', return_value=gray_frame[100:120, 100:150]), \
             patch('cv2.threshold', return_value=(None, np.zeros((20, 50), dtype=np.uint8))), \
             patch('cv2.findContours', return_value=([np.array([[[10, 10]], [[20, 20]]])], None)), \
             patch('cv2.contourArea', return_value=100), \
             patch('cv2.minEnclosingCircle', return_value=((25, 10), 5)):
            
            result = mock_tracker._get_eye_center(landmarks, eye_points, gray_frame)
            assert result == (125, 110)  # x + cx, y + cy
    
    def test_get_eye_center_no_contours(self, mock_tracker):
        """Test eye center calculation when no contours found."""
        landmarks = [(100, 100), (110, 100), (120, 100), (130, 100), (140, 100), (150, 100)]
        eye_points = [0, 1, 2, 3, 4, 5]
        gray_frame = np.zeros((200, 200), dtype=np.uint8)
        
        with patch('cv2.boundingRect', return_value=(100, 100, 50, 20)), \
             patch('cv2.equalizeHist', return_value=gray_frame[100:120, 100:150]), \
             patch('cv2.threshold', return_value=(None, np.zeros((20, 50), dtype=np.uint8))), \
             patch('cv2.findContours', return_value=([], None)):
            
            result = mock_tracker._get_eye_center(landmarks, eye_points, gray_frame)
            assert result is None
    
    def test_calculate_gaze_direction(self, mock_tracker):
        """Test gaze direction calculation."""
        left_center = (100, 200)
        right_center = (120, 200)
        
        result = mock_tracker._calculate_gaze_direction(left_center, right_center)
        
        assert "eye_midpoint" in result
        assert "left_eye" in result
        assert "right_eye" in result
        assert "direction" in result
        assert "confidence" in result
        assert result["eye_midpoint"] == (110, 200)
    
    def test_calculate_gaze_direction_missing_eyes(self, mock_tracker):
        """Test gaze direction calculation with missing eye data."""
        result = mock_tracker._calculate_gaze_direction(None, None)
        
        assert result["direction"] == "unknown"
        assert result["confidence"] == 0.0
    
    def test_start_camera_success(self, mock_tracker):
        """Test successful camera initialization."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            result = mock_tracker.start_camera()
            
            assert result is True
            assert mock_tracker.cap == mock_cap
            mock_cap.set.assert_called()
    
    def test_start_camera_failure(self, mock_tracker):
        """Test camera initialization failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            result = mock_tracker.start_camera()
            
            assert result is False
    
    def test_process_frame_no_faces(self, mock_tracker):
        """Test frame processing with no faces detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_tracker.detector.return_value = []
        
        result = mock_tracker.process_frame(frame)
        
        assert result["faces_detected"] == 0
        assert result["left_eye"] is None
        assert result["right_eye"] is None
        assert result["gaze_info"] is None
    
    def test_process_frame_with_face(self, mock_tracker):
        """Test frame processing with face detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_face = Mock()
        mock_shape = Mock()
        
        # Mock facial landmarks
        landmarks = [(100, 100) for _ in range(68)]
        for i in range(68):
            mock_shape.part.return_value.x = 100
            mock_shape.part.return_value.y = 100
        
        mock_tracker.detector.return_value = [mock_face]
        mock_tracker.predictor.return_value = mock_shape
        
        with patch.object(mock_tracker, '_get_eye_center', return_value=(150, 200)):
            result = mock_tracker.process_frame(frame)
            
            assert result["faces_detected"] == 1
            assert result["left_eye"] is not None
            assert result["right_eye"] is not None
            assert result["gaze_info"] is not None
    
    def test_draw_results(self, mock_tracker):
        """Test drawing results on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = {
            "left_eye": EyePosition(center=(100, 200), confidence=0.8, timestamp=0),
            "right_eye": EyePosition(center=(150, 200), confidence=0.8, timestamp=0),
            "fps": 30.0,
            "faces_detected": 1
        }
        
        with patch('cv2.circle'), \
             patch('cv2.putText'):
            
            result_frame = mock_tracker.draw_results(frame, results)
            
            assert result_frame is not None
            assert result_frame.shape == frame.shape
    
    def test_cleanup(self, mock_tracker):
        """Test cleanup functionality."""
        mock_cap = Mock()
        mock_tracker.cap = mock_cap
        
        with patch('cv2.destroyAllWindows'):
            mock_tracker.cleanup()
            
            mock_cap.release.assert_called_once()


def mock_open_config(config_data):
    """Helper function to mock config file opening."""
    def mock_open(file_path, mode='r'):
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        mock_file.read.return_value = yaml.dump(config_data)
        return mock_file
    return mock_open


class TestSyntheticDataGenerator:
    """Test cases for synthetic data generation."""
    
    def test_generate_eye_movement_pattern_random(self):
        """Test random eye movement pattern generation."""
        from data.synthetic.generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator(duration=1.0, fps=10.0)
        movements = generator.generate_eye_movement_pattern("random")
        
        assert len(movements) == 10
        assert all(isinstance(move, tuple) and len(move) == 2 for move in movements)
    
    def test_generate_eye_movement_pattern_circular(self):
        """Test circular eye movement pattern generation."""
        from data.synthetic.generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator(duration=1.0, fps=10.0)
        movements = generator.generate_eye_movement_pattern("circular")
        
        assert len(movements) == 10
        # Check that movements form a circle (approximately)
        first_move = movements[0]
        last_move = movements[-1]
        assert abs(first_move[0] - last_move[0]) < 1.0  # Should be close to starting point
    
    def test_generate_blink_pattern(self):
        """Test blink pattern generation."""
        from data.synthetic.generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator(duration=10.0, fps=30.0)
        blinks = generator.generate_blink_pattern()
        
        assert len(blinks) == 300
        assert all(isinstance(blink, bool) for blink in blinks)
        # Should have some blinks
        assert any(blinks)
    
    def test_generate_gaze_directions(self):
        """Test gaze direction generation."""
        from data.synthetic.generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator()
        movements = [(0, 0), (15, 0), (-15, 0), (0, 15), (0, -15)]
        directions = generator.generate_gaze_directions(movements)
        
        expected_directions = ["center", "right", "left", "down", "up"]
        assert directions == expected_directions
    
    def test_generate_synthetic_data(self):
        """Test complete synthetic data generation."""
        from data.synthetic.generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator(duration=1.0, fps=10.0)
        data = generator.generate_synthetic_data(pattern_type="random")
        
        assert len(data) == 10
        assert all(hasattr(point, 'timestamp') for point in data)
        assert all(hasattr(point, 'left_eye_x') for point in data)
        assert all(hasattr(point, 'right_eye_y') for point in data)
        assert all(hasattr(point, 'blink_detected') for point in data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
