"""
Synthetic Dataset Generator for Eye Tracking

This module generates synthetic eye tracking data for testing and demonstration
purposes when real camera data is not available.
"""

import numpy as np
import cv2
import pandas as pd
from typing import List, Tuple, Dict, Any
import random
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import time


@dataclass
class SyntheticEyeData:
    """Data class for synthetic eye tracking data."""
    timestamp: float
    left_eye_x: float
    left_eye_y: float
    right_eye_x: float
    right_eye_y: float
    left_confidence: float
    right_confidence: float
    gaze_direction: str
    blink_detected: bool
    fps: float


class SyntheticDatasetGenerator:
    """Generator for synthetic eye tracking datasets."""
    
    def __init__(self, 
                 frame_width: int = 640, 
                 frame_height: int = 480,
                 duration: float = 30.0,
                 fps: float = 30.0):
        """
        Initialize the synthetic dataset generator.
        
        Args:
            frame_width: Width of the synthetic frames
            frame_height: Height of the synthetic frames
            duration: Duration of the dataset in seconds
            fps: Frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        
        # Eye movement parameters
        self.eye_radius = 20  # Radius of eye movement
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Eye positions (relative to face center)
        self.left_eye_offset = (-30, -10)  # Left eye offset from face center
        self.right_eye_offset = (30, -10)   # Right eye offset from face center
        
    def generate_eye_movement_pattern(self, pattern_type: str = "random") -> List[Tuple[float, float]]:
        """
        Generate eye movement patterns.
        
        Args:
            pattern_type: Type of movement pattern ('random', 'circular', 'horizontal', 'vertical')
            
        Returns:
            List of (x, y) coordinates for eye movement
        """
        movements = []
        
        for i in range(self.total_frames):
            t = i / self.fps
            
            if pattern_type == "random":
                # Random movement with some smoothing
                if i == 0:
                    x, y = 0, 0
                else:
                    prev_x, prev_y = movements[-1]
                    x = prev_x + random.uniform(-2, 2)
                    y = prev_y + random.uniform(-2, 2)
                    x = np.clip(x, -self.eye_radius, self.eye_radius)
                    y = np.clip(y, -self.eye_radius, self.eye_radius)
            
            elif pattern_type == "circular":
                # Circular movement
                angle = 2 * np.pi * t / 5  # Complete circle every 5 seconds
                x = self.eye_radius * np.cos(angle)
                y = self.eye_radius * np.sin(angle)
            
            elif pattern_type == "horizontal":
                # Horizontal movement
                x = self.eye_radius * np.sin(2 * np.pi * t / 3)
                y = 0
            
            elif pattern_type == "vertical":
                # Vertical movement
                x = 0
                y = self.eye_radius * np.sin(2 * np.pi * t / 3)
            
            else:
                x, y = 0, 0
            
            movements.append((x, y))
        
        return movements
    
    def generate_blink_pattern(self) -> List[bool]:
        """
        Generate realistic blink patterns.
        
        Returns:
            List of boolean values indicating blink events
        """
        blinks = [False] * self.total_frames
        
        # Average blink rate: 15-20 blinks per minute
        blink_rate = random.uniform(15, 20) / 60  # blinks per second
        total_blinks = int(self.duration * blink_rate)
        
        for _ in range(total_blinks):
            # Random blink time
            blink_frame = random.randint(0, self.total_frames - 1)
            
            # Blink duration: 3-5 frames (100-167ms at 30fps)
            blink_duration = random.randint(3, 5)
            
            for i in range(blink_duration):
                if blink_frame + i < self.total_frames:
                    blinks[blink_frame + i] = True
        
        return blinks
    
    def generate_gaze_directions(self, movements: List[Tuple[float, float]]) -> List[str]:
        """
        Generate gaze directions based on eye movements.
        
        Args:
            movements: List of eye movement coordinates
            
        Returns:
            List of gaze direction strings
        """
        directions = []
        
        for x, y in movements:
            # Determine gaze direction based on eye position
            if abs(x) < 5 and abs(y) < 5:
                direction = "center"
            elif x > 10:
                direction = "right"
            elif x < -10:
                direction = "left"
            elif y > 10:
                direction = "down"
            elif y < -10:
                direction = "up"
            else:
                direction = "center"
            
            directions.append(direction)
        
        return directions
    
    def generate_synthetic_data(self, 
                              pattern_type: str = "random",
                              noise_level: float = 0.1) -> List[SyntheticEyeData]:
        """
        Generate complete synthetic eye tracking dataset.
        
        Args:
            pattern_type: Type of eye movement pattern
            noise_level: Level of noise to add (0.0 to 1.0)
            
        Returns:
            List of SyntheticEyeData objects
        """
        # Generate movement patterns
        movements = self.generate_eye_movement_pattern(pattern_type)
        blinks = self.generate_blink_pattern()
        gaze_directions = self.generate_gaze_directions(movements)
        
        synthetic_data = []
        
        for i in range(self.total_frames):
            timestamp = i / self.fps
            
            # Base eye positions
            base_x, base_y = movements[i]
            
            # Add noise
            noise_x = random.uniform(-noise_level, noise_level) * self.eye_radius
            noise_y = random.uniform(-noise_level, noise_level) * self.eye_radius
            
            # Calculate actual eye positions
            left_x = self.center_x + self.left_eye_offset[0] + base_x + noise_x
            left_y = self.center_y + self.left_eye_offset[1] + base_y + noise_y
            right_x = self.center_x + self.right_eye_offset[0] + base_x + noise_x
            right_y = self.center_y + self.right_eye_offset[1] + base_y + noise_y
            
            # Adjust confidence based on blink
            if blinks[i]:
                left_confidence = random.uniform(0.1, 0.3)
                right_confidence = random.uniform(0.1, 0.3)
            else:
                left_confidence = random.uniform(0.7, 0.95)
                right_confidence = random.uniform(0.7, 0.95)
            
            # Simulate FPS variation
            fps = random.uniform(self.fps * 0.9, self.fps * 1.1)
            
            data_point = SyntheticEyeData(
                timestamp=timestamp,
                left_eye_x=left_x,
                left_eye_y=left_y,
                right_eye_x=right_x,
                right_eye_y=right_y,
                left_confidence=left_confidence,
                right_confidence=right_confidence,
                gaze_direction=gaze_directions[i],
                blink_detected=blinks[i],
                fps=fps
            )
            
            synthetic_data.append(data_point)
        
        return synthetic_data
    
    def create_synthetic_video(self, 
                              data: List[SyntheticEyeData],
                              output_path: str = "data/synthetic/synthetic_eye_tracking.mp4") -> None:
        """
        Create a synthetic video showing eye tracking visualization.
        
        Args:
            data: Synthetic eye tracking data
            output_path: Path to save the video
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        for data_point in data:
            # Create frame
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            # Draw face outline
            face_center = (self.center_x, self.center_y)
            cv2.ellipse(frame, face_center, (100, 120), 0, 0, 360, (100, 100, 100), 2)
            
            # Draw eyes
            left_eye_pos = (int(data_point.left_eye_x), int(data_point.left_eye_y))
            right_eye_pos = (int(data_point.right_eye_x), int(data_point.right_eye_y))
            
            # Eye color based on confidence
            left_color = (0, int(255 * data_point.left_confidence), 0)
            right_color = (0, 0, int(255 * data_point.right_confidence))
            
            cv2.circle(frame, left_eye_pos, 8, left_color, -1)
            cv2.circle(frame, right_eye_pos, 8, right_color, -1)
            
            # Draw eye outlines
            cv2.circle(frame, left_eye_pos, 12, (255, 255, 255), 2)
            cv2.circle(frame, right_eye_pos, 12, (255, 255, 255), 2)
            
            # Add text information
            cv2.putText(frame, f"Gaze: {data_point.gaze_direction}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Blink: {'Yes' if data_point.blink_detected else 'No'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {data_point.fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Synthetic video saved to: {output_path}")
    
    def save_dataset(self, 
                    data: List[SyntheticEyeData],
                    output_dir: str = "data/synthetic") -> None:
        """
        Save synthetic dataset to files.
        
        Args:
            data: Synthetic eye tracking data
            output_dir: Directory to save the dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(d) for d in data])
        
        # Save CSV
        csv_path = output_path / "synthetic_eye_tracking.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV dataset saved to: {csv_path}")
        
        # Save JSON
        json_path = output_path / "synthetic_eye_tracking.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(d) for d in data], f, indent=2)
        print(f"JSON dataset saved to: {json_path}")
        
        # Save metadata
        metadata = {
            "total_frames": len(data),
            "duration": self.duration,
            "fps": self.fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "generated_at": time.time(),
            "description": "Synthetic eye tracking dataset for testing and demonstration"
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Generate synthetic datasets for testing."""
    print("Generating synthetic eye tracking datasets...")
    
    # Create generator
    generator = SyntheticDatasetGenerator(duration=60.0, fps=30.0)
    
    # Generate different patterns
    patterns = ["random", "circular", "horizontal", "vertical"]
    
    for pattern in patterns:
        print(f"\nGenerating {pattern} pattern...")
        
        # Generate data
        data = generator.generate_synthetic_data(pattern_type=pattern)
        
        # Save dataset
        output_dir = f"data/synthetic/{pattern}_pattern"
        generator.save_dataset(data, output_dir)
        
        # Create video
        video_path = f"{output_dir}/synthetic_eye_tracking.mp4"
        generator.create_synthetic_video(data, video_path)
    
    print("\nSynthetic dataset generation complete!")


if __name__ == "__main__":
    main()
