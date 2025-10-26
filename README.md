# Eye Tracking System

A comprehensive, modern eye tracking implementation using computer vision techniques with dlib for facial landmark detection and OpenCV for image processing. This project provides real-time eye tracking capabilities with enhanced accuracy, web interface, synthetic data generation, and extensive testing.

## Features

- **Real-time Eye Tracking**: Detects and tracks both eyes in real-time using webcam
- **Facial Landmark Detection**: Uses dlib's 68-point facial landmark model
- **Advanced Pupil Detection**: Sophisticated contour detection for accurate pupil localization
- **Web Interface**: Streamlit-based web application for easy interaction and demonstration
- **Synthetic Data Generation**: Create realistic synthetic datasets for testing and development
- **Comprehensive Testing**: Full test suite with unit tests and integration tests
- **Configuration Management**: YAML-based configuration system
- **Data Export**: Export tracking data in CSV and JSON formats
- **Performance Monitoring**: Real-time FPS and confidence metrics
- **Interactive Visualizations**: Live charts and analytics using Plotly

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- dlib facial landmark predictor model

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Eye-Tracking-System.git
   cd Eye-Tracking-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the facial landmark predictor**:
   ```bash
   # Create models directory
   mkdir -p data/models
   
   # Download the predictor (you'll need to get this from dlib's model zoo)
   # Place it in data/models/shape_predictor_68_face_landmarks.dat
   ```

### Basic Usage

#### Command Line Interface

```bash
# Run real-time eye tracking
python cli.py --mode realtime

# Generate synthetic data
python cli.py --mode synthetic --pattern circular --duration 30

# Run web interface
python cli.py --mode web

# Run tests
python cli.py --mode test
```

#### Python API

```python
from src.eye_tracking import EyeTracker

# Initialize tracker
tracker = EyeTracker("config/config.yaml")

# Run real-time tracking
tracker.run_realtime()
```

#### Web Interface

```bash
# Start the web interface
python cli.py --mode web

# Or directly with Streamlit
cd web_app
streamlit run app.py
```

## 📁 Project Structure

```
0239_Eye_tracking_implementation/
├── src/
│   └── eye_tracking/
│       ├── __init__.py
│       └── tracker.py          # Main eye tracking implementation
├── web_app/
│   └── app.py                  # Streamlit web interface
├── data/
│   ├── models/                 # Model files (dlib predictor)
│   └── synthetic/              # Synthetic datasets
│       └── generator.py         # Synthetic data generator
├── tests/
│   └── test_eye_tracking.py    # Unit tests
├── config/
│   └── config.yaml             # Configuration file
├── cli.py                      # Command line interface
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Configuration

The system uses YAML configuration files for easy customization. Key configuration options:

```yaml
model:
  predictor_path: "data/models/shape_predictor_68_face_landmarks.dat"
  confidence_threshold: 0.5

camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30

eye_tracking:
  left_eye_points: [36, 37, 38, 39, 40, 41]
  right_eye_points: [42, 43, 44, 45, 46, 47]
  pupil_threshold: 70
  min_contour_area: 50

visualization:
  pupil_radius: 5
  text_scale: 0.5
  text_thickness: 1
  colors:
    left_eye: [0, 255, 0]
    right_eye: [255, 0, 0]
```

## Usage Examples

### Real-time Eye Tracking

```python
from src.eye_tracking import EyeTracker

# Initialize with custom config
tracker = EyeTracker("config/config.yaml")

# Start real-time tracking
tracker.run_realtime()
```

### Synthetic Data Generation

```python
from data.synthetic.generator import SyntheticDatasetGenerator

# Create generator
generator = SyntheticDatasetGenerator(duration=60.0, fps=30.0)

# Generate circular movement pattern
data = generator.generate_synthetic_data(pattern_type="circular")

# Save dataset and create video
generator.save_dataset(data, "output/synthetic_data")
generator.create_synthetic_video(data, "output/synthetic_video.mp4")
```

### Custom Eye Tracking Processing

```python
import cv2
from src.eye_tracking import EyeTracker

tracker = EyeTracker()

# Process a single frame
frame = cv2.imread("image.jpg")
results = tracker.process_frame(frame)

# Access tracking results
if results["left_eye"]:
    left_center = results["left_eye"].center
    left_confidence = results["left_eye"].confidence
    print(f"Left eye at {left_center} with confidence {left_confidence}")

# Draw results on frame
annotated_frame = tracker.draw_results(frame, results)
cv2.imshow("Eye Tracking", annotated_frame)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python cli.py --mode test

# Or directly with pytest
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Synthetic Data

The system includes a sophisticated synthetic data generator that creates realistic eye tracking datasets:

### Available Patterns

- **Random**: Natural random eye movements
- **Circular**: Smooth circular eye movements
- **Horizontal**: Left-right eye movements
- **Vertical**: Up-down eye movements

### Generated Data

Each synthetic dataset includes:
- Eye position coordinates (x, y)
- Tracking confidence scores
- Blink detection events
- Gaze direction estimates
- Performance metrics (FPS)
- Timestamps

## Web Interface

The Streamlit web interface provides:

- **Live Camera Feed**: Real-time eye tracking visualization
- **Interactive Controls**: Start/stop tracking, data export
- **Analytics Dashboard**: Eye movement charts and performance metrics
- **Data Export**: Download tracking data as CSV
- **Settings Panel**: Configure tracking parameters

## 🔧 Advanced Features

### Gaze Direction Estimation

The system includes basic gaze direction estimation based on eye positions:

```python
# Calculate gaze direction
gaze_info = tracker._calculate_gaze_direction(left_center, right_center)
print(f"Gaze direction: {gaze_info['direction']}")
print(f"Confidence: {gaze_info['confidence']}")
```

### Performance Monitoring

Real-time performance metrics:

```python
results = tracker.process_frame(frame)
print(f"FPS: {results['fps']:.1f}")
print(f"Faces detected: {results['faces_detected']}")
```

### Data Export

Export tracking data for analysis:

```python
import pandas as pd

# Convert tracking results to DataFrame
df = pd.DataFrame(tracking_data)
df.to_csv("eye_tracking_data.csv", index=False)
```

## Troubleshooting

### Common Issues

1. **Camera not found**:
   - Check camera permissions
   - Verify camera device ID in config
   - Try different camera devices (0, 1, 2, etc.)

2. **Predictor file not found**:
   - Download `shape_predictor_68_face_landmarks.dat` from dlib's model zoo
   - Place it in `data/models/` directory
   - Update path in config file

3. **Poor tracking accuracy**:
   - Ensure good lighting conditions
   - Position face clearly in camera view
   - Adjust `pupil_threshold` in config
   - Check `min_contour_area` setting

4. **Low FPS**:
   - Reduce camera resolution in config
   - Close other applications
   - Use hardware acceleration if available

### Performance Optimization

- **Reduce resolution**: Lower camera width/height in config
- **Adjust FPS**: Set appropriate FPS for your hardware
- **Optimize thresholds**: Tune `pupil_threshold` and `min_contour_area`
- **Use GPU**: Consider GPU-accelerated OpenCV builds

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **dlib**: For facial landmark detection
- **OpenCV**: For computer vision capabilities
- **Streamlit**: For web interface framework
- **Plotly**: For interactive visualizations

## References

- [dlib Documentation](http://dlib.net/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Eye Tracking Research Papers](https://scholar.google.com/scholar?q=eye+tracking+computer+vision)

## Future Enhancements

- [ ] Machine learning-based gaze estimation
- [ ] Multi-person eye tracking
- [ ] Calibration system for improved accuracy
- [ ] Integration with accessibility tools
- [ ] Mobile app development
- [ ] Cloud-based processing
- [ ] Real-time analytics dashboard
- [ ] Integration with VR/AR systems


**Note**: This project requires the dlib facial landmark predictor model. Download it from dlib's model zoo and place it in the `data/models/` directory.
# Eye-Tracking-System
