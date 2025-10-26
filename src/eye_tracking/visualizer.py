"""
Visualization and Explainability Module

This module provides visualization tools and explainability features
for the eye tracking system, helping users understand how the system works.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class EyeTrackingVisualizer:
    """Visualization tools for eye tracking data and results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def visualize_eye_landmarks(self, 
                              frame: np.ndarray, 
                              landmarks: List[Tuple[int, int]],
                              eye_points: Dict[str, List[int]]) -> np.ndarray:
        """
        Visualize facial landmarks with emphasis on eye regions.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmark coordinates
            eye_points: Dictionary with left and right eye point indices
            
        Returns:
            Frame with landmark visualization
        """
        vis_frame = frame.copy()
        
        # Draw all landmarks
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_frame, (x, y), 2, (255, 255, 255), -1)
            cv2.putText(vis_frame, str(i), (x+3, y-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Highlight eye regions
        colors = {'left_eye': (0, 255, 0), 'right_eye': (255, 0, 0)}
        
        for eye_name, points in eye_points.items():
            color = colors.get(eye_name, (255, 255, 0))
            
            # Draw eye contour
            eye_contour = np.array([landmarks[i] for i in points], np.int32)
            cv2.polylines(vis_frame, [eye_contour], True, color, 2)
            
            # Draw eye center
            eye_center = np.mean(eye_contour, axis=0).astype(int)
            cv2.circle(vis_frame, tuple(eye_center), 5, color, -1)
        
        return vis_frame
    
    def visualize_pupil_detection(self, 
                                 eye_roi: np.ndarray,
                                 threshold: np.ndarray,
                                 contours: List[np.ndarray],
                                 pupil_center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Visualize pupil detection process.
        
        Args:
            eye_roi: Eye region of interest
            threshold: Thresholded image
            contours: Detected contours
            pupil_center: Detected pupil center
            
        Returns:
            Visualization of pupil detection
        """
        # Create visualization
        vis_height = eye_roi.shape[0]
        vis_width = eye_roi.shape[1] * 3  # Three panels
        
        vis_image = np.zeros((vis_height, vis_width), dtype=np.uint8)
        
        # Panel 1: Original ROI
        vis_image[:, :eye_roi.shape[1]] = eye_roi
        
        # Panel 2: Thresholded image
        vis_image[:, eye_roi.shape[1]:2*eye_roi.shape[1]] = threshold
        
        # Panel 3: Contours
        contour_vis = np.zeros_like(eye_roi)
        if contours:
            cv2.drawContours(contour_vis, contours, -1, 255, 1)
            if pupil_center:
                cv2.circle(contour_vis, pupil_center, 3, 255, -1)
        vis_image[:, 2*eye_roi.shape[1]:] = contour_vis
        
        return vis_image
    
    def plot_eye_movement_trajectory(self, 
                                   data: pd.DataFrame,
                                   title: str = "Eye Movement Trajectory") -> go.Figure:
        """
        Plot eye movement trajectory over time.
        
        Args:
            data: DataFrame with eye tracking data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Left Eye X', 'Left Eye Y', 'Right Eye X', 'Right Eye Y'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Left eye X
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['left_eye_x'], 
                      name='Left Eye X', line=dict(color='green')),
            row=1, col=1
        )
        
        # Left eye Y
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['left_eye_y'], 
                      name='Left Eye Y', line=dict(color='lightgreen')),
            row=1, col=2
        )
        
        # Right eye X
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['right_eye_x'], 
                      name='Right Eye X', line=dict(color='red')),
            row=2, col=1
        )
        
        # Right eye Y
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['right_eye_y'], 
                      name='Right Eye Y', line=dict(color='pink')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_confidence_over_time(self, 
                                  data: pd.DataFrame,
                                  title: str = "Tracking Confidence Over Time") -> go.Figure:
        """
        Plot tracking confidence over time.
        
        Args:
            data: DataFrame with eye tracking data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['left_confidence'],
            mode='lines+markers',
            name='Left Eye Confidence',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['right_confidence'],
            mode='lines+markers',
            name='Right Eye Confidence',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (seconds)',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_performance_metrics(self, 
                                data: pd.DataFrame,
                                title: str = "Performance Metrics") -> go.Figure:
        """
        Plot performance metrics over time.
        
        Args:
            data: DataFrame with eye tracking data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('FPS', 'Faces Detected', 'Average Confidence', 'Data Quality'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # FPS
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['fps'], 
                      name='FPS', line=dict(color='purple')),
            row=1, col=1
        )
        
        # Faces detected
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['faces_detected'], 
                      name='Faces', line=dict(color='brown')),
            row=1, col=2
        )
        
        # Average confidence
        avg_confidence = (data['left_confidence'] + data['right_confidence']) / 2
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=avg_confidence, 
                      name='Avg Confidence', line=dict(color='green')),
            row=2, col=1
        )
        
        # Data quality (based on missing values)
        quality = ((data['left_eye_x'].notna()) & (data['right_eye_x'].notna())).astype(int)
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=quality, 
                      name='Data Quality', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_heatmap(self, 
                      data: pd.DataFrame,
                      title: str = "Eye Position Heatmap") -> go.Figure:
        """
        Create a heatmap of eye positions.
        
        Args:
            data: DataFrame with eye tracking data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Filter valid data
        valid_data = data.dropna(subset=['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y'])
        
        if valid_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No valid data for heatmap", 
                              xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create heatmap data
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Left Eye', 'Right Eye')
        )
        
        # Left eye heatmap
        fig.add_trace(
            go.Histogram2d(
                x=valid_data['left_eye_x'],
                y=valid_data['left_eye_y'],
                colorscale='Viridis',
                name='Left Eye'
            ),
            row=1, col=1
        )
        
        # Right eye heatmap
        fig.add_trace(
            go.Histogram2d(
                x=valid_data['right_eye_x'],
                y=valid_data['right_eye_y'],
                colorscale='Plasma',
                name='Right Eye'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=400
        )
        
        return fig
    
    def generate_report(self, 
                        data: pd.DataFrame,
                        output_path: str = "eye_tracking_report.html") -> None:
        """
        Generate a comprehensive HTML report.
        
        Args:
            data: DataFrame with eye tracking data
            output_path: Path to save the report
        """
        # Create plots
        trajectory_fig = self.plot_eye_movement_trajectory(data)
        confidence_fig = self.plot_confidence_over_time(data)
        performance_fig = self.plot_performance_metrics(data)
        heatmap_fig = self.create_heatmap(data)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Eye Tracking Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Eye Tracking Analysis Report</h1>
            
            <h2>Summary Statistics</h2>
            <ul>
                <li>Total data points: {len(data)}</li>
                <li>Average FPS: {data['fps'].mean():.1f}</li>
                <li>Average confidence: {(data['left_confidence'].mean() + data['right_confidence'].mean()) / 2:.2f}</li>
                <li>Data quality: {(data['left_eye_x'].notna().sum() / len(data) * 100):.1f}%</li>
            </ul>
            
            <h2>Eye Movement Trajectory</h2>
            <div id="trajectory"></div>
            
            <h2>Tracking Confidence</h2>
            <div id="confidence"></div>
            
            <h2>Performance Metrics</h2>
            <div id="performance"></div>
            
            <h2>Eye Position Heatmap</h2>
            <div id="heatmap"></div>
            
            <script>
                Plotly.newPlot('trajectory', {trajectory_fig.to_json()});
                Plotly.newPlot('confidence', {confidence_fig.to_json()});
                Plotly.newPlot('performance', {performance_fig.to_json()});
                Plotly.newPlot('heatmap', {heatmap_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to: {output_path}")


def main():
    """Demo the visualization capabilities."""
    # Create sample data
    np.random.seed(42)
    n_points = 100
    
    data = pd.DataFrame({
        'timestamp': np.linspace(0, 10, n_points),
        'left_eye_x': 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 2, n_points),
        'left_eye_y': 200 + 15 * np.cos(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 2, n_points),
        'right_eye_x': 150 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 2, n_points),
        'right_eye_y': 200 + 15 * np.cos(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 2, n_points),
        'left_confidence': np.random.uniform(0.7, 0.95, n_points),
        'right_confidence': np.random.uniform(0.7, 0.95, n_points),
        'fps': np.random.uniform(25, 30, n_points),
        'faces_detected': np.ones(n_points)
    })
    
    # Create visualizer
    visualizer = EyeTrackingVisualizer()
    
    # Generate plots
    trajectory_fig = visualizer.plot_eye_movement_trajectory(data)
    confidence_fig = visualizer.plot_confidence_over_time(data)
    performance_fig = visualizer.plot_performance_metrics(data)
    heatmap_fig = visualizer.create_heatmap(data)
    
    # Show plots
    trajectory_fig.show()
    confidence_fig.show()
    performance_fig.show()
    heatmap_fig.show()
    
    # Generate report
    visualizer.generate_report(data, "eye_tracking_report.html")


if __name__ == "__main__":
    main()
