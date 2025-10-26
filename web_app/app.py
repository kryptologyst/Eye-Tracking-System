"""
Streamlit Web Interface for Eye Tracking

This module provides a web-based interface for the eye tracking system,
making it easy to demonstrate and interact with the functionality.
"""

import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from eye_tracking import EyeTracker, EyePosition


class EyeTrackingWebApp:
    """Web application for eye tracking demonstration."""
    
    def __init__(self):
        """Initialize the web app."""
        self.tracker = None
        self.tracking_data = []
        self.max_data_points = 100
    
    def setup_page(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Eye Tracking Demo",
            page_icon="👁️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("👁️ Modern Eye Tracking System")
        st.markdown("---")
    
    def initialize_tracker(self) -> bool:
        """Initialize the eye tracker."""
        try:
            if self.tracker is None:
                self.tracker = EyeTracker()
                return True
            return True
        except Exception as e:
            st.error(f"Failed to initialize eye tracker: {e}")
            return False
    
    def create_sidebar(self):
        """Create the sidebar with controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Tracking controls
        if st.sidebar.button("🎬 Start Tracking", key="start"):
            if self.initialize_tracker():
                st.session_state.tracking_active = True
                st.success("Tracking started!")
        
        if st.sidebar.button("⏹️ Stop Tracking", key="stop"):
            st.session_state.tracking_active = False
            if self.tracker:
                self.tracker.cleanup()
            st.info("Tracking stopped!")
        
        # Data controls
        if st.sidebar.button("🗑️ Clear Data", key="clear"):
            self.tracking_data = []
            st.success("Data cleared!")
        
        # Export controls
        if st.sidebar.button("📊 Export Data", key="export"):
            if self.tracking_data:
                df = pd.DataFrame(self.tracking_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"eye_tracking_data_{int(time.time())}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export!")
        
        # Settings
        st.sidebar.header("⚙️ Settings")
        self.max_data_points = st.sidebar.slider(
            "Max Data Points", 
            min_value=10, 
            max_value=500, 
            value=100
        )
    
    def display_camera_feed(self):
        """Display the camera feed with tracking overlay."""
        st.header("📹 Live Camera Feed")
        
        if not hasattr(st.session_state, 'tracking_active'):
            st.session_state.tracking_active = False
        
        if st.session_state.tracking_active and self.tracker:
            # Create placeholder for camera feed
            camera_placeholder = st.empty()
            
            # Initialize camera if not already done
            if not self.tracker.cap or not self.tracker.cap.isOpened():
                if not self.tracker.start_camera():
                    st.error("Failed to start camera!")
                    st.session_state.tracking_active = False
                    return
            
            # Process frames
            try:
                ret, frame = self.tracker.cap.read()
                if ret:
                    # Process frame
                    results = self.tracker.process_frame(frame)
                    
                    # Draw results
                    frame = self.tracker.draw_results(frame, results)
                    
                    # Store tracking data
                    self._store_tracking_data(results)
                    
                    # Display frame
                    camera_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # Display current tracking info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("FPS", f"{results['fps']:.1f}")
                    
                    with col2:
                        st.metric("Faces Detected", results['faces_detected'])
                    
                    with col3:
                        confidence = 0.0
                        if results['left_eye'] and results['right_eye']:
                            confidence = (results['left_eye'].confidence + results['right_eye'].confidence) / 2
                        st.metric("Tracking Confidence", f"{confidence:.2f}")
                
            except Exception as e:
                st.error(f"Error processing camera feed: {e}")
                st.session_state.tracking_active = False
        else:
            st.info("Click 'Start Tracking' to begin eye tracking!")
    
    def _store_tracking_data(self, results: Dict[str, Any]):
        """Store tracking data for analysis."""
        data_point = {
            'timestamp': results['timestamp'],
            'fps': results['fps'],
            'faces_detected': results['faces_detected'],
            'left_eye_x': results['left_eye'].center[0] if results['left_eye'] else None,
            'left_eye_y': results['left_eye'].center[1] if results['left_eye'] else None,
            'right_eye_x': results['right_eye'].center[0] if results['right_eye'] else None,
            'right_eye_y': results['right_eye'].center[1] if results['right_eye'] else None,
            'left_confidence': results['left_eye'].confidence if results['left_eye'] else 0.0,
            'right_confidence': results['right_eye'].confidence if results['right_eye'] else 0.0,
        }
        
        self.tracking_data.append(data_point)
        
        # Keep only recent data points
        if len(self.tracking_data) > self.max_data_points:
            self.tracking_data = self.tracking_data[-self.max_data_points:]
    
    def display_analytics(self):
        """Display tracking analytics and visualizations."""
        st.header("📊 Tracking Analytics")
        
        if not self.tracking_data:
            st.info("No tracking data available. Start tracking to see analytics!")
            return
        
        df = pd.DataFrame(self.tracking_data)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Eye Movement", "📊 Performance", "🎯 Confidence", "📋 Raw Data"])
        
        with tab1:
            self._plot_eye_movement(df)
        
        with tab2:
            self._plot_performance(df)
        
        with tab3:
            self._plot_confidence(df)
        
        with tab4:
            self._display_raw_data(df)
    
    def _plot_eye_movement(self, df: pd.DataFrame):
        """Plot eye movement over time."""
        if df.empty:
            st.info("No data to plot")
            return
        
        fig = go.Figure()
        
        # Plot left eye
        if 'left_eye_x' in df.columns and df['left_eye_x'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['left_eye_x'],
                mode='lines+markers',
                name='Left Eye X',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['left_eye_y'],
                mode='lines+markers',
                name='Left Eye Y',
                line=dict(color='lightgreen')
            ))
        
        # Plot right eye
        if 'right_eye_x' in df.columns and df['right_eye_x'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['right_eye_x'],
                mode='lines+markers',
                name='Right Eye X',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['right_eye_y'],
                mode='lines+markers',
                name='Right Eye Y',
                line=dict(color='pink')
            ))
        
        fig.update_layout(
            title="Eye Movement Over Time",
            xaxis_title="Time",
            yaxis_title="Pixel Position",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_performance(self, df: pd.DataFrame):
        """Plot performance metrics."""
        if df.empty:
            st.info("No data to plot")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # FPS over time
            fig_fps = px.line(df, x='timestamp', y='fps', title='FPS Over Time')
            st.plotly_chart(fig_fps, use_container_width=True)
        
        with col2:
            # Faces detected
            fig_faces = px.line(df, x='timestamp', y='faces_detected', title='Faces Detected')
            st.plotly_chart(fig_faces, use_container_width=True)
    
    def _plot_confidence(self, df: pd.DataFrame):
        """Plot tracking confidence."""
        if df.empty:
            st.info("No data to plot")
            return
        
        fig = go.Figure()
        
        if 'left_confidence' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['left_confidence'],
                mode='lines+markers',
                name='Left Eye Confidence',
                line=dict(color='blue')
            ))
        
        if 'right_confidence' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['right_confidence'],
                mode='lines+markers',
                name='Right Eye Confidence',
                line=dict(color='orange')
            ))
        
        fig.update_layout(
            title="Tracking Confidence Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_raw_data(self, df: pd.DataFrame):
        """Display raw tracking data."""
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Data Points", len(df))
            
            with col2:
                avg_fps = df['fps'].mean() if 'fps' in df.columns else 0
                st.metric("Average FPS", f"{avg_fps:.1f}")
            
            with col3:
                avg_faces = df['faces_detected'].mean() if 'faces_detected' in df.columns else 0
                st.metric("Average Faces", f"{avg_faces:.1f}")
            
            with col4:
                avg_confidence = 0
                if 'left_confidence' in df.columns and 'right_confidence' in df.columns:
                    avg_confidence = (df['left_confidence'].mean() + df['right_confidence'].mean()) / 2
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
    
    def display_info(self):
        """Display information about the eye tracking system."""
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### Features
        
        - **Real-time Eye Tracking**: Detects and tracks both eyes in real-time
        - **Facial Landmark Detection**: Uses dlib's 68-point facial landmark model
        - **Pupil Detection**: Advanced contour detection for accurate pupil localization
        - **Performance Monitoring**: Real-time FPS and confidence metrics
        - **Data Export**: Export tracking data for further analysis
        - **Interactive Visualizations**: Live charts and analytics
        
        ### Technical Details
        
        - **Computer Vision**: OpenCV for image processing
        - **Facial Detection**: dlib's frontal face detector
        - **Landmark Prediction**: 68-point facial landmark model
        - **Web Interface**: Streamlit for easy interaction
        - **Data Visualization**: Plotly for interactive charts
        
        ### Usage Instructions
        
        1. Click "Start Tracking" to begin real-time eye tracking
        2. Position your face in front of the camera
        3. The system will detect your eyes and track their movement
        4. View analytics and export data as needed
        5. Click "Stop Tracking" when finished
        """)
    
    def run(self):
        """Run the web application."""
        self.setup_page()
        self.create_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.display_camera_feed()
        
        with col2:
            self.display_info()
        
        # Analytics section
        self.display_analytics()


def main():
    """Main function to run the web app."""
    app = EyeTrackingWebApp()
    app.run()


if __name__ == "__main__":
    main()
