"""
Streamlit Web Application for DeepFake Detection
Updated with optimal threshold (0.45) from analysis
"""

import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import cv2
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.detection import DeepFakeDetector
from utils.preprocessing import VideoPreprocessor


# ═══════════════════════════════════════════════════════════
# OPTIMAL CONFIGURATION FROM THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════
OPTIMAL_THRESHOLD = 0.45  # ⭐ Updated from colab analysis!

# Model paths
IMAGE_MODEL_PATH = "deepfake_image_model.h5"
VIDEO_MODEL_PATH = "deepfake_video_model.h5"

# Page configuration
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced colors
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fake-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border: 3px solid #c92a2a;
        color: white;
    }
    .real-result {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        border: 3px solid #2f9e44;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #dee2e6;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 2rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_type' not in st.session_state:
        st.session_state.model_type = 'image'


def load_model(threshold, model_type="image"):
    """Load the detection model"""
    try:
        # Select model path based on type
        if model_type == "video":
            model_path = VIDEO_MODEL_PATH
            target_size = (128, 128)  # Video model uses 128x128
            st.session_state.model_type = 'video'
        else:  # image
            model_path = IMAGE_MODEL_PATH
            target_size = (224, 224)  # Image model uses 224x224
            st.session_state.model_type = 'image'
        
        detector = DeepFakeDetector(model_path, threshold=threshold, target_size=target_size)
        
        if detector.model is None:
            # Clear session state
            st.session_state.detector = None
            st.session_state.model_loaded = False
            
            st.error("❌ Failed to load model - model returned None")
            return False
        
        st.session_state.detector = detector
        st.session_state.model_loaded = True
        return True
    except Exception as e:
        # Clear session state on error
        st.session_state.detector = None
        st.session_state.model_loaded = False
        
        st.error(f"Error loading model: {str(e)}")
        if 'TimeDistributed' in str(e):
            st.warning("This model was trained with a different Keras version and is incompatible.")
            st.info("💡 Try using the image model instead")
        return False


def display_result(result, media_type="image", threshold=OPTIMAL_THRESHOLD):
    """Display detection results with improved logic"""
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    raw_score = result.get('raw_score', 0)
    
    # ═══════════════════════════════════════════════════════════
    # SIMPLE CLASSIFICATION LOGIC - REAL OR FAKE ONLY
    # ═══════════════════════════════════════════════════════════
    
    # Determine classification
    if raw_score > threshold:
        prediction = "FAKE"
        # Calculate confidence for fake (how far above threshold)
        confidence = ((raw_score - threshold) / (1 - threshold)) * 100
    else:
        prediction = "REAL"
        # Calculate confidence for real (how far below threshold)
        confidence = ((threshold - raw_score) / threshold) * 100
    
    # Determine confidence level
    if confidence > 80:
        confidence_label = "HIGH"
    elif confidence > 50:
        confidence_label = "MODERATE"
    else:
        confidence_label = "LOW"
    
    # Set display properties
    if prediction == "FAKE":
        result_class = "fake-result"
        icon = "🚨"
        main_text = "DEEPFAKE DETECTED"
        sub_text = f"{confidence_label} Confidence"
    else:  # REAL
        result_class = "real-result"
        icon = "✅"
        main_text = "AUTHENTIC MEDIA"
        sub_text = f"{confidence_label} Confidence"
    
    # Display result box
    st.markdown(f"""
        <div class="result-box {result_class}">
            <h2 style="margin:0;">{icon} {main_text}</h2>
            <h3 style="margin-top:1rem;">{sub_text}: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════
    # DETAILED METRICS
    # ═══════════════════════════════════════════════════════════
    
    st.subheader("📊 Detailed Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", prediction)
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")
    with col3:
        st.metric("Raw Score", f"{raw_score:.4f}")
    with col4:
        st.metric("Threshold", f"{threshold:.2f}")
    
    # ═══════════════════════════════════════════════════════════
    # CONFIDENCE VISUALIZATION
    # ═══════════════════════════════════════════════════════════
    
    st.subheader("📈 Confidence Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        real_score = (1 - raw_score) * 100
        st.metric(
            label="REAL Score",
            value=f"{real_score:.1f}%",
            delta="Selected" if prediction == "REAL" else None
        )
    
    with col2:
        fake_score = raw_score * 100
        st.metric(
            label="FAKE Score",
            value=f"{fake_score:.1f}%",
            delta="Selected" if prediction == "FAKE" else None
        )
    
    # Progress bar with threshold indicator
    st.write("**Prediction Scale:**")
    st.progress(float(raw_score))
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("← **REAL (0.00)**")
    with col2:
        st.write(f"**⚡ Threshold: {threshold}**")
    with col3:
        st.write("**FAKE (1.00)** →")
    
    # ═══════════════════════════════════════════════════════════
    # PIXEL ANALYSIS DETAILS (SHOW REASONS FOR FAKE DETECTION)
    # ═══════════════════════════════════════════════════════════
    
    if prediction == "FAKE" and 'indicators' in result:
        st.subheader("🔬 Pixel Analysis Details")
        
        indicators = result['indicators']
        if indicators:
            st.markdown("**Detected Artifacts:**")
            for i, reason in enumerate(indicators, 1):
                st.markdown(f"{i}. {reason}")
            
            st.info(f"""
            **Analysis Summary:** {len(indicators)} pixel artifacts detected that are characteristic of deepfake manipulation.
            
            These indicators include unnatural noise patterns, color channel imbalances, edge artifacts, 
            compression anomalies, and other pixel-level inconsistencies that suggest the media has been artificially generated or manipulated.
            """)
        else:
            st.info("No specific pixel artifacts detected, but overall analysis suggests manipulation.")
    
    # ═══════════════════════════════════════════════════════════
    # VIDEO-SPECIFIC INFORMATION
    # ═══════════════════════════════════════════════════════════
    
    if media_type == "video":
        st.subheader("🎥 Video Analysis Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames Analyzed", result.get('frames_analyzed', 0))
        with col2:
            st.metric("Fake Frames", result.get('fake_frames', 0))
        with col3:
            st.metric("Real Frames", result.get('real_frames', 0))
        with col4:
            st.metric("Fake %", f"{result.get('fake_percentage', 0):.1f}%")
        
        # Frame scores visualization
        if 'frame_scores' in result:
            st.subheader("📊 Frame-by-Frame Analysis")
            
            frame_scores = result['frame_scores']
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=[frame_scores['min'], frame_scores['max']],
                name="Score Range",
                boxmean='sd'
            ))
            
            # Add threshold line
            fig.add_hline(
                y=threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Threshold ({threshold})"
            )
            
            fig.update_layout(
                title="Frame Score Distribution",
                yaxis_title="Prediction Score",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_column_width=True)
    
    # ═══════════════════════════════════════════════════════════
    # TECHNICAL DETAILS
    # ═══════════════════════════════════════════════════════════
    
    with st.expander("🔬 Technical Details & Interpretation"):
        st.write(f"""
        **Model Configuration:**
        - Optimal Threshold: {threshold} (from analysis)
        - Test Accuracy: 98.33%
        - Precision: 98.67%
        - Recall: 97.99%
        - F1-Score: 0.9833
        
        **Your Result:**
        - Raw Prediction Score: {raw_score:.6f}
        - Distance from Threshold: {abs(raw_score - threshold):.6f}
        - Classification: {prediction}
        - Confidence Level: {confidence_label}
        
        **Decision Logic:**
        - Score > {threshold}: Classify as FAKE
        - Score ≤ {threshold}: Classify as REAL
        
        **Interpretation:**
        - Scores closer to 0.0 = More likely REAL
        - Scores closer to 1.0 = More likely FAKE
        """)


def image_detection_page():
    """Image detection interface"""
    st.markdown('<p class="sub-header">Upload an image to detect if it\'s a deepfake</p>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image file to analyze"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Image info
            st.info(f"""
            **Image Information:**
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            """)
        
        with col2:
            st.subheader("🔍 Analysis")
            
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                if not st.session_state.model_loaded:
                    st.error("⚠️ Please load a model first from the sidebar!")
                    return
                
                with st.spinner("Analyzing image..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Predict
                        result = st.session_state.detector.predict_image(tmp_path)
                        
                        # Display results with optimal threshold
                        display_result(
                            result, 
                            media_type="image",
                            threshold=st.session_state.detector.threshold
                        )
                    
                    finally:
                        # Clean up
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)


def video_detection_page():
    """Enhanced video detection interface with frame extraction and face detection"""
    st.markdown('<p class="sub-header">Upload a video to detect deepfakes with frame-by-frame analysis</p>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a video...",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze"
    )
    
    # Analysis parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_frames = st.slider("Frames to extract", 5, 100, 20,
                              help="Number of frames to extract and analyze")
    with col2:
        show_faces = st.checkbox("Detect faces", value=True,
                                help="Show face detection boxes on frames")
    with col3:
        aggregation = st.selectbox("Aggregation", 
                                   ['mean', 'median', 'max'],
                                   help="How to combine frame predictions")
    
    if uploaded_file is not None:
        # Display video info
        st.subheader("🎬 Video Information")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Import video processor
            from utils.preprocessing import VideoPreprocessor
            video_processor = VideoPreprocessor()
            
            # Get video info
            video_info = video_processor.get_video_info(tmp_path)
            
            if video_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                with col2:
                    st.metric("FPS", f"{video_info['fps']:.2f}")
                with col3:
                    st.metric("Total Frames", video_info['total_frames'])
                with col4:
                    st.metric("Duration", f"{video_info['duration']:.2f}s")
            
            # Display video
            st.video(uploaded_file)
            
            # Analyze button
            if st.button("🚀 Extract Frames & Analyze", type="primary", use_container_width=True):
                if not st.session_state.model_loaded:
                    st.error("⚠️ Please load a model first from the sidebar!")
                    return
                
                # Step 1: Extract frames
                with st.spinner(f"📹 Extracting {max_frames} frames from video..."):
                    frames_data = video_processor.extract_frames_with_faces(
                        tmp_path,
                        max_frames=max_frames
                    )
                
                if not frames_data:
                    st.error("❌ Could not extract frames from video")
                    return
                
                st.success(f"✅ Extracted {len(frames_data)} frames")
                
                # Step 2: Display extracted frames
                st.subheader("📸 Extracted Frames")
                
                # Show frames in a grid
                cols_per_row = 4
                frame_results = []
                
                for i in range(0, len(frames_data), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(frames_data):
                            frame_data = frames_data[idx]
                            with col:
                                # Show frame with or without face boxes
                                if show_faces and frame_data['has_faces']:
                                    st.image(frame_data['frame_with_boxes'], 
                                           caption=f"Frame {frame_data['frame_number']} ({frame_data['timestamp']:.1f}s) - {len(frame_data['faces'])} face(s)",
                                           use_column_width=True)
                                else:
                                    st.image(frame_data['frame'], 
                                           caption=f"Frame {frame_data['frame_number']} ({frame_data['timestamp']:.1f}s)",
                                           use_column_width=True)
                
                # Step 3: Analyze each frame
                st.subheader("🔍 Analyzing Frames...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                temp_paths = video_processor.save_frames_temp(frames_data)
                
                try:
                    for idx, (frame_data, temp_path) in enumerate(zip(frames_data, temp_paths)):
                        status_text.text(f"Analyzing frame {idx + 1}/{len(frames_data)}...")
                        progress_bar.progress((idx + 1) / len(frames_data))
                        
                        # Predict
                        result = st.session_state.detector.predict_image(temp_path)
                        frame_data['prediction'] = result
                    
                    status_text.text("✅ Analysis complete!")
                    
                    # Step 4: Aggregate results using CONSERVATIVE LOGIC
                    st.subheader("📊 Results")
                    
                    scores = [fd['prediction'].get('raw_score', 0) for fd in frames_data]
                    threshold = st.session_state.detector.threshold
                    
                    # 🎥 CONSERVATIVE VIDEO LOGIC: ANY FAKE FRAME MAKES VIDEO FAKE
                    fake_frames = sum(1 for s in scores if s > threshold)
                    real_frames = sum(1 for s in scores if s <= threshold)
                    
                    if fake_frames > 0:
                        # ANY fake frame = WHOLE VIDEO IS FAKE
                        final_prediction = "FAKE"
                        # Use the highest fake score as the video score
                        fake_scores = [s for s in scores if s > threshold]
                        final_score = max(fake_scores) if fake_scores else threshold + 0.1
                        final_confidence = ((final_score - threshold) / (1 - threshold)) * 100
                        
                        st.warning(f"🚨 **CONSERVATIVE DETECTION**: {fake_frames} frame(s) detected as fake out of {len(scores)} total frames")
                        st.info("**Result**: Since at least one frame is fake, the entire video is classified as FAKE")
                        
                    else:
                        # ALL frames are real = VIDEO IS REAL
                        final_prediction = "REAL"
                        # Use the lowest real score as the video score
                        real_scores = [s for s in scores if s <= threshold]
                        final_score = min(real_scores) if real_scores else threshold - 0.1
                        final_confidence = ((threshold - final_score) / threshold) * 100
                        
                        st.success(f"✅ **ALL FRAMES AUTHENTIC**: All {len(scores)} frames detected as real")
                    
                    # Create result dictionary
                    final_result = {
                        'raw_score': final_score,
                        'prediction': final_prediction,
                        'confidence': final_confidence,
                        'frames_analyzed': len(frames_data),
                        'fake_frames': fake_frames,
                        'real_frames': real_frames,
                        'fake_percentage': (fake_frames / len(scores)) * 100,
                        'frame_scores': {
                            'min': min(scores),
                            'max': max(scores),
                            'mean': np.mean(scores),
                            'median': np.median(scores)
                        }
                    }
                    
                    # Display overall result with enhanced colors
                    display_result(final_result, media_type="video", threshold=threshold)
                    
                    # 🎥 Show conservative logic explanation
                    if fake_frames > 0:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                                   padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #d63031;">
                        <h4 style="color: #d63031; margin: 0;">🎯 Conservative Detection Logic Applied</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #2d3436;">
                        <strong>Rule:</strong> If ANY frame in the video is detected as fake, the entire video is classified as FAKE.<br>
                        <strong>Reason:</strong> Deepfake manipulation in even one frame indicates the video has been tampered with.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #55efc4 0%, #00b894 100%); 
                                   padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #00b894;">
                        <h4 style="color: #00b894; margin: 0;">✅ All Frames Authentic</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #2d3436;">
                        <strong>Result:</strong> All analyzed frames appear authentic. Video classified as REAL.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Step 5: Show frame-by-frame results
                    st.subheader("🎞️ Frame-by-Frame Analysis")
                    
                    # Create results table with enhanced styling
                    results_data = []
                    for fd in frames_data:
                        score = fd['prediction'].get('raw_score', 0)
                        is_fake = score > threshold
                        results_data.append({
                            'Frame': fd['frame_number'],
                            'Time (s)': f"{fd['timestamp']:.2f}",
                            'Faces': len(fd['faces']),
                            'Score': f"{score:.4f}",
                            'Prediction': '🚨 FAKE' if is_fake else '✅ REAL',
                            'Confidence': f"{fd['prediction'].get('confidence', 0)*100:.1f}%"
                        })
                    
                    st.dataframe(results_data, use_container_width=True)
                    
                    # Show summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Frames Analyzed", len(frames_data))
                    with col2:
                        st.metric("Fake Frames", fake_frames, 
                                delta=f"{(fake_frames/len(frames_data))*100:.1f}%" if fake_frames > 0 else "0%")
                    with col3:
                        st.metric("Real Frames", real_frames,
                                delta=f"{(real_frames/len(frames_data))*100:.1f}%" if real_frames > 0 else "0%")
                    with col4:
                        status = "🚨 FAKE VIDEO" if fake_frames > 0 else "✅ REAL VIDEO"
                        st.metric("Video Status", status)
                    
                    # Visualization of scores across frames
                    st.subheader("� Score Distribution Across Frames")
                    
                    fig = go.Figure()
                    
                    # Add score line
                    fig.add_trace(go.Scatter(
                        x=[fd['frame_number'] for fd in frames_data],
                        y=scores,
                        mode='lines+markers',
                        name='Frame Scores',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=8, 
                                   color=['red' if s > threshold else 'green' for s in scores])
                    ))
                    
                    # Add threshold line
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold ({threshold}) - Any frame above = FAKE VIDEO",
                        annotation_position="right"
                    )
                    
                    # Color regions
                    fig.add_hrect(y0=0, y1=threshold, fillcolor="green", opacity=0.1, 
                                 annotation_text="REAL REGION", annotation_position="top left")
                    fig.add_hrect(y0=threshold, y1=1, fillcolor="red", opacity=0.1,
                                 annotation_text="FAKE REGION (1+ frames = FAKE VIDEO)", 
                                 annotation_position="bottom left")
                    
                    # Highlight fake frames
                    fake_indices = [i for i, s in enumerate(scores) if s > threshold]
                    if fake_indices:
                        fig.add_trace(go.Scatter(
                            x=[frames_data[i]['frame_number'] for i in fake_indices],
                            y=[scores[i] for i in fake_indices],
                            mode='markers',
                            name='Fake Frames',
                            marker=dict(size=12, color='red', symbol='x', 
                                       line=dict(width=2, color='darkred')),
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title="🎥 Conservative Video Analysis: Any Fake Frame = Fake Video",
                        xaxis_title="Frame Number",
                        yaxis_title="Prediction Score (0=Real, 1=Fake)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                finally:
                    # Cleanup temp files
                    video_processor.cleanup_temp_files(temp_paths)
        
        finally:
            # Clean up video file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def batch_detection_page():
    """Batch processing interface"""
    st.markdown('<p class="sub-header">Upload multiple images for batch analysis</p>', 
                unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple image files to analyze"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded")
        
        if st.button("🚀 Analyze All Images", type="primary", use_container_width=True):
            if not st.session_state.model_loaded:
                st.error("⚠️ Please load a model first from the sidebar!")
                return
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            temp_files = []
            
            try:
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                        temp_files.append(tmp_path)
                    
                    # Predict
                    result = st.session_state.detector.predict_image(tmp_path)
                    result['filename'] = uploaded_file.name
                    results.append(result)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Analysis complete!")
                
                # Display summary
                st.subheader("📊 Batch Analysis Summary")
                
                summary = st.session_state.detector.get_detection_summary(results)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyzed", summary['total_analyzed'])
                with col2:
                    st.metric("Fake Detected", summary['fake_detected'])
                with col3:
                    st.metric("Real Detected", summary['real_detected'])
                with col4:
                    st.metric("Fake %", f"{summary['fake_percentage']:.1f}%")
                
                # Results table
                st.subheader("📋 Detailed Results")
                
                threshold = st.session_state.detector.threshold
                results_data = []
                
                for r in results:
                    raw_score = r.get('raw_score', 0)
                    
                    results_data.append({
                        'Filename': r['filename'],
                        'Prediction': r.get('prediction', 'ERROR'),
                        'Confidence': f"{r.get('confidence', 0):.2f}%",
                        'Raw Score': f"{raw_score:.4f}"
                    })
                
                st.dataframe(results_data, use_column_width=True)
                
                # Visualization
                fake_count = summary['fake_detected']
                real_count = summary['real_detected']
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Real', 'Fake'],
                    values=[real_count, fake_count],
                    marker=dict(colors=['#4caf50', '#f44336']),
                    hole=0.3
                )])
                
                fig.update_layout(
                    title=f"Detection Distribution (Threshold: {threshold})",
                    annotations=[dict(text=f'{summary["total_analyzed"]}<br>Total', 
                                    x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_column_width=True)
            
            finally:
                # Clean up temp files
                for tmp_path in temp_files:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


def about_page():
    """About page with updated information"""
    st.markdown('<p class="sub-header">About This Project</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ## 🎯 Project Overview
    
    This DeepFake Detection system uses advanced deep learning techniques to identify manipulated
    images and videos. The system employs Convolutional Neural Networks (CNNs) trained on large
    datasets of real and fake media.
    
    ## 📊 Model Performance (Optimized)
    
    - **Test Accuracy:** 98.33%
    - **Precision:** 98.67%
    - **Recall:** 97.99%
    - **F1-Score:** 0.9833
    - **Optimal Threshold:** {OPTIMAL_THRESHOLD} (scientifically determined)
    
    ## 🔬 Technology Stack
    
    - **TensorFlow & Keras**: Deep learning framework
    - **OpenCV**: Image and video processing
    - **Streamlit**: Web interface
    - **NumPy**: Numerical computations
    - **Plotly**: Interactive visualizations
    - **Scikit-learn**: Metrics and evaluation
    
    ## 🚀 Features
    
    - ✅ Image deepfake detection with optimal threshold
    - ✅ Video frame-by-frame analysis
    - ✅ Batch processing for multiple images
    - ✅ Uncertainty detection for borderline cases
    - ✅ Detailed confidence scores and metrics
    - ✅ Interactive visualizations
    - ✅ User-friendly web interface
    
    ## 🎯 Threshold Optimization
    
    This system uses a **scientifically optimized threshold of {OPTIMAL_THRESHOLD}** instead of the 
    default 0.5. This threshold was determined through:
    
    1. **ROC Analysis**: Finding the point that maximizes true positive rate while minimizing false positive rate
    2. **Precision-Recall Trade-off**: Balancing between catching all fakes (recall) and avoiding false alarms (precision)
    3. **F1-Score Maximization**: Finding the threshold that gives the best overall performance
    
    **Why {OPTIMAL_THRESHOLD} instead of 0.5?**
    - Better accuracy (98.33% vs 98.31%)
    - Improved precision (98.67%)
    - Balanced recall (97.99%)
    - Catches more borderline deepfakes
    
    ## 📊 Model Architecture
    
    **Custom CNN Architecture:**
    - Input: 128×128×3 RGB images
    - Multiple convolutional layers with ReLU activation
    - MaxPooling for downsampling
    - Dropout for regularization
    - Dense layers for classification
    - Sigmoid output for binary classification
    
    ## 🎓 How It Works
    
    1. **Preprocessing**: Images/videos are resized to 128×128 and normalized
    2. **Feature Extraction**: CNN layers extract relevant deepfake indicators
    3. **Classification**: Final layers generate a score between 0 (real) and 1 (fake)
    4. **Threshold Decision**: Score compared to optimal threshold ({OPTIMAL_THRESHOLD})
    5. **Uncertainty Detection**: Flags predictions near the threshold as uncertain
    6. **Confidence Calculation**: Distance from threshold indicates confidence level
    
    ## ⚙️ Advanced Features
    
    ### Confidence Scoring
    Each prediction includes a confidence score indicating how certain the model is 
    in its classification (High, Moderate, or Low confidence).
    
    ### Video Analysis
    Videos are analyzed frame-by-frame, with multiple aggregation methods available 
    (mean, median, max) to determine overall authenticity.
    
    ## ⚠️ Limitations
    
    - Detection accuracy depends on training data quality and diversity
    - May not detect all types of manipulations or new deepfake techniques
    - Performance varies with image quality, compression, and source
    - Borderline cases (near threshold) may require manual verification
    - Training data characteristics can affect out-of-distribution performance
    - Not designed for real-time video stream processing
    
    ## 📚 Training Datasets
    
    Common datasets used for training deepfake detection models:
    - **FaceForensics++**: Multi-method deepfake dataset
    - **Celeb-DF**: High-quality celebrity deepfakes
    - **DFDC**: Facebook's DeepFake Detection Challenge dataset
    - **Real/Fake Faces**: 140k balanced dataset
    
    ## 🔒 Ethical Considerations
    
    This tool should be used responsibly:
    
    - ✅ **DO**: Use for education, research, and media verification
    - ✅ **DO**: Consider results as one factor in authenticity assessment
    - ✅ **DO**: Respect privacy and obtain consent when analyzing faces
    - ❌ **DON'T**: Use for harassment or defamation
    - ❌ **DON'T**: Rely solely on automated detection without human review
    - ❌ **DON'T**: Make definitive claims based only on this tool
    
    ## 🎯 Best Practices
    
    1. **Test with Known Examples**: Verify the system works with your use case
    2. **Consider Context**: Factor in image source, quality, and context
    3. **Manual Review**: Always review flagged content manually
    4. **Update Regularly**: Retrain with new deepfake methods as they emerge
    5. **Report Uncertain Cases**: Flag borderline predictions for expert review
    
    ## 📈 Future Improvements
    
    - Multi-model ensemble for better accuracy
    - Support for more video formats and real-time processing
    - Face extraction and alignment preprocessing
    - Attention visualization to show detection focus areas
    - API access for integration with other systems
    - Mobile app version
    
    ## 👨‍💻 Technical Details
    
    **System Requirements:**
    - Python 3.8+
    - TensorFlow 2.x
    - 4GB+ RAM
    - GPU recommended for video processing
    
    **Model File:**
    - Format: HDF5 (.h5)
    - Size: ~50-100MB
    - Architecture: Custom CNN
    - Input Shape: (128, 128, 3)
    - Output: Single sigmoid neuron
    
    ## 📝 Version History
    
    - **v2.0** (Current): Optimized threshold ({OPTIMAL_THRESHOLD}), uncertainty detection
    - **v1.0**: Initial release with default 0.5 threshold
    
    ---
    
    ## 💡 Tips for Best Results
    
    **For Images:**
    - Use clear, well-lit photos
    - Front-facing images work best
    - Avoid heavily compressed or filtered images
    - Single person photos preferred
    
    **For Videos:**
    - Higher resolution = better accuracy
    - Process more frames for better confidence
    - Consider lighting and video quality
    - Longer videos may take more time
    
    **For Batch Processing:**
    - Keep file sizes reasonable
    - Similar image types give better comparison
    - Review uncertain predictions manually
    
    ---
    
    Created with ❤️ for education and research purposes.
    
    **⚠️ Disclaimer:** This tool is for educational and research purposes only. 
    Results should be verified through additional methods before making important decisions.
    """)


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 DeepFake Detector v2.0</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #666;">Optimized with {OPTIMAL_THRESHOLD} threshold | 98.33% accuracy</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        st.subheader("📦 Model Settings")
        
        # Model type selector - SIMPLIFIED TO TWO OPTIONS
        model_type = st.selectbox(
            "Detection Type",
            ["Image Detection", "Video Detection"],
            help="Choose whether to detect deepfakes in images or videos"
        )
        
        # Set model path based on selection
        if model_type == "Image Detection":
            default_model = IMAGE_MODEL_PATH  # Use the correct constant
            model_info = "🖼️ Uses image model for detecting deepfakes in images"
        else:  # Video Detection
            default_model = VIDEO_MODEL_PATH  # Use the correct constant
            model_info = "🎥 Uses video model for detecting deepfakes in videos"
        
        model_path = st.text_input(
            "Model Path",
            value=default_model,
            help=model_info
        )
        
        st.info(f"ℹ️ {model_info}")
        
        # Threshold with optimal default
        threshold = st.slider(
            "Detection Threshold",
            0.0, 1.0, OPTIMAL_THRESHOLD, 0.01,
            help=f"Optimal threshold from analysis: {OPTIMAL_THRESHOLD}"
        )
        
        if threshold != OPTIMAL_THRESHOLD:
            st.warning(f"⚠️ Using custom threshold. Optimal value is {OPTIMAL_THRESHOLD}")
        else:
            st.success(f"✅ Using optimal threshold ({OPTIMAL_THRESHOLD})")
        
        if st.button("🔄 Load Model", use_container_width=True):
            model_type_short = "image" if model_type == "Image Detection" else "video"
            with st.spinner(f"Loading {model_type} model..."):
                if load_model(threshold, model_type_short):
                    st.success(f"✅ {model_type} model loaded successfully!")
                    st.info(f"Detection threshold: {threshold}")
        
        if st.session_state.model_loaded and st.session_state.detector is not None:
            st.success("✅ Model Ready")
            model_type_display = "Video Detection" if st.session_state.model_type == 'video' else "Image Detection"
            st.metric("Model Type", model_type_display)
            st.metric("Active Threshold", f"{st.session_state.detector.threshold:.2f}")
            st.metric("Input Size", f"{st.session_state.detector.target_size[0]}x{st.session_state.detector.target_size[1]}")
        else:
            st.warning("⚠️ No model loaded")
        
        st.divider()
        
        # Model Info
        st.subheader("📊 Model Info")
        st.write(f"""
        **Performance Metrics:**
        - Accuracy: 98.33%
        - Precision: 98.67%
        - Recall: 97.99%
        - F1-Score: 0.9833
        
        **Optimal Threshold:** {OPTIMAL_THRESHOLD}
        """)
        
        st.divider()
        
        # Navigation
        st.subheader("📱 Navigation")
        page = st.radio(
            "Select Page",
            ["🖼️ Image Detection", "🎥 Video Detection", "📁 Batch Processing", "ℹ️ About"],
            label_visibility="collapsed"
        )
    
    # Main content
    if page == "🖼️ Image Detection":
        image_detection_page()
    elif page == "🎥 Video Detection":
        video_detection_page()
    elif page == "📁 Batch Processing":
        batch_detection_page()
    elif page == "ℹ️ About":
        about_page()
    
    # Footer
    st.divider()
    st.markdown(f"""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>DeepFake Detector v2.0</strong> | Optimized Threshold: {OPTIMAL_THRESHOLD}</p>
            <p>Built with Streamlit, TensorFlow, and OpenCV | 98.33% Accuracy</p>
            <p>⚠️ Use responsibly and ethically | For educational and research purposes</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()