"""
Preprocessing utilities for images and videos
"""

import cv2
import numpy as np
from PIL import Image
import os


class ImagePreprocessor:
    """
    Image preprocessing utilities
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target size for images (width, height)
        """
        self.target_size = target_size
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.target_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            return img
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def preprocess_for_prediction(self, image_path):
        """
        Preprocess image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image array with batch dimension
        """
        img = self.load_and_preprocess_image(image_path)
        if img is not None:
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
        return img
    
    def detect_face(self, image_path):
        """
        Detect face in image using Haar Cascade
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Cropped face image or original image if no face detected
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # If face detected, crop it
            if len(faces) > 0:
                # Get the largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                
                # Add some padding
                padding = int(0.2 * max(w, h))
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)
                
                # Crop face
                face_img = img[y:y+h, x:x+w]
                return face_img
            
            # If no face detected, return original image
            return img
        
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return None


class VideoPreprocessor:
    """
    Video preprocessing utilities
    """
    
    def __init__(self, target_size=(224, 224), frame_skip=5):
        """
        Initialize video preprocessor
        
        Args:
            target_size: Target size for frames
            frame_skip: Number of frames to skip (process every nth frame)
        """
        self.target_size = target_size
        self.frame_skip = frame_skip
        self.image_preprocessor = ImagePreprocessor(target_size)
    
    def extract_frames(self, video_path, max_frames=None):
        """
        Extract frames from video - TRAINING STYLE (evenly spaced)
        This matches the Colab training extraction method
        
        Args:
            video_path: Path to video file
            max_frames: Number of frames to extract (default 10 for video model)
            
        Returns:
            Array of preprocessed frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get total frames in video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Default to 10 frames for video model
            frame_count = max_frames if max_frames else 10
            
            # Calculate step to get evenly spaced frames
            step = max(total_frames // frame_count, 1)
            
            frames = []
            
            for i in range(frame_count):
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                frame = cv2.resize(frame, self.target_size)
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
            
            cap.release()
            
            print(f"Extracted {len(frames)} evenly-spaced frames from {total_frames} total frames")
            return np.array(frames)
        
        except Exception as e:
            print(f"Error extracting frames from video: {str(e)}")
            return None
    
    def extract_faces_from_video(self, video_path, max_frames=None):
        """
        Extract faces from video frames
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            List of face images
        """
        try:
            faces = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % self.frame_skip == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    detected_faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    # Process detected faces
                    for (x, y, w, h) in detected_faces:
                        # Add padding
                        padding = int(0.2 * max(w, h))
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2*padding)
                        h = min(frame.shape[0] - y, h + 2*padding)
                        
                        # Crop face
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Convert to RGB
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        # Resize
                        face_img = cv2.resize(face_img, self.target_size)
                        
                        # Normalize
                        face_img = face_img.astype(np.float32) / 255.0
                        
                        faces.append(face_img)
                        extracted_count += 1
                        
                        if max_frames and extracted_count >= max_frames:
                            break
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            print(f"Extracted {len(faces)} faces from {frame_count} frames")
            return np.array(faces) if faces else None
        
        except Exception as e:
            print(f"Error extracting faces from video: {str(e)}")
            return None
    
    def get_video_info(self, video_path):
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
        
        except Exception as e:
            print(f"Error getting video info: {str(e)}")
            return None
    
    def extract_frames_with_faces(self, video_path, max_frames=10):
        """
        Extract frames from video with face detection information
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of dictionaries with frame data and face detection results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate step to get evenly spaced frames
            step = max(total_frames // max_frames, 1)
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            frames_data = []
            
            for i in range(max_frames):
                # Seek to specific frame
                frame_number = i * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Create frame with face boxes for display
                frame_with_boxes = frame_rgb.copy()
                has_faces = len(faces) > 0
                
                if has_faces:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Calculate timestamp
                timestamp = frame_number / fps
                
                frame_data = {
                    'frame': frame_rgb,
                    'frame_with_boxes': frame_with_boxes,
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'has_faces': has_faces,
                    'faces': faces.tolist() if has_faces else []
                }
                
                frames_data.append(frame_data)
            
            cap.release()
            
            print(f"Extracted {len(frames_data)} frames with face detection")
            return frames_data
        
        except Exception as e:
            print(f"Error extracting frames with faces: {str(e)}")
            return []
    
    def save_frames_temp(self, frames_data):
        """
        Save frames to temporary files
        
        Args:
            frames_data: List of frame data dictionaries
            
        Returns:
            List of temporary file paths
        """
        import tempfile
        import os
        
        temp_paths = []
        
        try:
            for i, frame_data in enumerate(frames_data):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    temp_path = tmp_file.name
                
                # Save frame as JPEG
                frame_img = Image.fromarray(frame_data['frame'])
                frame_img.save(temp_path, 'JPEG')
                
                temp_paths.append(temp_path)
            
            print(f"Saved {len(temp_paths)} frames to temporary files")
            return temp_paths
        
        except Exception as e:
            print(f"Error saving frames to temp: {str(e)}")
            return []
    
    def cleanup_temp_files(self, temp_paths):
        """
        Clean up temporary files
        
        Args:
            temp_paths: List of temporary file paths to delete
        """
        try:
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            print(f"Cleaned up {len(temp_paths)} temporary files")
        except Exception as e:
            print(f"Error cleaning up temp files: {str(e)}")


if __name__ == "__main__":
    # Example usage
    img_preprocessor = ImagePreprocessor()
    video_preprocessor = VideoPreprocessor()
    
    print("Preprocessing utilities loaded successfully!")
