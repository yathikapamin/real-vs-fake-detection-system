#!/usr/bin/env python3
"""
Test script for the updated face extraction consistency classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.detection import DeepFakeDetector

def test_face_extraction_logic():
    """Test the new face extraction consistency logic"""
    print("Testing face extraction consistency classification...")

    # Initialize detector (will try to load model but we won't use it for classification)
    try:
        detector = DeepFakeDetector("roboflow_model_deepfake.h5")
        print("✅ Detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

    # Test predict_image method (should use face detection logic)
    print("\nTesting predict_image method...")
    try:
        # Test with a non-existent image (should return FAKE due to face detection failure)
        result = detector.predict_image("non_existent_image.jpg")
        print(f"Result for non-existent image: {result}")

        if result.get('prediction') == 'FAKE' and 'Face detection failed' in result.get('reason', ''):
            print("✅ predict_image correctly classifies missing face as FAKE")
        else:
            print("❌ predict_image did not classify correctly")
            return False

    except Exception as e:
        print(f"❌ Error testing predict_image: {e}")
        return False

    # Test predict_video method (should use face detection logic and frame-by-frame analysis)
    print("\nTesting predict_video method...")
    try:
        # Test with a video without faces (should return FAKE due to face detection failure)
        result = detector.predict_video("test_clean_video.mp4")
        print(f"Result for video without faces: {result}")

        if result.get('prediction') == 'FAKE' and 'Face detection failed' in result.get('reason', ''):
            print("✅ predict_video correctly classifies video without faces as FAKE")
        else:
            print("❌ predict_video did not classify correctly")
            return False

        # Test that frame-by-frame analysis works (check for frame_results key)
        if 'frame_results' in result or 'frames_analyzed' in result:
            print("✅ predict_video includes frame analysis data")
        else:
            print("⚠️ predict_video may not be using frame-by-frame analysis")

    except Exception as e:
        print(f"❌ Error testing predict_video: {e}")
        return False

    print("\n✅ All tests passed! Face extraction consistency logic is working.")
    return True

if __name__ == "__main__":
    success = test_face_extraction_logic()
    sys.exit(0 if success else 1)