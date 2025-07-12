#!/usr/bin/env python3
"""
Simple webcam test to verify basic functionality.
"""

import cv2
import numpy as np
import time

def test_webcam():
    """Test basic webcam functionality."""
    print("Testing webcam access...")
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Webcam Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"Test completed: {frame_count} frames in {total_time:.1f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        
    return True

def test_model_loading():
    """Test if models can be loaded without errors."""
    print("\nTesting model loading...")
    
    try:
        import tensorflow as tf
        from dataset_builder import FeatureExtractor
        
        # Test feature extractor
        print("Loading feature extractor...")
        extractor = FeatureExtractor()
        print("✓ Feature extractor loaded successfully")
        
        # Test classification model
        classifier_path = "models/classification/exercise_classifier.keras"
        if os.path.exists(classifier_path):
            print("Loading classification model...")
            model = tf.keras.models.load_model(classifier_path, compile=False)
            print("✓ Classification model loaded successfully")
        else:
            print("⚠ Classification model not found")
        
        # Test segmentation models
        seg_dir = "models/segmentation"
        if os.path.exists(seg_dir):
            models_loaded = 0
            for exercise in ['push-ups', 'squats', 'pull-ups', 'dips']:
                model_path = os.path.join(seg_dir, f"{exercise}.keras")
                if os.path.exists(model_path):
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
                        models_loaded += 1
                        print(f"✓ {exercise} model loaded")
                    except Exception as e:
                        print(f"✗ {exercise} model failed: {e}")
                else:
                    print(f"⚠ {exercise} model not found")
            
            print(f"Loaded {models_loaded}/4 segmentation models")
        else:
            print("⚠ Segmentation models directory not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

if __name__ == '__main__':
    import os
    
    print("=== Simple Webcam and Model Test ===\n")
    
    # Test webcam
    webcam_ok = test_webcam()
    
    # Test model loading
    models_ok = test_model_loading()
    
    print(f"\n=== Test Results ===")
    print(f"Webcam: {'✓ OK' if webcam_ok else '✗ FAILED'}")
    print(f"Models: {'✓ OK' if models_ok else '✗ FAILED'}")
    
    if webcam_ok and models_ok:
        print("\nAll tests passed! You can now run the full pipeline.")
    else:
        print("\nSome tests failed. Please check the issues above.") 