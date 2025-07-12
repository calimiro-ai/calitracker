#!/usr/bin/env python3
"""
Debug script to test classification model directly
"""

import cv2
import numpy as np
import tensorflow as tf
import os
from dataset_builder import FeatureExtractor
from realtime_pipeline import ExerciseClassifier


def test_classification_on_video(video_path: str, model_path: str = "models/classification/exercise_classifier.keras"):
    """Test classification on a video file and print detailed results."""
    
    print(f"Testing classification on: {video_path}")
    print(f"Using model: {model_path}")
    
    # Load classifier
    classifier = ExerciseClassifier(model_path, window_size=30)
    print(f"Loaded classifier with classes: {classifier.class_names}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    frame_count = 0
    predictions = []
    confidences = []
    
    print("\nProcessing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get prediction
        exercise, confidence = classifier.update(frame)
        predictions.append(exercise)
        confidences.append(confidence)
        
        frame_count += 1
        
        # Print every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {exercise} (confidence: {confidence:.3f})")
    
    cap.release()
    
    # Analyze results
    print(f"\n{'='*50}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*50}")
    print(f"Total frames: {frame_count}")
    
    # Count predictions
    from collections import Counter
    pred_counts = Counter(predictions)
    
    print(f"\nPrediction distribution:")
    for exercise, count in pred_counts.most_common():
        percentage = count / frame_count * 100
        print(f"  {exercise}: {count} frames ({percentage:.1f}%)")
    
    # Confidence statistics
    if confidences:
        avg_conf = np.mean(confidences)
        max_conf = np.max(confidences)
        min_conf = np.min(confidences)
        print(f"\nConfidence statistics:")
        print(f"  Average: {avg_conf:.3f}")
        print(f"  Maximum: {max_conf:.3f}")
        print(f"  Minimum: {min_conf:.3f}")
    
    # Check for unknown frames
    unknown_count = pred_counts.get('unknown', 0)
    if unknown_count > 0:
        print(f"\nWarning: {unknown_count} frames ({unknown_count/frame_count*100:.1f}%) were classified as 'unknown'")
    
    return predictions, confidences


def test_feature_extraction(video_path: str):
    """Test feature extraction on a few frames."""
    
    print(f"\n{'='*50}")
    print("FEATURE EXTRACTION TEST")
    print(f"{'='*50}")
    
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    features_detected = 0
    
    while frame_count < 10:  # Test first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        features = extractor.extract_angles(frame)
        
        if features is not None:
            features_detected += 1
            print(f"Frame {frame_count}: Features extracted (shape: {features.shape})")
            print(f"  Sample angles: {features[:5]}")
        else:
            print(f"Frame {frame_count}: No pose detected")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nPose detection rate: {features_detected}/{frame_count} frames ({features_detected/frame_count*100:.1f}%)")


def main():
    """Main function."""
    import sys
    
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/raw/squats/squat_10.mp4"
    
    model_path = "models/classification/exercise_classifier.keras"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Test feature extraction
    test_feature_extraction(video_path)
    
    # Test classification
    result = test_classification_on_video(video_path, model_path)
    if result:
        predictions, confidences = result


if __name__ == '__main__':
    main() 