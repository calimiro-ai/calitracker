#!/usr/bin/env python3
"""
Test script for the real-time pipeline using a test video.

This script tests the pipeline with a video file to ensure everything works
before deploying with live streams.
"""

import cv2
import numpy as np
import time
from realtime_pipeline import RealTimePipeline


def test_with_video(video_path: str, output_path: str | None = None):
    """
    Test the real-time pipeline with a video file.
    
    Args:
        video_path: Path to test video
        output_path: Optional output video path for recording
    """
    # Initialize pipeline
    pipeline = RealTimePipeline(
        classifier_model="models/classification/exercise_classifier.keras",
        segmentation_models_dir="models/segmentation",
        window_size=30
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Testing pipeline with video: {video_path}")
    print(f"Video properties: {width}x{height} @ {fps}fps")
    print("Press 'q' to quit, 'r' to reset counts")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame through pipeline
            pipeline._process_frame(frame)
            
            # Draw overlay
            pipeline._draw_overlay(frame)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Test Pipeline', frame)
            
            # Write to output if specified
            if out:
                out.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pipeline._reset_all_counts()
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.1f}s")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"\nTest completed:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {frame_count/total_time:.1f}")
        print(f"  Final exercise: {pipeline.current_exercise}")
        print(f"  Final confidence: {pipeline.exercise_confidence:.3f}")
        print(f"  Rep counts: {pipeline.rep_counter.rep_counts}")


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test real-time pipeline with video')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', help='Output video file (optional)')
    
    args = parser.parse_args()
    
    test_with_video(args.input, args.output)


if __name__ == '__main__':
    main() 