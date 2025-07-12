#!/usr/bin/env python3
"""
Test script for realtime pipeline using training videos
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import argparse
import os
from collections import deque
from typing import Optional, Dict, List, Tuple

from dataset_builder import FeatureExtractor
from realtime_pipeline import ExerciseClassifier, RepCounter


class VideoTestPipeline:
    """Test pipeline that processes video files instead of live streams."""
    
    def __init__(self, 
                 video_path: str,
                 classifier_model: str = "models/classification/exercise_classifier.keras",
                 segmentation_models_dir: str = "models/segmentation",
                 window_size: int = 30):
        """
        Initialize the test pipeline.
        
        Args:
            video_path: Path to the video file to test
            classifier_model: Path to classification model
            segmentation_models_dir: Directory with segmentation models
            window_size: Window size for classification
        """
        self.video_path = video_path
        self.classifier = ExerciseClassifier(classifier_model, window_size)
        self.rep_counter = RepCounter(segmentation_models_dir)
        self.current_exercise = "unknown"
        self.exercise_confidence = 0.0
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        # Results tracking
        self.exercise_predictions = []
        self.confidence_scores = []
        self.rep_counts = []
        
        print(f"Test pipeline initialized with video: {video_path}")
    
    def run(self, output_path: Optional[str] = None, display: bool = True):
        """Run the test pipeline on the video file."""
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Initialize video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting video processing. Press 'q' to quit, 'r' to reset counts.")
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Process frame
                self._process_frame(frame)
                
                # Calculate FPS
                self._update_fps()
                
                # Draw overlay
                self._draw_overlay(frame)
                
                # Write frame if output is specified
                if writer:
                    writer.write(frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Video Test - Real-Time Exercise Detection', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self._reset_all_counts()
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count}/{frame_count} frames "
                          f"({self.frame_count/frame_count*100:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print final results
        self._print_results()
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame through the pipeline."""
        # Update exercise classifier
        self.current_exercise, self.exercise_confidence = self.classifier.update(frame)
        
        # Store predictions for analysis
        self.exercise_predictions.append(self.current_exercise)
        self.confidence_scores.append(self.exercise_confidence)
        
        # Update rep counter for current exercise
        if self.current_exercise != "unknown":
            rep_count = self.rep_counter.update(frame, self.current_exercise)
            self.rep_counts.append(rep_count)
        else:
            self.rep_counts.append(0)
        
        self.frame_count += 1
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _draw_overlay(self, frame: np.ndarray):
        """Draw information overlay on frame."""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Exercise information
        exercise_text = f"Exercise: {self.current_exercise.upper()}"
        confidence_text = f"Confidence: {self.exercise_confidence:.3f}"
        
        # Color based on confidence
        if self.exercise_confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif self.exercise_confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, exercise_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2)
        cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Rep counts for all exercises
        y_offset = 100
        for exercise, count in self.rep_counter.rep_counts.items():
            if exercise == self.current_exercise:
                text_color = (0, 255, 0)  # Green for current exercise
            else:
                text_color = (255, 255, 255)  # White for others
            
            cv2.putText(frame, f"{exercise}: {count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 25
        
        # Performance metrics
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000  # Convert to ms
            cv2.putText(frame, f"FPS: {self.fps}", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Latency: {avg_time:.1f}ms", (width - 150, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _reset_all_counts(self):
        """Reset all rep counts."""
        for exercise in self.rep_counter.rep_counts:
            self.rep_counter.reset_count(exercise)
        print("All rep counts reset")
    
    def _print_results(self):
        """Print final analysis results."""
        print("\n" + "="*50)
        print("VIDEO PROCESSING RESULTS")
        print("="*50)
        
        # Exercise classification results
        print(f"\nExercise Classification:")
        print(f"Total frames processed: {len(self.exercise_predictions)}")
        
        # Count predictions for each exercise
        from collections import Counter
        exercise_counts = Counter(self.exercise_predictions)
        for exercise, count in exercise_counts.most_common():
            percentage = count / len(self.exercise_predictions) * 100
            print(f"  {exercise}: {count} frames ({percentage:.1f}%)")
        
        # Confidence statistics
        if self.confidence_scores:
            avg_confidence = np.mean(self.confidence_scores)
            max_confidence = np.max(self.confidence_scores)
            min_confidence = np.min(self.confidence_scores)
            print(f"\nConfidence Statistics:")
            print(f"  Average: {avg_confidence:.3f}")
            print(f"  Maximum: {max_confidence:.3f}")
            print(f"  Minimum: {min_confidence:.3f}")
        
        # Rep counting results
        print(f"\nRep Counting Results:")
        for exercise, count in self.rep_counter.rep_counts.items():
            print(f"  {exercise}: {count} reps")
        
        # Performance statistics
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            max_time = np.max(self.processing_times) * 1000
            min_time = np.min(self.processing_times) * 1000
            print(f"\nPerformance Statistics:")
            print(f"  Average processing time: {avg_time:.1f}ms")
            print(f"  Maximum processing time: {max_time:.1f}ms")
            print(f"  Minimum processing time: {min_time:.1f}ms")
            print(f"  Theoretical max FPS: {1000/avg_time:.1f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Test realtime pipeline with video files'
    )
    parser.add_argument('video_path', type=str,
                       help='Path to the video file to test')
    parser.add_argument('--classifier', type=str, 
                       default='models/classification/exercise_classifier.keras',
                       help='Path to classification model')
    parser.add_argument('--models-dir', type=str, 
                       default='models/segmentation',
                       help='Directory containing segmentation models')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size for classification')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (faster processing)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Create and run test pipeline
    pipeline = VideoTestPipeline(
        video_path=args.video_path,
        classifier_model=args.classifier,
        segmentation_models_dir=args.models_dir,
        window_size=args.window_size
    )
    
    pipeline.run(
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    main() 