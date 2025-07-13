#!/usr/bin/env python3
"""
Test video segmentation pipeline with enhanced 30-frame processing and live rep counting
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import argparse
import os
import sys
import threading
from collections import deque
from typing import Optional, Dict, List, Tuple
from queue import Queue
import csv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.dataset_builder import FeatureExtractor
from core.realtime_pipeline import ExerciseClassifier


class PeakDetector:
    """Simple peak detection for exercise repetition counting with sparse data."""
    
    def __init__(self, 
                 min_peak_distance: int = 3,
                 min_threshold: float = 0.5):
        """
        Initialize simple peak detector.
        
        Args:
            min_peak_distance: Minimum data points between peaks
            min_threshold: Minimum probability threshold for peak detection
        """
        self.min_peak_distance = min_peak_distance
        self.min_threshold = min_threshold
        
        # State tracking
        self.probability_history = deque(maxlen=10)  # Store recent probabilities
        self.frame_history = deque(maxlen=10)       # Store corresponding frame indices
        self.peak_frames = []
        self.peak_probabilities = []  # Store peak probability values
        self.rep_count = 0
        self.last_peak_frame = -min_peak_distance
        
        # Peak detection state
        self.last_probability = 0.0
        self.rising = False
        self.peak_candidate = None
    
    def update(self, frame_idx: int, probability: float) -> bool:
        """
        Update peak detector with new probability value.
        
        Args:
            frame_idx: Current frame index
            probability: Current segmentation probability
            
        Returns:
            True if a new peak was detected, False otherwise
        """
        # Add to history
        self.probability_history.append(probability)
        self.frame_history.append(frame_idx)
        
        # Need at least 2 values for comparison
        if len(self.probability_history) < 2:
            self.last_probability = probability
            return False
        
        # Simple peak detection: rising then falling
        peak_detected = False
        
        if probability > self.last_probability:
            # Rising - update peak candidate to highest point
            if not self.rising:
                self.rising = True
            # Always update peak candidate to the highest point while rising
            self.peak_candidate = (frame_idx, probability)
        elif self.rising and probability < self.last_probability:
            # We were rising and now falling - this is a peak
            if self.peak_candidate is not None:
                peak_frame, peak_prob = self.peak_candidate
                
                # Check minimum distance and threshold
                if (frame_idx - self.last_peak_frame >= self.min_peak_distance and 
                    peak_prob >= self.min_threshold):
                    self.peak_frames.append(peak_frame)
                    self.peak_probabilities.append(peak_prob)
                    self.rep_count += 1
                    self.last_peak_frame = peak_frame
                    peak_detected = True
                    print(f"Peak detected at frame {peak_frame} with probability {peak_prob:.3f}")
                elif peak_prob < self.min_threshold:
                    print(f"Peak rejected at frame {peak_frame} with probability {peak_prob:.3f} (below threshold {self.min_threshold})")
                
                self.peak_candidate = None
            
            self.rising = False
        
        self.last_probability = probability
        return peak_detected
    
    def get_rep_count(self) -> int:
        """Get current repetition count."""
        return self.rep_count
    
    def get_peak_frames(self) -> List[int]:
        """Get list of peak frame indices."""
        return self.peak_frames.copy()
    
    def get_peak_probabilities(self) -> List[float]:
        """Get list of peak probability values."""
        return self.peak_probabilities.copy()
    
    def reset(self):
        """Reset peak detector state."""
        self.probability_history.clear()
        self.frame_history.clear()
        self.peak_frames.clear()
        self.peak_probabilities.clear()
        self.rep_count = 0
        self.last_peak_frame = -self.min_peak_distance
        self.last_probability = 0.0
        self.rising = False
        self.peak_candidate = None


class SegmentationProcessor:
    """Real-time segmentation processor for exercise repetition detection."""
    
    def __init__(self, models_dir: str = "models/segmentation"):
        """
        Initialize segmentation processor with exercise-specific models.
        
        Args:
            models_dir: Directory containing segmentation models
        """
        self.models_dir = models_dir
        self.models = {}
        self.exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips']
        
        # Load all segmentation models
        for exercise in self.exercise_types:
            model_path = os.path.join(models_dir, f"{exercise}.keras")
            if os.path.exists(model_path):
                try:
                    self.models[exercise] = tf.keras.models.load_model(model_path, compile=False)
                    print(f"Loaded {exercise} segmentation model")
                except Exception as e:
                    print(f"Error loading {exercise} model: {e}")
            else:
                print(f"Warning: {exercise} segmentation model not found at {model_path}")
        
        # CSV logging for probabilities
        self.csv_file = open("video_segmentation_probabilities.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "timestamp", "exercise", "probability"])
        self.frame_counter = 0
    
    def predict_window(self, window: np.ndarray, exercise_type: str) -> float:
        """
        Predict segmentation probability for a 30-frame window.
        
        Args:
            window: 30-frame sequence of joint angles (30, 25)
            exercise_type: Type of exercise to predict for
            
        Returns:
            Probability score (0-1) indicating if window contains a rep
        """
        if exercise_type not in self.models:
            return 0.0
        
        # Model expects (1, window_size, num_features)
        sequence = window[None, ...]
        
        # Get prediction - model outputs (1, window_size, 1)
        # We take the last frame probability as the current rep probability
        probability = self.models[exercise_type].predict(sequence, verbose=0)
        prob_value = float(probability[0, -1, 0])  # Last frame probability
        
        # Log to CSV
        self.frame_counter += 1
        self.csv_writer.writerow([self.frame_counter, time.time(), exercise_type, prob_value])
        
        # Flush every 10 predictions for performance
        if self.frame_counter % 10 == 0:
            self.csv_file.flush()
        
        return prob_value
    
    def close(self):
        """Close CSV file."""
        self.csv_file.close()


class VideoSegmentationTestPipeline:
    """Video test pipeline with enhanced segmentation processing and live rep counting."""
    
    def __init__(self, 
                 video_path: str,
                 classifier_model: str = "models/classification/exercise_classifier.keras",
                 segmentation_models_dir: str = "models/segmentation",
                 window_size: int = 30):
        """
        Initialize the video test pipeline.
        
        Args:
            video_path: Path to the video file to test
            classifier_model: Path to classification model
            segmentation_models_dir: Directory with segmentation models
            window_size: Window size for classification and segmentation
        """
        self.video_path = video_path
        self.classifier = ExerciseClassifier(classifier_model, window_size)
        self.segmentation = SegmentationProcessor(segmentation_models_dir)
        self.feature_extractor = FeatureExtractor()
        self.window_size = window_size
        
        # Peak detection for each exercise type with 0.5 threshold
        self.peak_detectors = {
            'push-ups': PeakDetector(min_peak_distance=5, min_threshold=0.5),
            'squats': PeakDetector(min_peak_distance=5, min_threshold=0.5),
            'pull-ups': PeakDetector(min_peak_distance=5, min_threshold=0.5),
            'dips': PeakDetector(min_peak_distance=5, min_threshold=0.5)
        }
        
        # Exercise change filtering
        self.exercise_history = deque(maxlen=3)  # Store last 3 predictions
        self.stable_exercise = "unknown"
        
        # Shared state between threads
        self.current_window = np.zeros((window_size, 25), dtype=np.float32)
        self.window_filled = False
        self.current_exercise = "unknown"
        self.exercise_confidence = 0.0
        self.current_probability = 0.0
        self.current_rep_count = 0
        self.frame_count = 0
        self.processed_frame_count = 0
        
        # Threading
        self.feature_queue = Queue(maxsize=100)  # Buffer for features
        self.result_queue = Queue(maxsize=10)    # Buffer for results
        self.running = True
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        # Results tracking
        self.exercise_predictions = []
        self.confidence_scores = []
        self.probability_scores = []
        self.rep_counts = []
        
        print(f"Video segmentation test pipeline initialized with video: {video_path}")
    
    def _filter_exercise_change(self, new_exercise: str) -> str:
        """
        Filter exercise changes to only occur after 3 consecutive predictions.
        
        Args:
            new_exercise: New exercise prediction
            
        Returns:
            Filtered exercise prediction
        """
        self.exercise_history.append(new_exercise)
        
        # Need at least 3 predictions
        if len(self.exercise_history) < 3:
            return self.stable_exercise
        
        # Check if last 3 predictions are the same
        last_three = list(self.exercise_history)[-3:]
        if len(set(last_three)) == 1 and last_three[0] != "unknown":
            # All 3 are the same and not unknown - change exercise
            self.stable_exercise = last_three[0]
        
        return self.stable_exercise
    
    def run(self, output_path: Optional[str] = None):
        """Run the video test pipeline."""
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
        
        # Start background processing thread
        processing_thread = threading.Thread(target=self._background_processor)
        processing_thread.start()
        
        print("Starting video segmentation processing with live rep counting...")
        print("Video will play at original speed with real-time analysis overlay")
        
        # Calculate frame delay for original speed
        frame_delay = 1.0 / fps
        
        try:
            while True:
                frame_start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Extract features and add to queue (non-blocking)
                features = self.feature_extractor.extract_angles(frame)
                if features is not None:
                    try:
                        self.feature_queue.put_nowait((self.frame_count, features))
                    except:
                        pass  # Queue full, skip this frame
                
                # Update current window (shift and add new frame)
                if features is not None:
                    self.current_window = np.roll(self.current_window, -1, axis=0)
                    self.current_window[-1] = features
                    
                    # Check if window is filled
                    if not self.window_filled and self.frame_count >= self.window_size - 1:
                        self.window_filled = True
                
                # Check for new results from background thread
                try:
                    while not self.result_queue.empty():
                        exercise, confidence, probability, rep_count = self.result_queue.get_nowait()
                        self.current_exercise = exercise
                        self.exercise_confidence = confidence
                        self.current_probability = probability
                        self.current_rep_count = rep_count
                except:
                    pass
                
                # Store predictions for analysis
                self.exercise_predictions.append(self.current_exercise)
                self.confidence_scores.append(self.exercise_confidence)
                self.probability_scores.append(self.current_probability)
                self.rep_counts.append(self.current_rep_count)
                
                # Draw overlay
                self._draw_overlay(frame)
                
                # Write frame if output is specified
                if writer:
                    writer.write(frame)
                
                # Display frame
                cv2.imshow('Video Segmentation Test - Real-Time Exercise Detection with Rep Counting', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_all_counts()
                
                # Maintain original video speed
                processing_time = time.time() - frame_start_time
                if processing_time < frame_delay:
                    time.sleep(frame_delay - processing_time)
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count}/{frame_count} frames "
                          f"({self.frame_count/frame_count*100:.1f}%) - Reps: {self.current_rep_count}")
                
                self.frame_count += 1
        
        finally:
            # Stop background processing
            self.running = False
            processing_thread.join()
            
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.segmentation.close()
        
        # Print final results and plot probabilities
        self._print_results()
        self._plot_probabilities()
    
    def _background_processor(self):
        """Background thread for model inference."""
        while self.running:
            try:
                # Get features from queue
                frame_idx, features = self.feature_queue.get(timeout=0.1)
                
                # Update window
                self.current_window = np.roll(self.current_window, -1, axis=0)
                self.current_window[-1] = features
                
                # Check if window is filled
                if not self.window_filled and frame_idx >= self.window_size - 1:
                    self.window_filled = True
                
                # Run models every 10th frame when window is filled
                if self.window_filled and frame_idx % 10 == 0:
                    start_time = time.time()
                    
                    # Use the current 30-frame window
                    window = self.current_window.copy()
                    
                    # Classification
                    exercise, confidence = self.classifier.predict_window(window)
                    
                    # Filter exercise change
                    self.current_exercise = self._filter_exercise_change(exercise)
                    self.exercise_confidence = confidence
                    
                    # Segmentation and rep counting
                    probability = 0.0
                    rep_count = 0
                    
                    if self.current_exercise != "unknown":
                        probability = self.segmentation.predict_window(window, self.current_exercise)
                        
                        # Update peak detector for this exercise
                        if self.current_exercise in self.peak_detectors:
                            peak_detected = self.peak_detectors[self.current_exercise].update(frame_idx, probability)
                            rep_count = self.peak_detectors[self.current_exercise].get_rep_count()
                    
                    # Add result to queue
                    try:
                        self.result_queue.put_nowait((self.current_exercise, self.exercise_confidence, probability, rep_count))
                    except:
                        pass  # Queue full, skip result
                    
                    # Track processing time
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    self.processed_frame_count += 1
                
            except:
                continue  # Timeout or queue empty, continue
    
    def _draw_overlay(self, frame: np.ndarray):
        """Draw information overlay on frame."""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Exercise information
        exercise_text = f"Exercise: {self.current_exercise.upper()}"
        confidence_text = f"Confidence: {self.exercise_confidence:.3f}"
        probability_text = f"Rep Probability: {self.current_probability:.3f}"
        rep_count_text = f"Rep Count: {self.current_rep_count}"
        
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
        
        # Segmentation probability with color coding
        if self.current_probability > 0.7:
            prob_color = (0, 255, 0)  # Green - high probability of rep
        elif self.current_probability > 0.5:
            prob_color = (0, 255, 255)  # Yellow - medium probability
        else:
            prob_color = (0, 0, 255)  # Red - low probability
        
        cv2.putText(frame, probability_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, prob_color, 2)
        
        # Rep count with highlight color
        rep_color = (0, 255, 255)  # Cyan for rep count
        cv2.putText(frame, rep_count_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, rep_color, 2)
        
        # Performance metrics
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000  # Convert to ms
            cv2.putText(frame, f"Model FPS: {1000/avg_time:.1f}", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Video FPS: 30.0", (width - 150, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _reset_all_counts(self):
        """Reset all rep counts."""
        for detector in self.peak_detectors.values():
            detector.reset()
        self.current_rep_count = 0
        print("Rep counts reset")
    
    def _print_results(self):
        """Print final analysis results."""
        print("\n" + "="*50)
        print("VIDEO SEGMENTATION PROCESSING RESULTS")
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
        
        # Rep counting results
        print(f"\nRep Counting Results:")
        for exercise, detector in self.peak_detectors.items():
            rep_count = detector.get_rep_count()
            peak_frames = detector.get_peak_frames()
            print(f"  {exercise}: {rep_count} reps detected at frames {peak_frames}")
        
        # Final rep count
        final_rep_count = max(self.rep_counts) if self.rep_counts else 0
        print(f"  Final rep count: {final_rep_count}")
        
        # Confidence statistics
        if self.confidence_scores:
            avg_confidence = np.mean(self.confidence_scores)
            max_confidence = np.max(self.confidence_scores)
            min_confidence = np.min(self.confidence_scores)
            print(f"\nConfidence Statistics:")
            print(f"  Average: {avg_confidence:.3f}")
            print(f"  Maximum: {max_confidence:.3f}")
            print(f"  Minimum: {min_confidence:.3f}")
        
        # Segmentation probability statistics
        if self.probability_scores:
            avg_probability = np.mean(self.probability_scores)
            max_probability = np.max(self.probability_scores)
            min_probability = np.min(self.probability_scores)
            print(f"\nSegmentation Probability Statistics:")
            print(f"  Average: {avg_probability:.3f}")
            print(f"  Maximum: {max_probability:.3f}")
            print(f"  Minimum: {min_probability:.3f}")
        
        # Performance statistics
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            max_time = np.max(self.processing_times) * 1000
            min_time = np.min(self.processing_times) * 1000
            print(f"\nPerformance Statistics:")
            print(f"  Model inference time: {avg_time:.1f}ms")
            print(f"  Maximum inference time: {max_time:.1f}ms")
            print(f"  Minimum inference time: {min_time:.1f}ms")
            print(f"  Model FPS: {1000/avg_time:.1f}")
            print(f"  Video FPS: 30.0 (original speed)")
            print(f"  Frames analyzed: {self.processed_frame_count}")
    
    def _plot_probabilities(self):
        """Plot segmentation probabilities with peaks in a single diagram."""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Read the CSV file
            df = pd.read_csv("video_segmentation_probabilities.csv")
            
            if len(df) == 0:
                print("No segmentation data to plot")
                return
            
            # Create the plot
            plt.figure(figsize=(15, 8))
            
            # Plot probabilities over time (sparse data - 3 per second)
            for exercise in self.segmentation.exercise_types:
                exercise_data = df[df['exercise'] == exercise]
                if len(exercise_data) > 0:
                    plt.plot(exercise_data['frame'], exercise_data['probability'], 
                            label=exercise, linewidth=2, alpha=0.8, marker='o', markersize=4)
            
            plt.xlabel('Frame Number')
            plt.ylabel('Repetition Probability')
            plt.title('Exercise Repetition Detection with Peak Detection (Sparse Data - 3/s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('video_segmentation_with_peaks_plot.png', dpi=300, bbox_inches='tight')
            print(f"Video segmentation with peaks plot saved to: video_segmentation_with_peaks_plot.png")
            
            # Show the plot
            plt.show()
            
        except ImportError:
            print("Matplotlib not available, skipping probability plot")
        except Exception as e:
            print(f"Error plotting probabilities: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Test video segmentation pipeline with enhanced processing and rep counting'
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
                       help='Window size for classification and segmentation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Create and run video test pipeline
    pipeline = VideoSegmentationTestPipeline(
        video_path=args.video_path,
        classifier_model=args.classifier,
        segmentation_models_dir=args.models_dir,
        window_size=args.window_size
    )
    
    pipeline.run(output_path=args.output)


if __name__ == '__main__':
    main() 