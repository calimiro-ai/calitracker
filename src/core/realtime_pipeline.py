#!/usr/bin/env python3
"""
Real-Time Exercise Detection & Rep Counting Pipeline

This module provides a complete real-time pipeline that:
1. Captures video from ESP32-CAM or webcam
2. Classifies the current exercise type
3. Counts repetitions using exercise-specific segmentation models
4. Displays results in real-time with minimal latency
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from collections import deque
from typing import Optional, Dict, List, Tuple
import argparse
import os
import csv

from .dataset_builder import FeatureExtractor


class ExerciseClassifier:
    """Real-time exercise classifier using TCN model."""
    
    def __init__(self, model_path: str, window_size: int = 30):
        """
        Initialize the exercise classifier.
        
        Args:
            model_path: Path to the trained classification model
            window_size: Number of frames to use for classification
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.window_size = window_size
        self.feature_buffer = deque(maxlen=window_size)
        self.extractor = FeatureExtractor()
        
        # Load class names
        class_names_path = model_path.replace('.keras', '_classes.txt')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = ['dips', 'pull-ups', 'push-ups', 'squats']
        
        print(f"Loaded classifier with {len(self.class_names)} classes: {self.class_names}")
    
    def update(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Update classifier with new frame and get prediction.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (predicted_exercise, confidence)
        """
        # Extract features
        features = self.extractor.extract_angles(frame)
        if features is None:
            # If no pose detected, use zero vector
            features = np.zeros(25, dtype=np.float32)
        
        # Add to buffer
        self.feature_buffer.append(features)
        
        # Only predict if we have enough frames
        if len(self.feature_buffer) < self.window_size:
            return "unknown", 0.0
        
        # Prepare sequence for prediction
        sequence = np.array(list(self.feature_buffer))
        sequence = sequence[None, ...]  # Add batch dimension
        
        # Get prediction
        probabilities = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(probabilities[0])
        confidence = float(probabilities[0][predicted_class])
        
        return self.class_names[predicted_class], confidence

    def predict_window(self, window: np.ndarray) -> Tuple[str, float]:
        # window shape: (window_size, num_features)
        # Modell erwartet (1, window_size, num_features)
        X = window[None, ...]
        preds = self.model.predict(X, verbose=0)
        pred = preds[0, -1]  # Letztes Zeitfenster
        class_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        class_name = self.class_names[class_idx] if confidence > 0.5 else "unknown"
        return class_name, confidence


class RepCounter:
    """Real-time repetition counter using segmentation models."""
    
    def __init__(self, models_dir: str = "models/segmentation"):
        """
        Initialize rep counter with exercise-specific models.
        
        Args:
            models_dir: Directory containing segmentation models
        """
        self.models_dir = models_dir
        self.models = {}
        self.extractors = {}
        self.feature_buffers = {}
        self.probability_buffers = {}
        self.last_prob = {}
        self.csv_file = open("probabilities.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "timestamp", "exercise", "probability"])
        self.frame_counter = 0
        exercise_types = ['push-ups', 'squats', 'pull-ups', 'dips']
        for exercise in exercise_types:
            model_path = os.path.join(models_dir, f"{exercise}.keras")
            if os.path.exists(model_path):
                self.models[exercise] = tf.keras.models.load_model(model_path, compile=False)
                self.extractors[exercise] = FeatureExtractor()
                self.feature_buffers[exercise] = deque(maxlen=30)
                self.probability_buffers[exercise] = deque(maxlen=30)
                self.last_prob[exercise] = 0.0
                print(f"Loaded {exercise} segmentation model")
            else:
                print(f"Warning: {exercise} segmentation model not found")

    def update_with_window(self, window: np.ndarray, exercise_type: str) -> float:
        if exercise_type not in self.models:
            return 0.0
        sequence = window[None, ...]
        probability = self.models[exercise_type].predict(sequence, verbose=0)
        prob_value = float(probability[0, -1, 0])
        self.last_prob[exercise_type] = prob_value
        # Log to CSV
        self.frame_counter += 1
        self.csv_writer.writerow([self.frame_counter, time.time(), exercise_type, prob_value])
        self.csv_file.flush()
        return prob_value

    def close(self):
        self.csv_file.close()


class RealTimePipeline:
    """Main real-time pipeline combining classification and rep counting.

    Args:
        stream_url: URL for video stream (None for webcam)
        classifier_model: Path to classification model
        segmentation_models_dir: Directory with segmentation models
        window_size: Number of frames for model input (default: 30)
        eval_interval: Alle wieviele Frames wird das Modell ausgeführt (default: 10)
    """
    def __init__(self, 
                 stream_url: str = None,
                 classifier_model: str = "models/classification/exercise_classifier.keras",
                 segmentation_models_dir: str = "models/segmentation",
                 window_size: int = 30,
                 eval_interval: int = 10):
        self.stream_url = stream_url
        self.classifier = ExerciseClassifier(classifier_model, window_size)
        self.rep_counter = RepCounter(segmentation_models_dir)
        self.current_exercise = "unknown"
        self.exercise_confidence = 0.0
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.processing_times = deque(maxlen=30)
        self.window_size = window_size
        self.eval_interval = eval_interval
        self.feature_buffer = deque(maxlen=window_size)
        self.last_classification = ("unknown", 0.0)
        print(f"Real-time pipeline initialized (window_size={window_size}, eval_interval={eval_interval})")

    def start(self):
        if self.stream_url:
            cap = cv2.VideoCapture(self.stream_url)
        else:
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"Error: Could not open video stream")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Starting real-time pipeline. Press 'q' to quit, 'r' to reset counts.")
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                frame = cv2.flip(frame, 1)
                # 1. MediaPipe/Feature Extraction auf jedem Frame
                features = FeatureExtractor().extract_angles(frame)
                if features is None:
                    features = np.zeros(25, dtype=np.float32)
                self.feature_buffer.append(features)
                self.frame_count += 1
                # 2. Nur alle eval_interval Frames: Modell-Inferenz
                if len(self.feature_buffer) == self.window_size and self.frame_count % self.eval_interval == 0:
                    window = np.array(self.feature_buffer)
                    # Klassifikation
                    self.current_exercise, self.exercise_confidence = self.classifier.predict_window(window)
                    self.last_classification = (self.current_exercise, self.exercise_confidence)
                    # Rep Counter (optional: kann auch auf jedem Frame laufen, aber hier synchron)
                    self.rep_counter.update_with_window(window, self.current_exercise)
                else:
                    # Zeige das letzte Ergebnis an
                    self.current_exercise, self.exercise_confidence = self.last_classification
                self._update_fps()
                self._draw_overlay(frame)
                cv2.imshow('Real-Time Exercise Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_all_counts()
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.rep_counter.close()

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
                # Show rep detection state for current exercise
                state = self.rep_counter.rep_states.get(exercise, 'rest')
                cv2.putText(frame, f"{exercise}: {count} ({state})", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
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


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Real-time exercise detection and rep counting pipeline'
    )
    parser.add_argument('--stream', type=str, default=None,
                       help='Stream URL (e.g., http://192.168.1.100/stream)')
    parser.add_argument('--classifier', type=str, 
                       default='models/classification/exercise_classifier.keras',
                       help='Path to classification model')
    parser.add_argument('--models-dir', type=str, 
                       default='models/segmentation',
                       help='Directory containing segmentation models')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size for classification')
    
    args = parser.parse_args()
    
    # Create and start pipeline
    pipeline = RealTimePipeline(
        stream_url=args.stream,
        classifier_model=args.classifier,
        segmentation_models_dir=args.models_dir,
        window_size=args.window_size
    )
    
    pipeline.start()


if __name__ == '__main__':
    main() 