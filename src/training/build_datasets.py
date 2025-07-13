#!/usr/bin/env python3
"""
Unified dataset builder for both classification and segmentation models.

This script builds datasets for:
1. Classification: Multi-class exercise type classification (push-ups, squats, pull-ups, dips)
2. Segmentation: Per-frame exercise repetition detection for each exercise type
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.dataset_builder import ClassificationDatasetBuilder, SegmentationDatasetBuilder


def build_classification_dataset(output_path: str = 'data/processed/classification_dataset.npz'):
    """Build the classification dataset for exercise type classification."""
    print("Building classification dataset...")
    
    # Initialize the dataset builder
    builder = ClassificationDatasetBuilder()
    
    # Build the dataset
    features, labels, label_mapping, class_names = builder.build()
    
    # Save the dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    builder.save(features, labels, label_mapping, class_names, output_path)
    
    print(f"Classification dataset saved to {output_path}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes: {class_names}")


def build_segmentation_dataset(exercise: str, output_path: str | None = None):
    """Build segmentation dataset for a specific exercise."""
    if output_path is None:
        output_path = f'data/processed/segmentation_dataset_{exercise}.npz'
    
    print(f"Building segmentation dataset for {exercise}...")
    
    try:
        # Initialize the segmentation dataset builder
        builder = SegmentationDatasetBuilder()
        
        # Build the dataset for this exercise
        print(f"Starting dataset building for {exercise}...")
        features, labels = builder.build(exercise_type=exercise)
        
        # Save the dataset
        print(f"Saving dataset to {output_path}...")
        builder.save(features, labels, output_path)
        
        print(f"Segmentation dataset for {exercise} saved to {output_path}")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return output_path
        
    except Exception as e:
        print(f"Error building dataset for {exercise}: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_all_segmentation_datasets():
    """Build segmentation datasets for all exercises."""
    exercises = ['push-ups', 'squats', 'pull-ups', 'dips']
    
    for exercise in exercises:
        try:
            build_segmentation_dataset(exercise)
        except Exception as e:
            print(f"Error building dataset for {exercise}: {e}")


def main():
    """Main function to build datasets."""
    parser = argparse.ArgumentParser(description='Build datasets for training')
    parser.add_argument('--mode', choices=['classification', 'segmentation', 'all'], 
                       default='all', help='Type of dataset to build')
    parser.add_argument('--exercise', type=str, 
                       choices=['push-ups', 'squats', 'pull-ups', 'dips'],
                       help='Exercise for segmentation dataset (required if mode=segmentation)')
    parser.add_argument('--output', type=str, help='Output path for the dataset')
    
    args = parser.parse_args()
    
    if args.mode == 'classification':
        output_path = args.output or 'data/processed/classification_dataset.npz'
        build_classification_dataset(output_path)
    
    elif args.mode == 'segmentation':
        if not args.exercise:
            parser.error("--exercise is required for segmentation mode")
        output_path = args.output or f'data/processed/segmentation_dataset_{args.exercise}.npz'
        build_segmentation_dataset(args.exercise, output_path)
    
    elif args.mode == 'all':
        # Build classification dataset
        build_classification_dataset()
        
        # Build segmentation datasets for all exercises
        build_all_segmentation_datasets()


if __name__ == '__main__':
    main() 