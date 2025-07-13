#!/usr/bin/env python3
"""
Unified script to train exercise segmentation models.

This script loads pre-built datasets and trains segmentation models for exercise repetition detection.
Datasets should be built first using build_datasets.py.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from segmentation.trainer import SegmentationTrainer, create_sequences


def load_segmentation_dataset(dataset_path: str):
    """Load a pre-built segmentation dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = np.load(dataset_path, allow_pickle=True)
    features = data['X']
    labels = data['y']
    
    print(f"Loaded dataset: {dataset_path}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Positive samples: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
    
    return features, labels


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train exercise repetition detection model'
    )
    parser.add_argument('--exercise', type=str, required=True,
                       choices=['push-ups', 'squats', 'pull-ups', 'dips'],
                       help='Exercise type to train')
    parser.add_argument('--dataset', type=str, 
                       help='Path to pre-built dataset (default: data/processed/segmentation_dataset_{exercise}.npz)')
    parser.add_argument('--window', type=int, default=30,
                       help='Window size for sequences (default: 30 for real-time)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--model-dir', type=str, default='models/segmentation',
                       help='Directory to save models (default: models/segmentation)')
    
    args = parser.parse_args()
    
    # Set default dataset path if not provided
    if args.dataset is None:
        args.dataset = f'data/processed/segmentation_dataset_{args.exercise}.npz'
    
    # Load the dataset
    features, labels = load_segmentation_dataset(args.dataset)
    
    # Create sequences
    X, y = create_sequences(features, labels, args.window)
    
    print(f"Created sequences with window size {args.window}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Positive sequences: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Initialize trainer
    model_path = os.path.join(args.model_dir, f"{args.exercise}.keras")
    trainer = SegmentationTrainer(
        model_path=model_path,
        window_size=args.window,
        batch_size=args.batch,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )
    
    # Combine train and validation for the trainer (it handles splitting internally)
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    # Train the model
    trainer.train(X_combined, y_combined)
    
    print(f"Training completed! Model saved as {model_path}")


if __name__ == '__main__':
    main() 