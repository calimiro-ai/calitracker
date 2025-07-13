#!/usr/bin/env python3
"""
Unified script to train the exercise classification model.

This script uses the ClassificationTrainer from classification/trainer.py
for consistent training across the project.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classification.trainer import ClassificationTrainer, load_classification_dataset, prepare_classification_data


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train exercise type classification model'
    )
    parser.add_argument('--dataset', type=str, default='data/processed/classification_dataset.npz',
                        help='Path to classification dataset file')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--model-dir', type=str, default='models/classification',
                        help='Directory to save the model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Fraction of remaining data for validation')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = os.path.join(args.model_dir, "exercise_classifier.keras")
    
    try:
        # 1. Load classification dataset
        print(f"Loading classification dataset: {args.dataset}")
        X, y, class_names = load_classification_dataset(args.dataset)
        
        # 2. Prepare data splits
        print(f"Preparing data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_classification_data(
            X, y, test_size=args.test_size, val_size=args.val_size
        )
        
        # 3. Train model
        print(f"Training classification model...")
        trainer = ClassificationTrainer(
            model_path=model_path,
            batch_size=args.batch,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            class_names=class_names
        )
        model = trainer.train(X_train, y_train, X_val, y_val)
        
        # 4. Evaluate on test set
        print(f"Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 