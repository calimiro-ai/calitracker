#!/usr/bin/env python3
"""
Standalone script to train the exercise classification model with the new dataset.
"""

import numpy as np
import os
import datetime
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from classification.model import build_exercise_classification_model


def load_classification_dataset(path: str = '../../data/processed/classification_dataset.npz'):
    """Load classification dataset with features and exercise type labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    print(f"Loading dataset from: {path}")
    data = np.load(path, allow_pickle=True)
    X = data['X']  # (num_samples, sequence_length, num_features)
    y = data['y']  # (num_samples,) - class indices
    
    # Handle class_names
    if 'class_names' in data:
        class_names = data['class_names'].tolist()
    elif 'label_mapping' in data:
        label_mapping = data['label_mapping'].item()
        class_names = list(label_mapping.values())
    else:
        num_classes = len(np.unique(y))
        default_names = ['push-ups', 'squats', 'pull-ups', 'dips']
        class_names = default_names[:num_classes]
    
    print(f"Loaded classification dataset: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class names: {class_names}")
    
    for i, class_name in enumerate(class_names):
        count = np.sum(y == i)
        print(f"  {class_name}: {count} samples")
    
    return X, y, class_names


def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Prepare data for classification training with train/validation/test splits."""
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, class_names, 
                batch_size=16, epochs=50, learning_rate=1e-3, patience=15):
    """Train the classification model."""
    # Determine input dimension and number of classes from data
    input_dim = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    print(f"Building classification model with input_dim={input_dim}, num_classes={num_classes}")
    
    # Build the model
    model = build_exercise_classification_model(
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    
    # Create models directory if it doesn't exist
    os.makedirs('models/classification', exist_ok=True)
    
    # Configure callbacks
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = os.path.join('../../logs', f'classifier_{timestamp}')
    model_path = '../../models/classification/exercise_classifier.keras'
    
    callbacks = [
        TensorBoard(log_dir=logdir, histogram_freq=1),
        EarlyStopping(
            monitor='val_accuracy', 
            patience=patience, 
            restore_best_weights=True,
            mode='max'
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    print(f"Training logs will be saved to: {logdir}")
    print(f"Model will be saved to: {model_path}")
    
    # Train the model
    print(f"Starting training with {X_train.shape[0]} training samples...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    # Save class names with the model
    class_names_path = model_path.replace('.keras', '_classes.txt')
    with open(class_names_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"Class names saved to: {class_names_path}")
    
    print(f"Training completed successfully!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, history


def main():
    """Main training function."""
    print("Starting exercise classification model training...")
    
    try:
        # 1. Load classification dataset
        print("Loading classification dataset...")
        X, y, class_names = load_classification_dataset()
        
        # 2. Prepare data splits
        print("Preparing data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)
        
        # 3. Train model
        print("Training classification model...")
        model, history = train_model(
            X_train, y_train, X_val, y_val, class_names,
            batch_size=16, epochs=50, learning_rate=1e-3, patience=15
        )
        
        # 4. Evaluate on test set
        print("Evaluating on test set...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        if len(test_results) >= 2:
            test_loss, test_accuracy = test_results[0], test_results[1]
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test loss: {test_loss:.4f}")
        else:
            print(f"Test results: {test_results}")
        
        print("Training completed successfully!")
        print("Model saved to: models/classification/exercise_classifier.keras")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 