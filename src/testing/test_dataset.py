#!/usr/bin/env python3
"""
Test script to verify the classification dataset format.
"""

import numpy as np
import os

def test_dataset():
    """Test the classification dataset."""
    dataset_path = 'data/processed/classification_dataset.npz'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return False
    
    print(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    print("Available keys:", list(data.keys()))
    
    X = data['X']
    y = data['y']
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")
    
    # Check for class names
    if 'class_names' in data:
        class_names = data['class_names'].tolist()
        print(f"Class names: {class_names}")
    elif 'label_mapping' in data:
        label_mapping = data['label_mapping'].item()
        class_names = list(label_mapping.values())
        print(f"Class names from label_mapping: {class_names}")
    else:
        print("No class names found, using defaults")
        class_names = ['push-ups', 'squats', 'pull-ups', 'dips']
    
    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for i, (class_idx, count) in enumerate(zip(unique_classes, counts)):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        print(f"  {class_name}: {count} samples")
    
    # Check data ranges
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}")
    print(f"y range: {y.min()} to {y.max()}")
    
    # Check for NaN or inf values
    if np.isnan(X).any():
        print("WARNING: NaN values found in X!")
    else:
        print("No NaN values in X")
    
    if np.isinf(X).any():
        print("WARNING: Inf values found in X!")
    else:
        print("No Inf values in X")
    
    print("Dataset test completed successfully!")
    return True

if __name__ == '__main__':
    test_dataset() 