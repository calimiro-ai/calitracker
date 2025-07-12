#!/usr/bin/env python3
"""
Check dataset and fix class names.
"""

import numpy as np
import os

def check_and_fix_classes():
    """Check the dataset and fix class names."""
    dataset_path = 'data/processed/classification_dataset.npz'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    print(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    print("Available keys:", list(data.keys()))
    
    X = data['X']
    y = data['y']
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Check for class names or label mapping
    if 'class_names' in data:
        class_names = data['class_names'].tolist()
        print(f"Class names from dataset: {class_names}")
    elif 'label_mapping' in data:
        label_mapping = data['label_mapping'].item()
        class_names = list(label_mapping.values())
        print(f"Class names from label_mapping: {class_names}")
    else:
        print("No class names found in dataset")
        # Based on the training output, we have 4 classes
        # Let's assume the order is: push-ups, squats, pull-ups, dips
        class_names = ['push-ups', 'squats', 'pull-ups', 'dips']
        print(f"Using default class names: {class_names}")
    
    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for i, (class_idx, count) in enumerate(zip(unique_classes, counts)):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        print(f"  {class_name} (class {class_idx}): {count} samples")
    
    # Fix the class names file
    class_names_path = 'models/classification/exercise_classifier_classes.txt'
    print(f"\nUpdating class names file: {class_names_path}")
    
    with open(class_names_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Class names updated to: {class_names}")

if __name__ == '__main__':
    check_and_fix_classes() 