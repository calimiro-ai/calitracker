#!/usr/bin/env python3
import numpy as np
import os

path = 'data/processed/classification_dataset.npz'
if not os.path.exists(path):
    print(f"Dataset file not found: {path}")
    exit(1)

data = np.load(path, allow_pickle=True)
X = data['X']
y = data['y']

if 'class_names' in data:
    class_names = data['class_names'].tolist()
elif 'label_mapping' in data:
    label_mapping = data['label_mapping'].item()
    class_names = list(label_mapping.values())
else:
    class_names = ['push-ups', 'squats', 'pull-ups', 'dips']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class names: {class_names}")

unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for idx, count in zip(unique, counts):
    print(f"  {idx}: {class_names[idx] if idx < len(class_names) else '?'} - {count} samples")

print("\nFirst 10 labels:")
for i in range(10):
    print(f"  y[{i}] = {y[i]} ({class_names[y[i]] if y[i] < len(class_names) else '?'})") 