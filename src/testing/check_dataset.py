#!/usr/bin/env python3
import numpy as np

# Load the classification dataset
data = np.load('data/processed/classification_dataset.npz', allow_pickle=True)

print('Dataset Info:')
print(f'X shape: {data["X"].shape}')
print(f'y shape: {data["y"].shape}')
print(f'Label mapping: {data["label_mapping"].item()}')

# Check class distribution
unique, counts = np.unique(data['y'], return_counts=True)
print('\nClass distribution:')
for u, c in zip(unique, counts):
    print(f'  Class {u}: {c} samples ({c/len(data["y"])*100:.1f}%)')

# Check feature statistics
print(f'\nFeature statistics:')
print(f'  Mean: {np.mean(data["X"]):.4f}')
print(f'  Std: {np.std(data["X"]):.4f}')
print(f'  Min: {np.min(data["X"]):.4f}')
print(f'  Max: {np.max(data["X"]):.4f}')

# Check for any NaN or infinite values
print(f'\nData quality:')
print(f'  NaN values: {np.isnan(data["X"]).sum()}')
print(f'  Infinite values: {np.isinf(data["X"]).sum()}') 