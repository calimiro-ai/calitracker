# Training Scripts

This directory contains the unified training scripts for the WorkoutTracker project. The structure has been cleaned up to separate dataset building from model training.

## Scripts Overview

### 1. Dataset Building
- **`build_datasets.py`** - Unified dataset builder for both classification and segmentation models

### 2. Model Training
- **`train_classifier.py`** - Train the exercise type classification model
- **`train_segmentation.py`** - Train exercise repetition detection models

## Usage

### Step 1: Build Datasets

First, build the datasets using the unified dataset builder:

```bash
# Build all datasets (classification + all segmentation datasets)
python3 src/training/build_datasets.py --mode all

# Or build specific datasets
python3 src/training/build_datasets.py --mode classification
python3 src/training/build_datasets.py --mode segmentation --exercise push-ups
```

### Step 2: Train Models

#### Classification Model
```bash
# Train the exercise type classifier
python3 src/training/train_classifier.py

# With custom parameters
python3 src/training/train_classification.py \
    --dataset data/processed/classification_dataset.npz \
    --batch 32 \
    --epochs 50 \
    --lr 1e-3 \
    --patience 15
```

#### Segmentation Models
```bash
# Train push-up repetition detection
python3 src/training/train_segmentation.py --exercise push-ups

# Train with custom parameters
python3 src/training/train_segmentation.py \
    --exercise push-ups \
    --dataset data/processed/segmentation_dataset_push-ups.npz \
    --window 30 \
    --batch 16 \
    --epochs 100 \
    --lr 1e-3 \
    --patience 10

# Train other exercises
python3 src/training/train_segmentation.py --exercise squats
python3 src/training/train_segmentation.py --exercise pull-ups
python3 src/training/train_segmentation.py --exercise dips
```

## Output Structure

### Datasets
- `data/processed/classification_dataset.npz` - Classification dataset
- `data/processed/segmentation_dataset_{exercise}.npz` - Segmentation datasets

### Models
- `models/classification/exercise_classifier.keras` - Classification model
- `models/classification/exercise_classifier_classes.txt` - Class names
- `models/segmentation/{exercise}.keras` - Segmentation models

### Logs
- `logs/classifier_{timestamp}/` - Classification training logs
- `logs/segmenter_{timestamp}/` - Segmentation training logs

## Key Improvements

1. **Centralized Dataset Building**: All dataset creation is now in `build_datasets.py`
2. **Single Exercise Training**: Segmentation models are trained one exercise at a time
3. **Consistent Interfaces**: All scripts use similar argument patterns
4. **Clean Separation**: Dataset building and model training are separate concerns
5. **Preserved Functionality**: All original training capabilities are maintained

## Migration from Old Scripts

The old scripts (`retrain_segmentation_realtime.py`, `train_classifier.py` in root) have been replaced with these unified versions. The new scripts produce the same results but with a cleaner, more maintainable structure. 