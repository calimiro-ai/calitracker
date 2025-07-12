# Real-Time Exercise Detection & Rep Counting Pipeline

This guide explains how to use the real-time pipeline for live exercise detection and rep counting.

## Overview

The pipeline combines:
1. **Exercise Classification** - Identifies the current exercise type (push-ups, squats, pull-ups, dips)
2. **Rep Counting** - Counts repetitions using exercise-specific segmentation models
3. **Real-time Display** - Shows results with minimal latency

## Prerequisites

1. **Trained Models**:
   - Classification model: `models/classification/exercise_classifier.keras`
   - Segmentation models: `models/segmentation/{exercise}.keras` for each exercise type

2. **Video Stream**:
   - ESP32-CAM stream URL (e.g., `http://192.168.1.100/stream`)
   - Webcam (default camera)
   - Video file for testing

## Usage

### 1. Test with Video File

Before using with live streams, test the pipeline with a video file:

```bash
# Test with a video file
python3 test_realtime.py --input test_videos/test1.mp4

# Test and save output video
python3 test_realtime.py --input test_videos/test1.mp4 --output test_output.mp4
```

### 2. Live Stream with ESP32-CAM

```bash
# Connect to ESP32-CAM stream
python3 realtime_pipeline.py --stream http://192.168.1.100/stream

# With custom model paths
python3 realtime_pipeline.py \
    --stream http://192.168.1.100/stream \
    --classifier models/classification/exercise_classifier.keras \
    --models-dir models/segmentation \
    --window-size 30
```

### 3. Webcam (Default Camera)

```bash
# Use default webcam
python3 realtime_pipeline.py
```

## Controls

- **'q'** - Quit the application
- **'r'** - Reset all rep counts

## Performance Optimization

### Latency Reduction

1. **Reduce Window Size**: Use smaller classification windows (e.g., 15-20 frames)
2. **Frame Skipping**: Process every 2nd or 3rd frame
3. **Model Optimization**: Use TensorFlow Lite models for faster inference
4. **Resolution**: Lower video resolution for faster processing

### Example Optimized Settings

```bash
# Fast mode with smaller window
python3 realtime_pipeline.py --window-size 15

# With custom stream settings
python3 realtime_pipeline.py \
    --stream http://192.168.1.100/stream \
    --window-size 20
```

## Architecture Details

### Exercise Classifier
- Uses 30-frame sliding window (configurable)
- TCN architecture for temporal modeling
- Outputs exercise type and confidence

### Rep Counter
- Exercise-specific segmentation models
- Peak detection for rep counting
- Maintains separate counts for each exercise

### Real-time Pipeline
- Frame-by-frame processing
- Overlay display with exercise info and rep counts
- Performance monitoring (FPS, latency)

## Troubleshooting

### Common Issues

1. **High Latency**:
   - Reduce window size
   - Lower video resolution
   - Use more powerful hardware

2. **Poor Classification**:
   - Ensure good lighting
   - Position camera properly
   - Check model training quality

3. **Incorrect Rep Counting**:
   - Adjust peak detection thresholds
   - Check segmentation model quality
   - Ensure proper exercise form

### Debug Mode

Add debug prints to see detailed information:

```python
# In realtime_pipeline.py, add debug prints
print(f"Exercise: {self.current_exercise}, Confidence: {self.exercise_confidence}")
print(f"Rep counts: {self.rep_counter.rep_counts}")
```

## Integration with Smart Mirror

The pipeline can be integrated into your smart mirror system:

```python
from realtime_pipeline import RealTimePipeline

# Initialize pipeline
pipeline = RealTimePipeline(
    stream_url="http://your-esp32-cam/stream",
    classifier_model="path/to/classifier.keras",
    segmentation_models_dir="path/to/segmentation/models"
)

# Start processing
pipeline.start()
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | FPS | Latency | Notes |
|----------|-----|---------|-------|
| CPU (Intel i5) | 15-20 | 50-70ms | Good for testing |
| GPU (GTX 1060) | 25-30 | 30-40ms | Recommended |
| Edge Device | 10-15 | 70-100ms | Mobile/embedded |

## Next Steps

1. **Model Optimization**: Convert to TensorFlow Lite for faster inference
2. **Multi-threading**: Process classification and rep counting in parallel
3. **Cloud Integration**: Send data to cloud for analytics
4. **Mobile App**: Create companion app for settings and history 