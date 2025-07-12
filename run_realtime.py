#!/usr/bin/env python3
"""
SmartMirror - Real-time Exercise Detection

Main entry point for running the real-time exercise detection pipeline.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.realtime_pipeline import RealTimePipeline


def main():
    """Run the real-time pipeline."""
    print("Starting SmartMirror Real-time Exercise Detection...")
    print("Press 'q' to quit, 'r' to reset counts")
    
    # Create and start pipeline with default parameters
    # window_size=30, eval_interval=10
    pipeline = RealTimePipeline()
    pipeline.start()


if __name__ == '__main__':
    main() 