#!/usr/bin/env python3
"""
Frontend Interface Module

This module provides functions for interacting with the frontend interface.
Currently contains placeholder functions that will be implemented by the frontend team.
"""

import json
from typing import Dict


def update_workout_state(total_reps: Dict[str, int], current_exercise: str) -> None:
    """
    Update the frontend with current workout state.
    
    This is a placeholder function that will be implemented by the frontend team.
    It should update the frontend interface with the current workout progress.
    
    Args:
        total_reps: Dictionary mapping exercise names to total rep counts
                   Example: {'push-ups': 5, 'squats': 3, 'pull-ups': 0, 'dips': 2}
        current_exercise: Currently detected exercise (or 'unknown' if none detected)
                         Example: 'push-ups', 'squats', 'pull-ups', 'dips', 'unknown'
    
    Returns:
        None
    """
    # TODO: Implement frontend state update logic
    # This function will be implemented by the frontend team
    # For now, just print the state for debugging purposes
    
    # Create JSON structure for frontend
    workout_state = {
        "current_exercise": current_exercise,
        "total_reps": total_reps,
        "timestamp": None  # Can be added if needed
    }
    
    # Convert to JSON string
    workout_state_json = json.dumps(workout_state, indent=2)
    
    print(f"[FRONTEND] Updating workout state:")
    print(f"JSON payload:")
    print(workout_state_json)