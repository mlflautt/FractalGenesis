#!/usr/bin/env python3
"""
Generate sample selection data for testing the AI preference learning system.

Creates synthetic fractal selection sessions with varied parameters to provide
enough training data to test the AI training pipeline.
"""

import json
import random
import time
from pathlib import Path
from datetime import datetime


def generate_fractal_params():
    """Generate random fractal parameters for testing."""
    return {
        "scale": random.uniform(0.5, 4.0),
        "rotate": random.uniform(0, 360),
        "center": f"{random.uniform(-2, 2)} {random.uniform(-2, 2)}",
        "brightness": random.uniform(0.5, 2.0),
        "gamma": random.uniform(1.0, 4.0),
        "vibrancy": random.uniform(0.0, 1.5),
        "xforms": [
            {
                "weight": random.uniform(0.1, 1.0),
                "color": random.uniform(0, 1),
                "variations": {
                    random.choice(["linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar", "heart"]): random.uniform(0.1, 1.0)
                }
            }
            for _ in range(random.randint(2, 6))
        ],
        "colors": [
            {"rgb": f"{random.randint(0, 255)} {random.randint(0, 255)} {random.randint(0, 255)}"}
            for _ in range(256)
        ]
    }


def create_selection_session(generation, session_id):
    """Create a selection session with 4 candidates."""
    # Generate 4 fractal candidates
    candidates = [generate_fractal_params() for _ in range(4)]
    
    # Simulate user preference patterns - prefer certain characteristics
    scores = []
    for candidate in candidates:
        score = 0
        # Prefer moderate scale values
        if 1.5 <= candidate["scale"] <= 2.5:
            score += 2
        # Prefer certain brightness ranges
        if 1.0 <= candidate["brightness"] <= 1.5:
            score += 2
        # Prefer certain variations
        for xform in candidate["xforms"]:
            if "sinusoidal" in xform["variations"] or "spherical" in xform["variations"]:
                score += 1
        # Add some randomness
        score += random.uniform(0, 2)
        scores.append(score)
    
    # Select the highest scoring candidate
    selected_index = scores.index(max(scores))
    
    # Create session data
    session_data = {
        "timestamp": time.time(),
        "generation": generation,
        "session_id": session_id,
        "fractal_parameters": candidates,
        "selected_index": selected_index,
        "selection_time": random.uniform(5.0, 30.0),  # Simulated selection time
        "user_notes": "Generated test data"
    }
    
    return session_data


def main():
    """Generate multiple selection sessions for training."""
    data_dir = Path("data/user_selections")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    num_sessions = 15  # Generate 15 sessions for training
    
    print(f"Generating {num_sessions} test selection sessions...")
    
    for i in range(num_sessions):
        generation = random.randint(1, 5)
        session_id = f"test_session_{i+1:03d}"
        
        session_data = create_selection_session(generation, session_id)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selection_test_gen{generation:02d}_{timestamp}_{i:03d}.json"
        file_path = data_dir / filename
        
        # Save session data
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Created session {i+1}/{num_sessions}: {filename} (Gen {generation}, Selected #{session_data['selected_index']})")
        
        # Small delay to ensure unique timestamps
        time.sleep(0.1)
    
    print(f"\nGenerated {num_sessions} test selection sessions in {data_dir}")
    print("You can now run 'python3 manage_ai.py train' to create an AI selector!")


if __name__ == "__main__":
    main()
