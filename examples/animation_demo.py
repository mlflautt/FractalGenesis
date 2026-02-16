#!/usr/bin/env python3
"""
Quick Animation Demo
====================

Test the animation system with a power morph animation.

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from animation import FractalAnimator

def main():
    print("=" * 50)
    print("Fractal Animation Demo")
    print("=" * 50)
    
    animator = FractalAnimator()
    
    # Test 1: Power morph animation
    print("\n1. Creating power morph animation (4 -> 12)...")
    start_params = {
        "fractal_type": "mandelbulb",
        "power": 4.0,
        "camera_pos": (0.0, 0.0, -3.0),
        "color_palette": "warm"
    }
    end_params = {
        "fractal_type": "mandelbulb", 
        "power": 12.0,
        "camera_pos": (0.0, 0.0, -3.0),
        "color_palette": "fire"
    }
    
    frames = animator.animate(
        start_params, end_params,
        num_frames=10,  # Keep small for quick test
        output_dir="output/animation_test",
        name="power_morph",
        width=300,
        height=300
    )
    
    # Create GIF
    gif_path = "output/animation_test/power_morph.gif"
    animator.create_gif(frames, gif_path, fps=5)
    
    print(f"\n✓ Animation complete!")
    print(f"  Frames: {len(frames)}")
    print(f"  GIF: {gif_path}")

if __name__ == "__main__":
    main()