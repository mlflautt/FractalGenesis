#!/usr/bin/env python3
"""
Fractal Variety Demo
====================

Demonstrates different fractal types available in the Python 3D renderer.

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from renderers.python_3d import FractalRenderer, FractalParams
from renderers.verification import verify_render
import time

def main():
    renderer = FractalRenderer()
    
    test_cases = [
        ("mandelbulb", {"fractal_type": "mandelbulb", "power": 8.0, "color_palette": "warm"}),
        ("mandelbulb_power4", {"fractal_type": "mandelbulb", "power": 4.0, "color_palette": "cool"}),
        ("mandelbulb_power12", {"fractal_type": "mandelbulb", "power": 12.0, "color_palette": "fire"}),
        ("julia", {"fractal_type": "julia", "julia_c": (-0.2, 0.1, 0.0), "color_palette": "cool"}),
        ("mandelbox", {"fractal_type": "mandelbox", "scale": 2.0, "min_r": 0.5, "color_palette": "rainbow"}),
        ("mandelbox_neg", {"fractal_type": "mandelbox", "scale": -1.5, "min_r": 0.5, "color_palette": "fire"}),
        ("burning_ship", {"fractal_type": "burning_ship", "power": 8.0, "color_palette": "ice"}),
    ]
    
    output_dir = Path("output/fractal_variety")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Fractal Variety Demo")
    print("=" * 60)
    
    results = []
    for name, overrides in test_cases:
        params = FractalParams(
            width=300,
            height=300,
            camera_pos=(0.0, 0.0, -3.0),
            **overrides
        )
        
        output_path = output_dir / f"{name}.png"
        print(f"\nRendering {name}...")
        
        start = time.time()
        img, metrics = renderer.render(params)
        
        import matplotlib.pyplot as plt
        plt.imsave(output_path, img)
        
        elapsed = time.time() - start
        
        # Verify
        verify = verify_render(str(output_path))
        
        results.append({
            "name": name,
            "time": elapsed,
            "status": verify.status,
            "mean": verify.checks.get("mean_pixel_value", 0),
            "diversity": "OK" if verify.is_valid else "FAIL"
        })
        
        print(f"  ✓ {name}: {elapsed:.2f}s - mean={verify.checks.get('mean_pixel_value', 0):.1f}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        status_icon = "✓" if r["status"] == "passed" else "⚠" if r["status"] == "warning" else "✗"
        print(f"{status_icon} {r['name']}: {r['time']:.2f}s, mean={r['mean']:.1f}")

if __name__ == "__main__":
    main()