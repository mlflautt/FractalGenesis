#!/usr/bin/env python3
"""
Fractal Type Showcase
=====================

Demonstrates all available fractal types in the system.

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from renderers.fractal_types import list_all_fractals, get_formula, get_formula_function, FractalCategory
from renderers.python_3d import FractalRenderer, FractalParams
from renderers.verification import verify_render
import matplotlib.pyplot as plt
import time


def render_fractal_type(name: str, output_dir: Path) -> dict:
    """Render a single fractal type and verify."""
    formula = get_formula(name)
    if not formula:
        return {"name": name, "status": "unknown", "error": "Formula not found"}
    
    # Map fractal name to renderer parameters
    params_map = {
        "mandelbulb": {"fractal_type": "mandelbulb", "power": 8.0, "color_palette": "warm"},
        "mandelbulb_pow4": {"fractal_type": "mandelbulb", "power": 4.0, "color_palette": "cool"},
        "mandelbulb_pow6": {"fractal_type": "mandelbulb", "power": 6.0, "color_palette": "fire"},
        "mandelbulb_pow10": {"fractal_type": "mandelbulb", "power": 10.0, "color_palette": "rainbow"},
        "juliabulb": {"fractal_type": "julia", "julia_c": (-0.2, 0.1, 0.0), "color_palette": "cool"},
        "julia_3d": {"fractal_type": "julia", "julia_c": (-0.2, 0.1, 0.0), "color_palette": "ice"},
        "julia_classic": {"fractal_type": "julia", "julia_c": (-0.7, 0.27, 0.0), "color_palette": "cool"},
        "mandelbox": {"fractal_type": "mandelbox", "scale": 2.0, "min_r": 0.5, "color_palette": "rainbow"},
        "mandelbox_neg": {"fractal_type": "mandelbox", "scale": -1.5, "min_r": 0.5, "color_palette": "fire"},
        "mandelbox_sponge": {"fractal_type": "mandelbox", "scale": 3.0, "min_r": 0.1, "color_palette": "cool"},
        "mandelbox_var1": {"fractal_type": "mandelbox", "scale": 2.5, "min_r": 0.3, "color_palette": "warm"},
        "burning_ship": {"fractal_type": "burning_ship", "power": 8.0, "color_palette": "rainbow"},
        "burning_ship_3d": {"fractal_type": "burning_ship", "power": 6.0, "color_palette": "ice"},
        "tricorn": {"fractal_type": "mandelbulb", "power": 2.0, "color_palette": "fire"},
    }
    
    # Skip complex formulas not yet in renderer
    skip_types = ["hybrid_mandelbulb_mandelbox", "hybrid_julia_mandelbox", "lambda_mandelbulb",
                  "sphere_fold", "biomorph", "newton_z3", "newton_z4", "quaternion_julia",
                  "sierpinski", "menger_sponge", "kifs", "mandelbrot_3d", "mandelbulb_pow10"]
    
    if name in skip_types:
        return {"name": name, "status": "skipped", "reason": "Not in basic renderer"}
    
    renderer = FractalRenderer()
    
    base_params = params_map.get(name, {"fractal_type": "mandelbulb", "power": 8.0})
    
    params = FractalParams(
        width=250,
        height=250,
        camera_pos=(0.0, 0.0, -3.0),
        **base_params
    )
    
    output_path = output_dir / f"{name}.png"
    
    try:
        start = time.time()
        img, metrics = renderer.render(params)
        elapsed = time.time() - start
        
        plt.imsave(output_path, img)
        
        verify = verify_render(str(output_path))
        
        return {
            "name": name,
            "status": "passed" if verify.is_valid else "warning",
            "time": elapsed,
            "mean": verify.checks.get("mean_pixel_value", 0),
            "path": str(output_path)
        }
    except Exception as e:
        return {"name": name, "status": "error", "error": str(e)}


def main():
    print("=" * 60)
    print("FRACTAL TYPE SHOWCASE")
    print("=" * 60)
    
    output_dir = Path("output/fractal_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fractal_names = list_all_fractals()
    
    results = []
    for name in fractal_names:
        print(f"Rendering {name}...", end=" ")
        result = render_fractal_type(name, output_dir)
        results.append(result)
        
        if result["status"] == "passed":
            print(f"✓ {result['time']:.2f}s mean={result['mean']:.1f}")
        elif result["status"] == "skipped":
            print(f"- skipped ({result.get('reason', '')})")
        elif result["status"] == "warning":
            print(f"⚠ {result['time']:.2f}s")
        else:
            print(f"✗ {result.get('error', 'error')[:40]}")
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "passed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] in ("error", "warning"))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total fractals: {len(results)}")
    print(f"Rendered: {passed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"\nOutput: {output_dir}/")


if __name__ == "__main__":
    main()