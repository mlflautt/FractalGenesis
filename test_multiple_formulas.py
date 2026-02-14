#!/usr/bin/env python3
"""
Quick test of multiple fractal formulas
"""

from renderers.python_3d.fractal_formulas import FormulaRegistry, FormulaRenderer
import matplotlib.pyplot as plt
import numpy as np
import os

def test_formulas():
    registry = FormulaRegistry()
    renderer = FormulaRenderer(registry)

    # Test key formulas that were previously broken and some others
    key_formulas = ['mandelbulb', 'mandelbox', 'amazing_box', 'sierpinski', 'buffalo', 'celtic', 'julia', 'tricorn']

    output_dir = os.path.join(os.getcwd(), 'output', 'formula_showcase')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for formula_name in key_formulas:
        print(f"Testing {formula_name}...")
        try:
            formula = registry.get_formula(formula_name)
            params = formula.get_default_parameters()

            # Render with smaller size for speed
            image, metrics = renderer.render_formula(formula_name, params, width=400, height=300)

            # Save image
            plt.figure(figsize=(8, 6))
            plt.imshow(image, origin='upper')
            plt.axis('off')
            plt.title(f"{formula_name}")
            filepath = f'{output_dir}/{formula_name}.png'
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='black')
            plt.close()

            print(f"  ✓ {formula_name}: {metrics['render_time']:.2f}s, {metrics['surface_pixels']} pixels, saved to {filepath}")

        except Exception as e:
            print(f"  ✗ {formula_name}: {e}")

if __name__ == "__main__":
    test_formulas()