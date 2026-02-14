#!/usr/bin/env python3
"""
Quick test of fractal formulas
"""

from renderers.python_3d.fractal_formulas import FormulaRegistry

def test_formulas():
    registry = FormulaRegistry()

    print("Testing fractal formulas...")

    # Test a few formulas
    test_points = [(0.1, 0.1, 0.1), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]

    for formula_name in ['mandelbulb', 'mandelbox', 'julia']:
        formula = registry.get_formula(formula_name)
        if formula:
            params = formula.get_default_parameters()
            print(f"\nTesting {formula_name}:")
            for x, y, z in test_points:
                try:
                    de, trap, iters = formula.distance_estimate(x, y, z, params)
                    print(f"  Point ({x},{y},{z}): DE={de:.4f}, trap={trap:.4f}, iters={iters}")
                except Exception as e:
                    print(f"  Point ({x},{y},{z}): ERROR - {e}")

if __name__ == "__main__":
    test_formulas()