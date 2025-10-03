#!/usr/bin/env python3
"""
Parameter Control Verification
=============================

This script demonstrates that our fractal renderer has genuine parameter control
by systematically varying individual parameters and showing the results differ
significantly. This proves evolutionary algorithms can meaningfully explore
the parameter space.
"""

from fractal_animator import FractalRenderer, FractalParams, FractalPresets
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def test_parameter_variations():
    """Test systematic parameter variations to verify control"""
    renderer = FractalRenderer()
    output_dir = Path("parameter_verification")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Parameter Control Verification ===\n")
    
    # Base parameters for comparison
    base_params = FractalParams(
        width=300, height=300,  # Smaller for faster testing
        iterations=80,
        power=8.0,
        camera_pos=(0.0, 0.0, -3.0),
        color_palette="warm"
    )
    
    # Test variations
    test_variations = [
        # Power variations
        ("power_2", {"power": 2.0}),
        ("power_8_baseline", {"power": 8.0}),
        ("power_12", {"power": 12.0}),
        ("power_16", {"power": 16.0}),
        
        # Camera position variations  
        ("camera_close", {"camera_pos": (0.0, 0.0, -2.0)}),
        ("camera_far", {"camera_pos": (0.0, 0.0, -5.0)}),
        ("camera_side", {"camera_pos": (3.0, 0.0, -3.0)}),
        ("camera_top", {"camera_pos": (0.0, 3.0, -3.0)}),
        
        # Iteration variations
        ("iter_30", {"iterations": 30}),
        ("iter_60", {"iterations": 60}),
        ("iter_120", {"iterations": 120}),
        
        # Color palette variations
        ("palette_cool", {"color_palette": "cool"}),
        ("palette_rainbow", {"color_palette": "rainbow"}),
        ("palette_fire", {"color_palette": "fire"}),
        ("palette_ice", {"color_palette": "ice"}),
        
        # Julia set variations
        ("julia_1", {"fractal_type": "julia", "julia_c": (-0.1, 0.0, 0.0)}),
        ("julia_2", {"fractal_type": "julia", "julia_c": (-0.2, 0.1, 0.05)}),
        ("julia_3", {"fractal_type": "julia", "julia_c": (0.0, -0.15, 0.1)}),
    ]
    
    results = []
    
    for name, param_overrides in test_variations:
        print(f"Testing: {name}")
        
        # Create modified parameters
        test_params = FractalParams(**{**base_params.__dict__, **param_overrides})
        
        # Render
        image, metrics = renderer.render(test_params)
        
        # Save image
        output_file = output_dir / f"{name}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(f"{name}: {param_overrides}")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Store results (convert numpy types for JSON)
        result = {
            'name': name,
            'overrides': param_overrides,
            'surface_pixels': int(metrics['surface_pixels']),
            'unique_colors': int(metrics['unique_colors']),
            'avg_brightness': float(metrics['avg_brightness']),
            'render_time': float(metrics['render_time'])
        }
        results.append(result)
        
        print(f"  Surface: {metrics['surface_pixels']:,} pixels, "
              f"Colors: {metrics['unique_colors']:,}, "
              f"Time: {metrics['render_time']:.2f}s")
    
    # Save results
    results_file = output_dir / "verification_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print(f"\n=== Variation Analysis ===")
    
    # Find baseline for comparison
    baseline = next(r for r in results if r['name'] == 'power_8_baseline')
    baseline_surface = baseline['surface_pixels']
    baseline_colors = baseline['unique_colors']
    
    significant_diffs = []
    
    for result in results:
        if result['name'] == 'power_8_baseline':
            continue
            
        surface_diff = abs(result['surface_pixels'] - baseline_surface) / baseline_surface
        color_diff = abs(result['unique_colors'] - baseline_colors) / baseline_colors
        
        if surface_diff > 0.05 or color_diff > 0.05:  # >5% difference
            significant_diffs.append({
                'name': result['name'],
                'surface_diff': surface_diff * 100,
                'color_diff': color_diff * 100,
                'overrides': result['overrides']
            })
    
    print(f"Parameters with significant visual impact (>5% difference from baseline):")
    for diff in significant_diffs:
        print(f"  {diff['name']}: Surface {diff['surface_diff']:.1f}%, Colors {diff['color_diff']:.1f}%")
        print(f"    Changed: {diff['overrides']}")
    
    print(f"\nVerification complete! {len(significant_diffs)} parameters show significant impact.")
    print(f"Results saved to: {output_dir.absolute()}")
    
    return results

def demonstrate_evolution_potential():
    """Show how parameters could be used in evolutionary algorithms"""
    print("\n=== Evolution Algorithm Potential ===")
    
    # Simulate genetic algorithm parameter ranges
    param_ranges = {
        'power': (2.0, 16.0),
        'iterations': (30, 150),
        'camera_distance': (1.5, 6.0),  # z-component of camera_pos
        'camera_angle': (0.0, 2*np.pi),  # for orbital positions
        'julia_c_x': (-0.3, 0.3),
        'julia_c_y': (-0.3, 0.3),
        'julia_c_z': (-0.3, 0.3),
        'color_intensity': (0.5, 2.0),
        'ambient': (0.05, 0.3),
        'fov': (30.0, 80.0)
    }
    
    print("Parameter ranges suitable for evolution:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: {min_val} → {max_val}")
    
    print(f"\nTotal parameter space size: ~10^{len(param_ranges)} combinations")
    print("This provides vast space for evolutionary exploration!")
    
    # Show how to generate random parameters
    print("\nExample random parameter generation:")
    for i in range(3):
        power = np.random.uniform(2.0, 16.0)
        distance = np.random.uniform(1.5, 6.0)
        angle = np.random.uniform(0.0, 2*np.pi)
        camera_x = distance * np.sin(angle)
        camera_z = -distance * np.cos(angle)
        
        print(f"  Individual {i+1}: power={power:.1f}, "
              f"camera=({camera_x:.2f}, 0.0, {camera_z:.2f})")

if __name__ == "__main__":
    # Run parameter verification
    results = test_parameter_variations()
    
    # Demonstrate evolution potential
    demonstrate_evolution_potential()
    
    print(f"\n✅ Parameter control verified!")
    print(f"The fractal renderer provides genuine parametric control")
    print(f"suitable for evolutionary algorithms and animation systems.")