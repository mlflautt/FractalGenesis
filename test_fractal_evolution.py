#!/usr/bin/env python3
"""
Comprehensive Fractal Evolution Testing
======================================

Tests both 3D and 2D fractal evolution with diversity verification.
Shows detailed results and catches "same silhouette" issues.

Usage:
    python3 test_fractal_evolution.py --test-3d
    python3 test_fractal_evolution.py --test-2d
    python3 test_fractal_evolution.py --test-all
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from verification.diversity_engine import DiversityEngine


def test_3d_evolution():
    """Test 3D fractal evolution with verification"""
    print("\n" + "="*70)
    print("TESTING 3D FRACTAL EVOLUTION")
    print("="*70 + "\n")
    
    # Run evolution
    from fractal_evolution_3d import FractalEvolution3D
    from renderers.unified import RendererType
    
    evolution = FractalEvolution3D(
        renderer_type=RendererType.PYTHON_3D,
        output_dir="output/test_3d",
        population_size=6,
        mutation_rate=0.3
    )
    
    results = evolution.run_evolution(generations=3)
    
    # Verify diversity
    print("\n" + "="*70)
    print("DIVERSITY VERIFICATION")
    print("="*70)
    
    all_images = []
    for result in results:
        all_images.extend(result.image_paths)
    
    if len(all_images) >= 4:
        engine = DiversityEngine()
        verify_result = engine.verify_batch_diversity(
            all_images,
            min_diversity_threshold=0.25,
            max_similar_pairs_ratio=0.3
        )
        
        print(engine.generate_report(verify_result))
        
        return verify_result['passed'], verify_result['statistics']['mean_diversity']
    else:
        print("⚠️  Not enough images for diversity verification")
        return False, 0.0


def test_2d_evolution():
    """Test 2D flame evolution with verification"""
    print("\n" + "="*70)
    print("TESTING 2D FRACTAL FLAME EVOLUTION")
    print("="*70 + "\n")
    
    try:
        # Run evolution
        from fractal_evolution_2d import FractalEvolution2D
        
        evolution = FractalEvolution2D(
            output_dir="output/test_2d",
            population_size=6,
            mutation_rate=0.3,
            quality=50  # Lower quality for speed
        )
        
        results = evolution.run_evolution(generations=2)
        
        # Verify diversity
        print("\n" + "="*70)
        print("DIVERSITY VERIFICATION")
        print("="*70)
        
        all_images = []
        for result in results:
            all_images.extend(result.image_paths)
        
        if len(all_images) >= 4:
            engine = DiversityEngine()
            verify_result = engine.verify_batch_diversity(
                all_images,
                min_diversity_threshold=0.25,
                max_similar_pairs_ratio=0.3
            )
            
            print(engine.generate_report(verify_result))
            
            return verify_result['passed'], verify_result['statistics']['mean_diversity']
        else:
            print("⚠️  Not enough images for diversity verification")
            return False, 0.0
            
    except Exception as e:
        print(f"\n✗ 2D evolution failed: {e}")
        print("  Note: Requires flam3 to be installed")
        return False, 0.0


def test_renderers():
    """Test all available renderers"""
    print("\n" + "="*70)
    print("TESTING ALL RENDERERS")
    print("="*70 + "\n")
    
    from renderers.unified import get_available_renderers
    
    available = get_available_renderers()
    
    print("Available renderers:")
    for rt, name, is_available in available:
        status = "✓" if is_available else "✗"
        print(f"  {status} {name}")
    
    working = sum(1 for _, _, avail in available if avail)
    total = len(available)
    
    print(f"\nResult: {working}/{total} renderers available")
    
    return working >= 2  # Need at least 2 for good testing


def main():
    parser = argparse.ArgumentParser(description='Test Fractal Evolution Systems')
    parser.add_argument('--test-3d', action='store_true', help='Test 3D evolution')
    parser.add_argument('--test-2d', action='store_true', help='Test 2D evolution')
    parser.add_argument('--test-all', action='store_true', help='Test everything')
    
    args = parser.parse_args()
    
    results = {}
    
    # Test renderers
    if args.test_all:
        results['renderers'] = test_renderers()
    
    # Test 3D
    if args.test_3d or args.test_all:
        passed_3d, diversity_3d = test_3d_evolution()
        results['3d_evolution'] = passed_3d
        results['3d_diversity'] = diversity_3d
    
    # Test 2D
    if args.test_2d or args.test_all:
        passed_2d, diversity_2d = test_2d_evolution()
        results['2d_evolution'] = passed_2d
        results['2d_diversity'] = diversity_2d
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for key, value in results.items():
        if isinstance(value, bool):
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"  {status}: {key}")
        else:
            print(f"  📊 {key}: {value:.3f}")
    
    # Overall result
    all_passed = all(v for k, v in results.items() if isinstance(v, bool))
    
    if all_passed:
        print("\n✅ All tests passed!")
        print("   2D and 3D fractal evolution systems are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check logs above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
