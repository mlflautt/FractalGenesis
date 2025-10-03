#!/usr/bin/env python3
"""
Python 3D Fractal Rendering Comparison Suite
=============================================

Comprehensive testing and comparison of different Python approaches for 3D fractal rendering:
1. NumPy + Matplotlib (baseline)
2. Taichi GPU acceleration
3. Numba JIT compilation
4. Additional methods from research

This suite evaluates performance, quality, and usability of each approach.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import our test modules
from numpy_mandelbulb import test_numpy_mandelbulb
from taichi_mandelbulb import test_taichi_mandelbulb
from numba_mandelbulb import test_numba_mandelbulb

def install_missing_packages():
    """Check for and install missing packages"""
    import subprocess
    
    packages_to_try = [
        ("taichi", "Taichi GPU acceleration"),
        ("numba", "Numba JIT compilation"), 
        ("vispy", "Vispy scientific visualization"),
        ("moderngl", "ModernGL real-time rendering")
    ]
    
    available_packages = []
    missing_packages = []
    
    for package, description in packages_to_try:
        try:
            __import__(package)
            available_packages.append((package, description))
            print(f"✅ {package}: {description}")
        except ImportError:
            missing_packages.append((package, description))
            print(f"❌ {package}: {description} (not installed)")
    
    if missing_packages:
        print(f"\n📦 To install missing packages, run:")
        for package, _ in missing_packages:
            print(f"    pip install {package}")
    
    return available_packages, missing_packages

def analyze_results(all_results):
    """Analyze and compare results from different rendering methods"""
    if not all_results:
        print("No results to analyze")
        return
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Performance analysis
    methods = {}
    for method_name, results in all_results.items():
        if results:
            render_times = [r['render_time'] for r in results]
            surface_pixels = [r['surface_pixels'] for r in results]
            unique_colors = [r['unique_colors'] for r in results]
            file_sizes = [r['file_size'] for r in results]
            
            methods[method_name] = {
                'count': len(results),
                'avg_render_time': np.mean(render_times),
                'min_render_time': np.min(render_times),
                'max_render_time': np.max(render_times),
                'avg_surface_pixels': np.mean(surface_pixels),
                'avg_unique_colors': np.mean(unique_colors),
                'avg_file_size': np.mean(file_sizes),
                'quality_score': np.mean([r.get('contrast', 0) for r in results])
            }
    
    if not methods:
        print("No valid results to analyze")
        return
    
    # Performance ranking
    print("\n🏁 PERFORMANCE RANKING (by speed)")
    sorted_by_speed = sorted(methods.items(), key=lambda x: x[1]['avg_render_time'])
    for i, (method, stats) in enumerate(sorted_by_speed, 1):
        print(f"{i}. {method:15} - {stats['avg_render_time']:.2f}s avg "
              f"({stats['min_render_time']:.2f}s - {stats['max_render_time']:.2f}s)")
    
    # Quality analysis
    print("\n🎨 QUALITY ANALYSIS")
    for method, stats in methods.items():
        print(f"{method:15} - {stats['avg_surface_pixels']:8.0f} surface pixels, "
              f"{stats['avg_unique_colors']:8.0f} colors, "
              f"{stats['avg_file_size']/1024:6.1f}KB avg")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    fastest = sorted_by_speed[0]
    print(f"Fastest method: {fastest[0]} ({fastest[1]['avg_render_time']:.2f}s avg)")
    
    # Best quality (most surface pixels and colors)
    quality_ranking = sorted(methods.items(), 
                           key=lambda x: x[1]['avg_surface_pixels'] + x[1]['avg_unique_colors'], 
                           reverse=True)
    best_quality = quality_ranking[0]
    print(f"Best quality: {best_quality[0]} ({best_quality[1]['avg_surface_pixels']:.0f} pixels, "
          f"{best_quality[1]['avg_unique_colors']:.0f} colors)")
    
    # Overall recommendation
    print(f"\n🎯 OVERALL RECOMMENDATION:")
    if len(methods) > 1:
        # Balance of speed and quality
        combined_scores = {}
        max_speed = max(m['avg_render_time'] for m in methods.values())
        max_quality = max(m['avg_surface_pixels'] + m['avg_unique_colors'] 
                         for m in methods.values())
        
        for method, stats in methods.items():
            # Normalize scores (lower time is better, higher quality is better)
            speed_score = 1.0 - (stats['avg_render_time'] / max_speed)
            quality_score = (stats['avg_surface_pixels'] + stats['avg_unique_colors']) / max_quality
            combined_scores[method] = (speed_score + quality_score) / 2
        
        best_overall = max(combined_scores.items(), key=lambda x: x[1])
        print(f"Best overall: {best_overall[0]} (balanced speed + quality)")
    else:
        print(f"Single method available: {list(methods.keys())[0]}")

def create_visual_comparison(all_results, output_dir):
    """Create visual comparison charts"""
    if not all_results:
        return
    
    # Performance comparison chart
    methods = []
    render_times = []
    surface_pixels = []
    
    for method_name, results in all_results.items():
        if results:
            methods.append(method_name)
            render_times.append(np.mean([r['render_time'] for r in results]))
            surface_pixels.append(np.mean([r['surface_pixels'] for r in results]))
    
    if len(methods) > 1:
        # Render time comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.bar(methods, render_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_ylabel('Average Render Time (seconds)')
        ax1.set_title('Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Quality comparison  
        ax2.bar(methods, surface_pixels, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_ylabel('Average Surface Pixels')
        ax2.set_title('Quality Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visual comparison saved: {output_dir / 'method_comparison.png'}")

def save_results_summary(all_results, output_dir):
    """Save detailed results to JSON file"""
    summary = {
        'test_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'methods_tested': list(all_results.keys()),
            'total_renders': sum(len(results) for results in all_results.values())
        },
        'results': all_results
    }
    
    summary_file = output_dir / 'fractal_comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Detailed results saved: {summary_file}")

def main():
    """Run comprehensive Python 3D fractal rendering comparison"""
    print("🚀 Python 3D Fractal Rendering Comparison Suite")
    print("="*50)
    
    # Create output directory
    output_dir = Path("fractal_comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    # Check available packages
    print("\n📦 Checking available packages...")
    available, missing = install_missing_packages()
    
    # Run tests for each available method
    all_results = {}
    
    print(f"\n🧪 Running tests...")
    print(f"Output directory: {output_dir.absolute()}")
    
    # NumPy baseline (should always be available)
    print(f"\n" + "="*50)
    print("TESTING: NumPy + Matplotlib (Baseline)")
    print("="*50)
    try:
        numpy_results = test_numpy_mandelbulb()
        all_results['NumPy'] = numpy_results
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        all_results['NumPy'] = []
    
    # Taichi GPU acceleration
    print(f"\n" + "="*50)
    print("TESTING: Taichi GPU Acceleration")
    print("="*50)
    try:
        taichi_results = test_taichi_mandelbulb()
        all_results['Taichi'] = taichi_results
    except Exception as e:
        print(f"❌ Taichi test failed: {e}")
        all_results['Taichi'] = []
    
    # Numba JIT compilation
    print(f"\n" + "="*50)
    print("TESTING: Numba JIT Compilation") 
    print("="*50)
    try:
        numba_results = test_numba_mandelbulb()
        all_results['Numba'] = numba_results
    except Exception as e:
        print(f"❌ Numba test failed: {e}")
        all_results['Numba'] = []
    
    # Analyze and compare results
    analyze_results(all_results)
    
    # Create visual comparisons
    create_visual_comparison(all_results, output_dir)
    
    # Save detailed results
    save_results_summary(all_results, output_dir)
    
    # Final summary
    total_successful = sum(1 for results in all_results.values() if results)
    total_renders = sum(len(results) for results in all_results.values())
    
    print(f"\n🏁 TESTING COMPLETE")
    print(f"Methods tested: {len(all_results)}")
    print(f"Successful methods: {total_successful}")
    print(f"Total renders: {total_renders}")
    print(f"All outputs saved in: {output_dir.absolute()}")
    
    # Show directory contents
    print(f"\n📁 Generated test outputs:")
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            image_count = len(list(subdir.glob("*.png")))
            print(f"  {subdir.name}/: {image_count} images")
    
    comparison_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.json"))
    if comparison_files:
        print(f"  Comparison files: {len(comparison_files)}")

if __name__ == "__main__":
    main()