#!/usr/bin/env python3
"""
Simple test of Mandelbulber integration with evolution system.
Tests parameter generation, rendering, and variation.
"""

import sys
import os
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from renderers.mandelbulber.parameters import MandelbulberParameters
from renderers.mandelbulber.renderer import MandelbulberRenderer
from renderers.mandelbulber.templates import ParameterTemplates

def test_mandelbulber_evolution():
    """Test Mandelbulber with simple evolution-like workflow."""
    print("🌀 Testing Mandelbulber Evolution Integration")
    print("=" * 50)
    
    # Setup renderer
    renderer = MandelbulberRenderer(output_dir="./renders")
    if not renderer.is_available:
        print("❌ Mandelbulber not available")
        return False
    
    print("✅ Mandelbulber renderer available")
    
    # Test 1: Template rendering
    print("\n🧬 Generation 1: Templates")
    templates = ParameterTemplates.get_all_templates()
    
    template_results = []
    for name, template in list(templates.items())[:3]:  # Test 3 templates
        print(f"  Rendering template: {name}...")
        
        # Make smaller for faster testing
        template.render.image_width = 256
        template.render.image_height = 256
        
        output_path = renderer.render_single(
            template, 
            output_filename=f"template_{name}.png"
        )
        
        if output_path and output_path.exists():
            size = output_path.stat().st_size
            template_results.append((name, size))
            print(f"    ✅ {name}: {size} bytes")
        else:
            print(f"    ❌ {name}: Failed")
    
    print(f"Template success: {len(template_results)}/3")
    
    # Test 2: Mutation and crossover
    print(f"\n🧬 Generation 2: Mutations")
    
    if template_results:
        # Get the first successful template
        base_name = template_results[0][0]
        base_template = templates[base_name].copy()
        
        mutation_results = []
        for i in range(3):
            print(f"  Creating mutation {i+1}...")
            
            # Create mutated version
            mutated = base_template.copy()
            mutated.mutate(mutation_rate=0.3, mutation_strength=0.2)
            mutated.render.image_width = 256
            mutated.render.image_height = 256
            
            output_path = renderer.render_single(
                mutated,
                output_filename=f"mutation_{i+1}.png"
            )
            
            if output_path and output_path.exists():
                size = output_path.stat().st_size
                mutation_results.append((f"mutation_{i+1}", size))
                print(f"    ✅ mutation_{i+1}: {size} bytes")
            else:
                print(f"    ❌ mutation_{i+1}: Failed")
        
        print(f"Mutation success: {len(mutation_results)}/3")
    
    # Test 3: Random generation
    print(f"\n🧬 Generation 3: Random")
    
    random_results = []
    for i in range(2):
        print(f"  Creating random fractal {i+1}...")
        
        random_params = MandelbulberParameters()
        random_params.randomize()
        random_params.render.image_width = 256
        random_params.render.image_height = 256
        
        output_path = renderer.render_single(
            random_params,
            output_filename=f"random_{i+1}.png"
        )
        
        if output_path and output_path.exists():
            size = output_path.stat().st_size
            random_results.append((f"random_{i+1}", size))
            print(f"    ✅ random_{i+1}: {size} bytes")
        else:
            print(f"    ❌ random_{i+1}: Failed")
    
    print(f"Random success: {len(random_results)}/2")
    
    # Summary
    total_successful = len(template_results) + len(mutation_results) + len(random_results)
    total_attempted = 8
    
    print(f"\n📊 Overall Results:")
    print(f"✅ Successful renders: {total_successful}/{total_attempted}")
    
    if total_successful >= 6:
        print("🎉 Mandelbulber integration is working well!")
        
        # Check for diversity in file sizes
        all_results = template_results + mutation_results + random_results
        sizes = [size for _, size in all_results]
        
        if len(set(sizes)) > 1:
            print("🌈 Good diversity detected!")
            print("File sizes:", sizes)
        else:
            print("⚠️  All images are the same size - limited diversity")
            
        return True
    else:
        print("❌ Integration needs improvement")
        return False

if __name__ == "__main__":
    success = test_mandelbulber_evolution()
    if success:
        print(f"\n✨ Mandelbulber is ready for FractalGenesis evolution!")
    else:
        print(f"\n🔧 Mandelbulber integration needs fixes")